"""Score the room layout (floor, walls) and support relations against ground truth.

Usage (from the repo root):
    python scene_graph/eval_layout.py --input gt              # GT depth + GT labels (pipeline check, CPU)
    python scene_graph/eval_layout.py --input model --fp16    # DA V2 Metric (FOV-corrected) + SegFormer-B2

Floor: RANSAC plane on floor-labelled points; scored by height error (GT floor is z = 0) and tilt.
Walls: main wall lines on the top-down map; reference = the same estimator on GT depth + GT labels;
       a reference wall (>= 0.5 m) is found if an estimated wall is within 10 deg and 0.3 m.
Eye height: camera height above the estimated floor (GT rig: 1.60 m +- head bob).
Relations: support relations (floor / on <class> / wall / free) of the scene-graph objects, from the
       graphs saved by eval_graph.py (gt and owl_modeldepth variants), compared on matched objects.
Outputs: eval/results/layout/<input>/summary.md, summary.json, <walk>_layout.png
"""
import argparse, json, os, sys, time
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "eval"))
from dataset.egoscene import list_walks, load_meta, load_frame, backproject   # noqa: E402
from scene_graph.layout import estimate_floor, estimate_walls, support_relations   # noqa: E402
from scene_graph.graph import box_iou_3d                                          # noqa: E402
from scene_graph.eval_graph import K_of, MODEL_IMPLIED_FOV                         # noqa: E402


def collect(frames_iter, K, floor_ids, wall_ids, per_frame=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    F, Wp, cams = [], [], []
    for depth, sem, c2w in frames_iter:
        cams.append(c2w[:3, 3])
        for ids, out in ((floor_ids, F), (wall_ids, Wp)):
            p, _ = backproject(depth, K, c2w, mask=np.isin(sem, ids))
            if len(p):
                out.append(p[rng.choice(len(p), min(per_frame, len(p)), replace=False)])
    return (np.concatenate(F) if F else np.zeros((0, 3))), (np.concatenate(Wp) if Wp else np.zeros((0, 3))), np.array(cams)


def match_walls(ref, est, max_angle=10, max_off=0.3, min_len=0.5):
    out = []
    for r in ref:
        if r["length"] < min_len:
            continue
        p0, p1 = np.array(r["p0"]), np.array(r["p1"]); mid = (p0 + p1) / 2; dr = (p1 - p0) / np.linalg.norm(p1 - p0)
        best = None
        for e in est:
            q0, q1 = np.array(e["p0"]), np.array(e["p1"]); de = (q1 - q0) / np.linalg.norm(q1 - q0)
            ang = np.degrees(np.arccos(np.clip(abs(dr @ de), 0, 1)))
            nrm = np.array([-de[1], de[0]]); off = abs((mid - q0) @ nrm)
            if ang <= max_angle and off <= max_off and (best is None or off < best[1]):
                best = (ang, off)
        out.append({"length": r["length"], "found": best is not None,
                    "angle_err_deg": best[0] if best else None, "offset_cm": best[1] * 100 if best else None})
    return out


def load_graph(path):
    if not os.path.exists(path):
        return None
    objs = []
    for o in json.load(open(path))["objects"]:
        c, s = np.array(o["centroid"]), np.array(o["size"])
        objs.append(SimpleNamespace(id=o["id"], cls=o["class"], box_min=c - s / 2, box_max=c + s / 2, surface=o["surface"]))
    return objs


def rel_type(r):
    return r.split("#")[0]          # "on counter#8" -> "on counter"


def plot(path, ref_walls, est_walls, floor_pts, wall_pts, cams, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 7))
    sel = np.random.default_rng(0).choice(len(wall_pts), min(len(wall_pts), 20000), replace=False)
    ax.scatter(wall_pts[sel, 0], wall_pts[sel, 1], s=0.2, c="#bbb")
    for w in ref_walls:
        ax.plot([w["p0"][0], w["p1"][0]], [w["p0"][1], w["p1"][1]], c="#2a9d4b", lw=5, alpha=0.35)
    for w in est_walls:
        ax.plot([w["p0"][0], w["p1"][0]], [w["p0"][1], w["p1"][1]], c="#1f5fbf", lw=1.8)
    ax.plot(cams[:, 0], cams[:, 1], c="#d33", lw=1)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.plot([], [], c="#2a9d4b", lw=5, alpha=0.35, label="walls from ground truth")
    ax.plot([], [], c="#1f5fbf", lw=1.8, label="estimated walls"); ax.plot([], [], c="#d33", label="camera path")
    ax.legend(loc="lower right", fontsize=8); plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="gt", choices=["gt", "model"])
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--walks", nargs="*")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "layout"))
    args = ap.parse_args()
    out_dir = os.path.join(args.out, args.input); os.makedirs(out_dir, exist_ok=True)

    depth_model = seg = None
    if args.input == "model":
        import torch
        import depth_benchmark as DB, seg_benchmark as SB
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = args.fp16 and device == "cuda"
        depth_model = DB.HFDepthModel(DB.MODELS["da2-metric-indoor-small"][0], fp16, device)
    walks = [w for w in list_walks(args.data) if not args.walks or w["walk_id"] in args.walks]
    rows, t0 = [], time.time()
    for w in walks:
        wdir = os.path.join(args.data, w["path"]); meta = load_meta(wdir); K = K_of(meta)
        classes = meta["classes"]
        floor_ids, wall_ids = [classes.index("floor")], [classes.index("wall")]
        if args.input == "model" and seg is None:
            seg = SB.SegFormerModel(SB.MODELS["segformer-b2-ade"][0], classes, fp16, device)
        fov_factor = float(np.tan(np.radians(MODEL_IMPLIED_FOV) / 2) / ((meta["width"] / 2) / K[0, 0]))
        idx = list(range(0, len(meta["frames"]), args.stride))
        frames = [load_frame(wdir, i, meta) for i in idx]
        gtF, gtW, cams = collect(((f["depth"], f["semantic"], f["c2w"]) for f in frames), K, floor_ids, wall_ids)
        gt_floor = {"height": 0.0, "normal": [0, 0, 1]}
        ref_walls = estimate_walls(gtW, gt_floor)
        if args.input == "gt":
            F, Wp = gtF, gtW
        else:
            it = ((depth_model(f["rgb"]) * fov_factor, seg(f["rgb"]), f["c2w"]) for f in frames)
            F, Wp, _ = collect(it, K, floor_ids, wall_ids)
        floor = estimate_floor(F)
        walls = estimate_walls(Wp, floor)
        wm = match_walls(ref_walls, walls)
        eye = float(np.median(cams[:, 2]) - floor["height"])
        r = {"walk_id": w["walk_id"], "lighting": w["lighting"], "floor_height_err_cm": abs(floor["height"]) * 100,
             "floor_tilt_deg": floor["tilt_deg"], "eye_height_m": eye, "walls_ref": len(wm),
             "walls_found": sum(x["found"] for x in wm), "walls_est": len(walls),
             "wall_angle_err_deg": float(np.mean([x["angle_err_deg"] for x in wm if x["found"]])) if any(x["found"] for x in wm) else None,
             "wall_offset_cm": float(np.mean([x["offset_cm"] for x in wm if x["found"]])) if any(x["found"] for x in wm) else None}
        # support relations from the saved scene graphs
        g_gt = load_graph(os.path.join(REPO, "eval", "results", "scene_graph", "gt", "graphs", f"{w['walk_id']}.json"))
        g_m = load_graph(os.path.join(REPO, "eval", "results", "scene_graph", "owl_modeldepth", "graphs", f"{w['walk_id']}.json"))
        if g_gt:
            rel_gt = support_relations(g_gt, {"height": 0.0})
            r["relations_gt"] = {f"{o.cls}#{o.id}": rel_gt[o.id] for o in g_gt}
            if g_m and args.input == "model":
                rel_m = support_relations(g_m, floor)
                cand = sorted(((box_iou_3d(a.box_min, a.box_max, b.box_min, b.box_max), i, j)
                               for i, a in enumerate(g_gt) for j, b in enumerate(g_m) if a.cls == b.cls), reverse=True)
                ui, uj, agree, n = set(), set(), 0, 0
                for iou, i, j in cand:
                    if iou < 0.25: break
                    if i in ui or j in uj: continue
                    ui.add(i); uj.add(j); n += 1
                    agree += rel_type(rel_gt[g_gt[i].id]) == rel_type(rel_m[g_m[j].id])
                r["relation_agreement"] = agree / n if n else None
                r["relation_pairs"] = n
        rows.append(r)
        plot(os.path.join(out_dir, f"{w['walk_id']}_layout.png"), ref_walls, walls, F, Wp, cams,
             f"{w['walk_id']} ({w['lighting']}), {args.input} inputs: {r['walls_found']}/{r['walls_ref']} walls, "
             f"floor err {r['floor_height_err_cm']:.1f} cm")
        print(f"== {w['walk_id']}: floor err {r['floor_height_err_cm']:.1f} cm, tilt {r['floor_tilt_deg']:.2f} deg, "
              f"eye {eye:.2f} m, walls {r['walls_found']}/{r['walls_ref']} (offset {r['wall_offset_cm'] or 0:.1f} cm)  [{time.time() - t0:.0f}s]", flush=True)

    mean = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None
    o = {"floor_height_err_cm": mean("floor_height_err_cm"), "floor_tilt_deg": mean("floor_tilt_deg"),
         "eye_height_m": mean("eye_height_m"), "walls_recall": sum(r["walls_found"] for r in rows) / max(sum(r["walls_ref"] for r in rows), 1),
         "wall_angle_err_deg": mean("wall_angle_err_deg"), "wall_offset_cm": mean("wall_offset_cm"),
         "relation_agreement": mean("relation_agreement")}
    json.dump({"input": args.input, "overall": o, "per_walk": rows}, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    fmt = lambda v, f: "-" if v is None else format(v, f)
    L = [f"# Room layout vs ground truth: `{args.input}` inputs", "",
         f"**Floor height error {fmt(o['floor_height_err_cm'], '.1f')} cm · tilt {fmt(o['floor_tilt_deg'], '.2f')}° · "
         f"eye height {fmt(o['eye_height_m'], '.2f')} m (rig: 1.60) · walls found {o['walls_recall']*100:.0f} % · "
         f"wall angle error {fmt(o['wall_angle_err_deg'], '.1f')}° · wall offset {fmt(o['wall_offset_cm'], '.1f')} cm"
         + (f" · support relations agree {o['relation_agreement']*100:.0f} %" if o["relation_agreement"] is not None else "") + "**", "",
         "| Walk | Lighting | Floor err (cm) | Tilt (°) | Eye height (m) | Walls found | Wall offset (cm) |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['walk_id']} | {r['lighting']} | {r['floor_height_err_cm']:.1f} | {r['floor_tilt_deg']:.2f} | {r['eye_height_m']:.2f} | "
          f"{r['walls_found']}/{r['walls_ref']} | {fmt(r['wall_offset_cm'], '.1f')} |" for r in rows]
    if rows and rows[0].get("relations_gt"):
        L += ["", f"Support relations in `{rows[0]['walk_id']}` (scene graph from GT inputs):", ""]
        L += [f"- {k}: {v}" for k, v in rows[0]["relations_gt"].items()]
    L += ["", f"![layout](./{rows[0]['walk_id']}_layout.png)"]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:3])); print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
