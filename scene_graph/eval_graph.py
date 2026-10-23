"""Build the scene graph along each walk and score it against ground-truth 3D objects.

Usage (from the repo root):
    python scene_graph/eval_graph.py --input gt                 # GT masks + GT depth (pipeline check)
    python scene_graph/eval_graph.py --input gt --walks walk_04

Inputs (--input):
    gt    ground-truth instance masks (one per object) + ground-truth depth + GT poses

Ground-truth objects: for each walk, every visible "thing" object (same 11 classes as the detection
benchmark), fused from all frames where it covers >= 300 px, box from robust percentiles.
A predicted object matches a GT object of the same class with 3D IoU >= --iou (greedy).

Metrics per walk and overall: object recall / precision, duplicates (extra predicted objects of the
same class overlapping an already matched GT object), mean 3D IoU, centroid error (cm), size error
(cm), and per-stage update time (ms per frame).
Outputs in eval/results/scene_graph/<input>/: summary.md, summary.json, graphs/<walk>.json
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from dataset.egoscene import list_walks, load_meta, load_frame   # noqa: E402
from scene_graph.graph import SceneGraph, Detection, aabb, box_iou_3d   # noqa: E402

THINGS = ["table", "chair", "sofa", "shelf", "plant", "lamp", "counter", "sink", "stove", "tv", "sign"]


def K_of(meta):
    k = meta["intrinsics"]
    return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])


def object_groups(meta):
    """actor -> (class, [instance ids]) for thing classes (multi-part objects share an actor)."""
    g = {}
    for iid, v in meta["instances"].items():
        if v["class"] in THINGS:
            g.setdefault(v.get("actor", iid), (v["class"], []))[1].append(int(iid))
    return g


def gt_detections(fr, groups, min_px):
    dets = []
    for actor, (cls, ids) in groups.items():
        m = np.isin(fr["instance"], ids)
        if m.sum() >= min_px:
            dets.append(Detection(cls=cls, mask=m, score=1.0, track_id=None))
            dets[-1].actor = actor
    return dets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="gt", choices=["gt"])
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--walks", nargs="*")
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame (3 = 10 Hz)")
    ap.add_argument("--min-px", type=int, default=300)
    ap.add_argument("--iou", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "scene_graph"))
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.input)
    os.makedirs(os.path.join(out_dir, "graphs"), exist_ok=True)
    walks = [w for w in list_walks(args.data) if not args.walks or w["walk_id"] in args.walks]
    per_walk, totals = [], {"tp": 0, "gt": 0, "pred": 0, "dup": 0, "iou": [], "cent": [], "size": []}
    timings = {"lift": [], "associate": [], "update": []}
    t_start = time.time()
    for w in walks:
        wdir = os.path.join(args.data, w["path"]); meta = load_meta(wdir)
        K, groups, fps = K_of(meta), object_groups(meta), meta.get("fps", 30)
        sg = SceneGraph()
        gt_pts = {}                                   # actor -> list of point arrays (all frames)
        for i in range(0, len(meta["frames"]), args.stride):
            fr = load_frame(wdir, i, meta)
            dets = gt_detections(fr, groups, args.min_px)
            for d in dets:                            # GT reference uses the same lifting
                p = sg.lift(d.mask, fr["depth"], K, fr["c2w"])
                if p is not None:
                    gt_pts.setdefault(d.actor, []).append(p)
            if args.input == "gt":
                sg.update(dets, fr["depth"], K, fr["c2w"], fr["frame"] / fps)
        for k in timings:
            timings[k] += sg.timings[k]

        # ground-truth objects: seen in >= 3 processed frames
        gts = []
        for actor, plist in gt_pts.items():
            if len(plist) >= 3:
                P = np.concatenate(plist)
                bmin, bmax = aabb(P)
                gts.append({"actor": actor, "cls": groups[actor][0], "min": bmin, "max": bmax})
        preds = [o for o in sg.objects if o.n_obs >= 3]          # same visibility rule as the GT objects

        # greedy matching by IoU
        cand = sorted(((box_iou_3d(o.box_min, o.box_max, g["min"], g["max"]), pi, gi)
                       for pi, o in enumerate(preds) for gi, g in enumerate(gts) if o.cls == g["cls"]), reverse=True)
        mp, mg = {}, {}
        for iou, pi, gi in cand:
            if iou < args.iou: break
            if pi not in mp and gi not in mg:
                mp[pi] = gi; mg[gi] = pi
        dup = sum(1 for pi, o in enumerate(preds) if pi not in mp and any(
            o.cls == gts[gi]["cls"] and box_iou_3d(o.box_min, o.box_max, gts[gi]["min"], gts[gi]["max"]) > 0.05 for gi in mg))
        ious = [box_iou_3d(preds[pi].box_min, preds[pi].box_max, gts[gi]["min"], gts[gi]["max"]) for pi, gi in mp.items()]
        cent = [float(np.linalg.norm(preds[pi].centroid - (gts[gi]["min"] + gts[gi]["max"]) / 2)) for pi, gi in mp.items()]
        size = [float(np.mean(np.abs(preds[pi].size - (gts[gi]["max"] - gts[gi]["min"])))) for pi, gi in mp.items()]
        r = {"walk_id": w["walk_id"], "lighting": w["lighting"], "gt_objects": len(gts), "pred_objects": len(preds),
             "matched": len(mp), "duplicates": dup,
             "recall": len(mp) / max(len(gts), 1), "precision": len(mp) / max(len(preds), 1),
             "mean_iou": float(np.mean(ious)) if ious else 0.0, "centroid_err_cm": float(np.mean(cent) * 100) if cent else None,
             "size_err_cm": float(np.mean(size) * 100) if size else None,
             "missed": [gts[gi]["actor"] for gi in range(len(gts)) if gi not in mg]}
        per_walk.append(r)
        totals["tp"] += len(mp); totals["gt"] += len(gts); totals["pred"] += len(preds); totals["dup"] += dup
        totals["iou"] += ious; totals["cent"] += cent; totals["size"] += size
        json.dump({"walk_id": w["walk_id"], "objects": sg.snapshot(min_obs=3)},
                  open(os.path.join(out_dir, "graphs", f"{w['walk_id']}.json"), "w"), indent=1)
        print(f"== {w['walk_id']}: {r['matched']}/{r['gt_objects']} GT objects matched, {r['pred_objects']} in graph, "
              f"{dup} duplicates, IoU {r['mean_iou']:.2f}, centroid err {r['centroid_err_cm'] or 0:.1f} cm", flush=True)

    ms = {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))} for k, v in timings.items() if v}
    total_ms = [a + b + c for a, b, c in zip(timings["lift"], timings["associate"], timings["update"])]
    overall = {"recall": totals["tp"] / max(totals["gt"], 1), "precision": totals["tp"] / max(totals["pred"], 1),
               "duplicates": totals["dup"], "gt_objects": totals["gt"], "pred_objects": totals["pred"],
               "mean_iou": float(np.mean(totals["iou"])), "centroid_err_cm": float(np.mean(totals["cent"]) * 100),
               "size_err_cm": float(np.mean(totals["size"]) * 100),
               "ms_per_frame": ms, "ms_total": {"median": float(np.median(total_ms)), "p95": float(np.percentile(total_ms, 95))}}
    json.dump({"input": args.input, "stride": args.stride, "iou_thr": args.iou, "overall": overall, "per_walk": per_walk},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    o = overall
    L = [f"# Scene graph vs ground truth: `{args.input}` inputs", "",
         f"Every {args.stride}th frame ({30 / args.stride:.0f} Hz) · object match = same class and 3D IoU ≥ {args.iou}", "",
         f"**Recall {o['recall']*100:.1f} % · precision {o['precision']*100:.1f} % · {o['duplicates']} duplicates · "
         f"mean 3D IoU {o['mean_iou']:.2f} · centroid error {o['centroid_err_cm']:.1f} cm · size error {o['size_err_cm']:.1f} cm**", "",
         f"**Update time:** {o['ms_total']['median']:.2f} ms/frame median, {o['ms_total']['p95']:.2f} ms p95 "
         f"(lift {ms['lift']['median']:.2f} · associate {ms['associate']['median']:.2f} · update {ms['update']['median']:.2f}; CPU, numpy)", "",
         "| Walk | Lighting | GT objects | In graph | Matched | Duplicates | 3D IoU | Centroid err (cm) | Missed |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in per_walk:
        L.append(f"| {r['walk_id']} | {r['lighting']} | {r['gt_objects']} | {r['pred_objects']} | {r['matched']} | "
                 f"{r['duplicates']} | {r['mean_iou']:.2f} | {r['centroid_err_cm'] or 0:.1f} | {', '.join(r['missed']) or '-'} |")
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))
    print(f"Wrote {out_dir} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
