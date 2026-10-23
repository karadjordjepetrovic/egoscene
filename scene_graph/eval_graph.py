"""Build the scene graph along each walk and score it against ground-truth 3D objects.

Usage (from the repo root):
    python scene_graph/eval_graph.py --input gt                  # GT masks + GT depth (pipeline check)
    python scene_graph/eval_graph.py --input gtbox               # GT boxes -> depth masks + GT depth (CPU)
    python scene_graph/eval_graph.py --input model --fp16        # all model variants in one pass (GPU)

Input variants (each is one graph; --input model runs the last three together):
    gt                 GT instance masks + GT depth              upper bound, tests the graph logic
    gtbox              GT 2D boxes -> depth-based masks + GT depth   cost of lifting from boxes instead of masks
    owl_gtdepth        OWLv2 boxes -> masks + GT depth           cost of the detector
    gtmask_modeldepth  GT masks + DA V2 Metric (FOV-corrected)   cost of the depth model
    owl_modeldepth     OWLv2 boxes + DA V2 Metric (FOV-corrected) the real, ground-truth-free pipeline
    *_aligned          same, with temporal scale alignment of the depth (ScaleAligner: each frame's
                       depth rescaled to agree with the previous one, using the known camera motion)
Poses are ground truth in every variant (no SLAM in v1).

Ground-truth objects: every visible "thing" object (11 classes, as in the detection benchmark) fused
from all processed frames where it covers >= 300 px and seen in >= 3 of them; box from robust percentiles.
A graph object (seen >= 3 times) matches a GT object of the same class with 3D IoU >= --iou (greedy).

Outputs in eval/results/scene_graph/<variant>/: summary.md, summary.json, graphs/<walk>.json,
plus eval/results/scene_graph/comparison.md when several variants run together.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "eval"))
from dataset.egoscene import list_walks, load_meta, load_frame   # noqa: E402
from scene_graph.graph import SceneGraph, Detection, aabb, box_iou_3d, mask_from_box, ScaleAligner   # noqa: E402

THINGS = ["table", "chair", "sofa", "shelf", "plant", "lamp", "counter", "sink", "stove", "tv", "sign"]
MODEL_IMPLIED_FOV = 74.0
VARIANTS = {"gt": ["gt"], "gtbox": ["gtbox"],
            "model": ["owl_gtdepth", "gtmask_modeldepth", "owl_modeldepth",
                      "gtmask_modeldepth_aligned", "owl_modeldepth_aligned"]}


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
            d = Detection(cls=cls, mask=m, score=1.0)
            d.actor = actor
            dets.append(d)
    return dets


def boxes_to_dets(boxes, depth):
    """[(cls, score, box)] -> Detections with depth-based masks."""
    return [Detection(cls=c, mask=mask_from_box(b, depth), score=s) for c, s, b in boxes]


def score_graph(sg, gts, iou_thr):
    preds = [o for o in sg.objects if o.n_obs >= 3]          # same visibility rule as the GT objects
    cand = sorted(((box_iou_3d(o.box_min, o.box_max, g["min"], g["max"]), pi, gi)
                   for pi, o in enumerate(preds) for gi, g in enumerate(gts) if o.cls == g["cls"]), reverse=True)
    mp, mg = {}, {}
    for iou, pi, gi in cand:
        if iou < iou_thr:
            break
        if pi not in mp and gi not in mg:
            mp[pi] = gi; mg[gi] = pi
    dup = sum(1 for pi, o in enumerate(preds) if pi not in mp and any(
        o.cls == gts[gi]["cls"] and box_iou_3d(o.box_min, o.box_max, gts[gi]["min"], gts[gi]["max"]) > 0.05 for gi in mg))
    ious = [box_iou_3d(preds[pi].box_min, preds[pi].box_max, gts[gi]["min"], gts[gi]["max"]) for pi, gi in mp.items()]
    cent = [float(np.linalg.norm(preds[pi].centroid - (gts[gi]["min"] + gts[gi]["max"]) / 2)) for pi, gi in mp.items()]
    size = [float(np.mean(np.abs(preds[pi].size - (gts[gi]["max"] - gts[gi]["min"])))) for pi, gi in mp.items()]
    per_cls = {}
    for gi, g in enumerate(gts):
        pc = per_cls.setdefault(g["cls"], [0, 0]); pc[1] += 1; pc[0] += gi in mg
    return {"gt_objects": len(gts), "pred_objects": len(preds), "matched": len(mp), "duplicates": dup,
            "false_positives": len(preds) - len(mp) - dup,
            "recall": len(mp) / max(len(gts), 1), "precision": len(mp) / max(len(preds), 1),
            "ious": ious, "cent": cent, "size": size, "per_class": per_cls,
            "missed": [gts[gi]["actor"] for gi in range(len(gts)) if gi not in mg]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="gt", choices=list(VARIANTS))
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--walks", nargs="*")
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame (3 = 10 Hz)")
    ap.add_argument("--min-px", type=int, default=300)
    ap.add_argument("--iou", type=float, default=0.25)
    ap.add_argument("--score-thr", type=float, default=0.2, help="OWLv2 detection threshold")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "scene_graph"))
    args = ap.parse_args()
    variants = VARIANTS[args.input]

    det = depth_model = None
    if args.input == "model":
        import torch
        import depth_benchmark as DB
        import det_benchmark as TB
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = args.fp16 and device == "cuda"
        depth_model = DB.HFDepthModel(DB.MODELS["da2-metric-indoor-small"][0], fp16, device)
        det = TB.OWLv2(TB.MODELS["owlv2-base"], [TB.THING_PROMPTS[c] for c in THINGS], fp16, device)

    walks = [w for w in list_walks(args.data) if not args.walks or w["walk_id"] in args.walks]
    res = {v: {"per_walk": [], "timings": {"lift": [], "associate": [], "update": []}} for v in variants}
    model_ms = {"depth": [], "owlv2": []}
    t_start = time.time()
    for w in walks:
        wdir = os.path.join(args.data, w["path"]); meta = load_meta(wdir)
        K, groups, fps = K_of(meta), object_groups(meta), meta.get("fps", 30)
        fov_factor = float(np.tan(np.radians(MODEL_IMPLIED_FOV) / 2) / ((meta["width"] / 2) / K[0, 0]))
        graphs = {v: SceneGraph() for v in variants}
        aligner = ScaleAligner() if args.input == "model" else None
        gt_pts = {}
        ref = SceneGraph()                                   # only used for its lift()
        for i in range(0, len(meta["frames"]), args.stride):
            fr = load_frame(wdir, i, meta)
            t = fr["frame"] / fps
            gdets = gt_detections(fr, groups, args.min_px)
            for d in gdets:
                p = ref.lift(d.mask, fr["depth"], K, fr["c2w"])
                if p is not None:
                    gt_pts.setdefault(d.actor, []).append(p)
            inputs = {}
            if "gt" in variants:
                inputs["gt"] = (gdets, fr["depth"])
            if "gtbox" in variants:
                boxes = []
                for d in gdets:
                    ys, xs = np.nonzero(d.mask)
                    boxes.append((d.cls, 1.0, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]))
                inputs["gtbox"] = (boxes_to_dets(boxes, fr["depth"]), fr["depth"])
            if args.input == "model":
                n0 = len(depth_model.times_ms)
                mdepth = (depth_model(fr["rgb"]) * fov_factor).astype(np.float32)
                owl = det(fr["rgb"], args.score_thr)
                model_ms["depth"].append(depth_model.times_ms[-1]); model_ms["owlv2"].append(det.times_ms[-1])
                boxes = [(THINGS[l], s, b) for s, l, b in owl]
                inputs["owl_gtdepth"] = (boxes_to_dets(boxes, fr["depth"]), fr["depth"])
                inputs["gtmask_modeldepth"] = (gdets, mdepth)
                inputs["owl_modeldepth"] = (boxes_to_dets(boxes, mdepth), mdepth)
                adepth = aligner(mdepth, K, fr["c2w"]).astype(np.float32)
                inputs["gtmask_modeldepth_aligned"] = (gdets, adepth)
                inputs["owl_modeldepth_aligned"] = (boxes_to_dets(boxes, adepth), adepth)
            for v in variants:
                dets, depth = inputs[v]
                graphs[v].update(dets, depth, K, fr["c2w"], t)

        gts = []
        for actor, plist in gt_pts.items():
            if len(plist) >= 3:
                bmin, bmax = aabb(np.concatenate(plist))
                gts.append({"actor": actor, "cls": groups[actor][0], "min": bmin, "max": bmax})
        line = []
        for v in variants:
            r = score_graph(graphs[v], gts, args.iou)
            r.update(walk_id=w["walk_id"], lighting=w["lighting"])
            res[v]["per_walk"].append(r)
            for k in res[v]["timings"]:
                res[v]["timings"][k] += graphs[v].timings[k]
            vdir = os.path.join(args.out, v, "graphs"); os.makedirs(vdir, exist_ok=True)
            json.dump({"walk_id": w["walk_id"], "variant": v, "objects": graphs[v].snapshot(min_obs=3)},
                      open(os.path.join(vdir, f"{w['walk_id']}.json"), "w"), indent=1)
            line.append(f"{v}: {r['matched']}/{r['gt_objects']} (+{r['pred_objects'] - r['matched']} extra)")
        print(f"== {w['walk_id']} ({w['lighting']}): " + " | ".join(line) + f"  [{time.time() - t_start:.0f}s]", flush=True)

    comp = []
    for v in variants:
        pw = res[v]["per_walk"]
        cat = lambda k: sum((r[k] for r in pw), [])
        tot = lambda k: sum(r[k] for r in pw)
        tm = res[v]["timings"]
        total_ms = [a + b + c for a, b, c in zip(tm["lift"], tm["associate"], tm["update"])]
        per_cls = {}
        for r in pw:
            for c, (m, n) in r["per_class"].items():
                pc = per_cls.setdefault(c, [0, 0]); pc[0] += m; pc[1] += n
        o = {"recall": tot("matched") / max(tot("gt_objects"), 1), "precision": tot("matched") / max(tot("pred_objects"), 1),
             "gt_objects": tot("gt_objects"), "pred_objects": tot("pred_objects"), "duplicates": tot("duplicates"),
             "false_positives": tot("false_positives"),
             "mean_iou": float(np.mean(cat("ious"))) if cat("ious") else 0.0,
             "centroid_err_cm": float(np.mean(cat("cent")) * 100) if cat("cent") else float("nan"),
             "size_err_cm": float(np.mean(cat("size")) * 100) if cat("size") else float("nan"),
             "recall_by_class": {c: m / n for c, (m, n) in per_cls.items()},
             "graph_ms": {"median": float(np.median(total_ms)), "p95": float(np.percentile(total_ms, 95))}}
        if args.input == "model":
            o["model_ms"] = {k: float(np.median(x)) for k, x in model_ms.items()}
        for r in pw:
            for k in ("ious", "cent", "size"):
                r.pop(k)
        vdir = os.path.join(args.out, v)
        json.dump({"variant": v, "stride": args.stride, "iou_thr": args.iou, "overall": o, "per_walk": pw},
                  open(os.path.join(vdir, "summary.json"), "w"), indent=1)
        L = [f"# Scene graph vs ground truth: `{v}`", "",
             f"Every {args.stride}th frame ({30 / args.stride:.0f} Hz) · match = same class and 3D IoU ≥ {args.iou} · GT poses", "",
             f"**Recall {o['recall']*100:.1f} % · precision {o['precision']*100:.1f} % · {o['duplicates']} duplicates · "
             f"{o['false_positives']} false positives · 3D IoU {o['mean_iou']:.2f} · centroid error {o['centroid_err_cm']:.1f} cm · "
             f"size error {o['size_err_cm']:.1f} cm**", "",
             f"**Graph update:** {o['graph_ms']['median']:.2f} ms/frame median, {o['graph_ms']['p95']:.2f} ms p95 (CPU, numpy)"
             + (f" · models: depth {o['model_ms']['depth']:.0f} ms, OWLv2 {o['model_ms']['owlv2']:.0f} ms" if "model_ms" in o else ""), "",
             "| Class | Recall (%) |", "|---|---|"]
        L += [f"| {c} | {x*100:.0f} |" for c, x in sorted(o["recall_by_class"].items(), key=lambda kv: kv[1])]
        L += ["", "| Walk | Lighting | GT objects | In graph | Matched | Dup. | False pos. | Missed |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['walk_id']} | {r['lighting']} | {r['gt_objects']} | {r['pred_objects']} | {r['matched']} | {r['duplicates']} | "
              f"{r['false_positives']} | {', '.join(r['missed']) or '-'} |" for r in pw]
        open(os.path.join(vdir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        comp.append((v, o))
        print("\n".join(L[:6]))
    if len(comp) > 1:
        C = ["| Variant | Recall (%) | Precision (%) | 3D IoU | Centroid err (cm) | Size err (cm) |", "|---|---|---|---|---|---|"]
        C += [f"| `{v}` | {o['recall']*100:.1f} | {o['precision']*100:.1f} | {o['mean_iou']:.2f} | {o['centroid_err_cm']:.1f} | {o['size_err_cm']:.1f} |"
              for v, o in comp]
        open(os.path.join(args.out, "comparison.md"), "w", encoding="utf-8").write("\n".join(C) + "\n")
        print("\n".join(C))
    print(f"Done ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
