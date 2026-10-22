"""EgoScene object detection benchmark: open-vocabulary detectors vs ground-truth boxes.

Usage (from the repo root, venv active):
    python eval/det_benchmark.py --model owlv2-base --fp16
    python eval/det_benchmark.py --model gt-jitter          # no GPU: sanity check of this script

Ground truth: one box per visible object instance, from the instance masks (tight box around the
instance's visible pixels). "Stuff" classes (wall, floor, ceiling, outdoor) and door/window are not
detected. Boxes smaller than --min-box pixels on either side are ignored (neither TP nor FP).

Metrics: AP@0.5 per class (COCO-style, 101-point interpolation), mAP over classes, recall at
--score-thr by object distance (median GT depth inside the instance) and by box size, per lighting,
and model forward ms/frame. Note: counter and sink are textured blockouts in v1, not real models.

Outputs in eval/results/det/<model>/: summary.md, summary.json, figures/.
"""
import argparse, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataset.egoscene import list_walks, load_meta, load_frame  # noqa: E402

THING_PROMPTS = {
    "table": "a table", "chair": "a chair", "sofa": "a sofa", "shelf": "a shelf",
    "plant": "a potted plant", "lamp": "a lamp", "counter": "a kitchen counter", "sink": "a sink",
    "stove": "a stove", "tv": "a television", "sign": "a sign with text",
}
DIST_BINS = [(0, 1), (1, 2), (2, 4), (4, 100)]
SIZE_BINS = [("small <32²", 0, 32 ** 2), ("medium <96²", 32 ** 2, 96 ** 2), ("large", 96 ** 2, 1e12)]


def gt_boxes(instance, semantic, depth, inst_class, thing_ids, min_box, inst_actor=None):
    """-> list of dicts: class id, box [x0,y0,x1,y1], depth, area, ignore.
    Instance ids that belong to the same actor (e.g. the counter's 6 parts) form one object."""
    out = []
    groups = {}
    for iid in np.unique(instance):
        c = inst_class.get(int(iid))
        if c is None or c not in thing_ids:
            continue
        key = (inst_actor or {}).get(int(iid), int(iid))
        groups.setdefault((key, c), []).append(int(iid))
    for (key, c), ids in groups.items():
        ys, xs = np.nonzero(np.isin(instance, ids))
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        d = depth[ys, xs]; d = d[d > 0]
        out.append({"cls": c, "box": np.array([x0, y0, x1, y1], float), "area": float((x1 - x0) * (y1 - y0)),
                    "depth": float(np.median(d)) if len(d) else 0.0,
                    "ignore": (x1 - x0) < min_box or (y1 - y0) < min_box})
    return out


def iou(a, b):
    """a: (4,), b: (N,4) -> (N,)"""
    x0 = np.maximum(a[0], b[:, 0]); y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2]); y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def match(dets, gts, cls, thr=0.5):
    """Greedy COCO-style matching for one class in one frame. dets: list of (score, box).
    Returns list of (score, is_tp or None-if-ignored), and per-GT matched flags."""
    g = [x for x in gts if x["cls"] == cls]
    gb = np.array([x["box"] for x in g]) if g else np.zeros((0, 4))
    used = np.zeros(len(g), bool)
    res = []
    for score, box in sorted(dets, key=lambda t: -t[0]):
        if len(g):
            ious = iou(box, gb)
            # prefer unused non-ignored GT
            order = np.argsort(-ious)
            hit = None
            for j in order:
                if ious[j] < thr:
                    break
                if not used[j] and not g[j]["ignore"]:
                    hit = j; break
            if hit is None:
                ig = [j for j in order if ious[j] >= thr and g[j]["ignore"]]
                if ig:
                    res.append((score, None)); continue
                res.append((score, False)); continue
            used[hit] = True; res.append((score, True))
        else:
            res.append((score, False))
    return res, g, used


def average_precision(scored, n_gt):
    if n_gt == 0:
        return float("nan")
    scored = [s for s in scored if s[1] is not None]
    if not scored:
        return 0.0
    scored.sort(key=lambda t: -t[0])
    tp = np.cumsum([1 if t else 0 for _, t in scored]); fp = np.cumsum([0 if t else 1 for _, t in scored])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))


# ----------------------------------------------------------------------------- models
class OWLv2:
    def __init__(self, hf_id, prompts, fp16, device):
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.torch, self.device, self.times_ms = torch, device, []
        self.dtype = torch.float16 if fp16 else torch.float32
        self.processor = Owlv2Processor.from_pretrained(hf_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(hf_id).to(device, self.dtype).eval()
        self.prompts = prompts

    def __call__(self, rgb, thr):
        torch = self.torch
        inp = self.processor(text=[self.prompts], images=rgb, return_tensors="pt")
        inp = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)) for k, v in inp.items()}
        self.input_hw = tuple(inp["pixel_values"].shape[-2:])
        with torch.inference_mode():
            if self.device == "cuda":
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); out = self.model(**inp); e.record(); torch.cuda.synchronize()
                self.times_ms.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter(); out = self.model(**inp); self.times_ms.append((time.perf_counter() - t0) * 1000)
        side = max(rgb.shape[:2])          # OWLv2 pads to a square: boxes are relative to the padded image
        sizes = torch.tensor([[side, side]], device=self.device)
        # newer transformers renamed it (post_process_grounded_object_detection); older ones have the old name
        pp = getattr(self.processor, "post_process_grounded_object_detection", None)
        if pp is not None:
            r = pp(outputs=out, threshold=thr, target_sizes=sizes)[0]
        else:
            pp = getattr(self.processor, "post_process_object_detection", None) or \
                 self.processor.image_processor.post_process_object_detection
            r = pp(outputs=out, threshold=thr, target_sizes=sizes)[0]
        return [(float(s), int(l), b.float().cpu().numpy()) for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]


class GTJitter:
    """GT boxes with jitter, random misses and false positives (debug)."""
    def __init__(self):
        self.rng, self.times_ms = np.random.default_rng(0), []

    def __call__(self, gts, cls_index, H, W):
        out = []
        for g in gts:
            if self.rng.random() < 0.2:
                continue
            b = g["box"].copy(); w, h = b[2] - b[0], b[3] - b[1]
            b += self.rng.normal(0, 0.05, 4) * [w, h, w, h]
            out.append((float(self.rng.uniform(0.3, 1)), cls_index[g["cls"]], b))
        for _ in range(2):
            x, y = self.rng.uniform(0, W - 100), self.rng.uniform(0, H - 100)
            out.append((float(self.rng.uniform(0, 0.5)), int(self.rng.integers(len(cls_index))), np.array([x, y, x + 80, y + 80])))
        return out


MODELS = {"owlv2-base": "google/owlv2-base-patch16-ensemble", "owlv2-large": "google/owlv2-large-patch14-ensemble",
          "gt-jitter": None}


def save_figure(path, rgb, gts, dets, names, title, score_thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.8)); ax.imshow(rgb); ax.axis("off")
    for g in gts:
        x0, y0, x1, y1 = g["box"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2, ls="--" if g["ignore"] else "-"))
        ax.text(x0, y0 - 3, names[g["cls"]], color="lime", fontsize=8)
    for s, c, b in dets:
        if s < score_thr:
            continue
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=1.5))
        ax.text(b[0], b[3] + 10, f"{names[c]} {s:.2f}", color="red", fontsize=8)
    ax.set_title(title + "   (green = ground truth, red = detections)"); plt.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--walks", nargs="*")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--min-box", type=int, default=12, help="ignore GT boxes smaller than this (px) on a side")
    ap.add_argument("--score-thr", type=float, default=0.1, help="score threshold for recall and figures")
    ap.add_argument("--figures", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "det"))
    args = ap.parse_args()

    walks = [w for w in list_walks(args.data) if not args.walks or w["walk_id"] in args.walks]
    meta0 = load_meta(os.path.join(args.data, walks[0]["path"]))
    classes = meta0["classes"]
    thing_names = [c for c in THING_PROMPTS if c in classes]
    thing_ids = [classes.index(c) for c in thing_names]
    cls_index = {cid: k for k, cid in enumerate(thing_ids)}          # class id -> prompt index
    inst_class = {int(i): classes.index(v["class"]) for i, v in meta0["instances"].items()}
    inst_actor = {int(i): v.get("actor", i) for i, v in meta0["instances"].items()}

    hf_id = MODELS[args.model]
    if hf_id is None:
        model, device = GTJitter(), "cpu"
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OWLv2(hf_id, [THING_PROMPTS[c] for c in thing_names], args.fp16 and device == "cuda", device)

    out_dir = os.path.join(args.out, args.model)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    per_cls = {c: {"scored": [], "n_gt": 0} for c in thing_ids}
    per_light = {}
    recall_dist = np.zeros((len(DIST_BINS), 2)); recall_size = np.zeros((len(SIZE_BINS), 2))   # hit, total
    frame_quality = []
    t0, warmed = time.time(), False

    def detect(fr, gts):
        H, W = fr["rgb"].shape[:2]
        if isinstance(model, GTJitter):
            return model(gts, cls_index, H, W)
        return model(fr["rgb"], 0.01)

    for w in walks:
        wdir = os.path.join(args.data, w["path"]); meta = load_meta(wdir)
        idx = list(range(0, len(meta["frames"]), args.stride))
        print(f"== {w['walk_id']} ({w['lighting']}): {len(idx)} frames", flush=True)
        pl = per_light.setdefault(w["lighting"], {c: {"scored": [], "n_gt": 0} for c in thing_ids})
        for i in idx:
            fr = load_frame(wdir, i, meta)
            gts = gt_boxes(fr["instance"], fr["semantic"], fr["depth"], inst_class, thing_ids, args.min_box, inst_actor)
            if not warmed and not isinstance(model, GTJitter):
                for _ in range(3): detect(fr, gts)
                model.times_ms.clear(); warmed = True
            dets = detect(fr, gts)
            hits = total = 0
            for k, c in enumerate(thing_ids):
                cd = [(s, b) for s, l, b in dets if l == k]
                res, g, used = match(cd, gts, c)
                n = sum(1 for x in g if not x["ignore"])
                for acc in (per_cls[c], pl[c]):
                    acc["scored"] += res; acc["n_gt"] += n
                # recall at score threshold, by distance and size
                _, _, used_thr = match([(s, b) for s, b in cd if s >= args.score_thr], gts, c)
                for x, u in zip(g, used_thr):
                    if x["ignore"]:
                        continue
                    total += 1; hits += u
                    for bi, (lo, hi) in enumerate(DIST_BINS):
                        if lo <= x["depth"] < hi: recall_dist[bi] += [u, 1]
                    for bi, (_, lo, hi) in enumerate(SIZE_BINS):
                        if lo <= x["area"] < hi: recall_size[bi] += [u, 1]
            frame_quality.append((hits / total if total else 1.0, -total, w["walk_id"], i))
        print(f"   done, {time.time() - t0:.0f}s", flush=True)

    ap_cls = {classes[c]: average_precision(v["scored"], v["n_gt"]) for c, v in per_cls.items()}
    n_gt = {classes[c]: v["n_gt"] for c, v in per_cls.items()}
    valid = [v for v in ap_cls.values() if not np.isnan(v)]
    m_ap = float(np.mean(valid)) if valid else float("nan")
    by_light = {}
    for L, d in per_light.items():
        aps = [average_precision(v["scored"], v["n_gt"]) for v in d.values()]
        aps = [a for a in aps if not np.isnan(a)]
        by_light[L] = float(np.mean(aps)) if aps else float("nan")
    ms = model.times_ms
    summary = {"model": args.model, "hf_id": hf_id, "fp16": bool(args.fp16), "device": device,
               "input_hw": getattr(model, "input_hw", None), "stride": args.stride, "min_box": args.min_box,
               "score_thr": args.score_thr,
               "ms_per_frame": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95))} if ms else None,
               "mAP50": m_ap, "AP50": ap_cls, "n_gt": n_gt, "mAP50_by_lighting": by_light,
               "recall_by_distance": {f"{lo}-{hi} m" if hi < 100 else f">{lo} m": (r[0] / r[1] if r[1] else None)
                                      for (lo, hi), r in zip(DIST_BINS, recall_dist)},
               "recall_by_size": {name: (r[0] / r[1] if r[1] else None) for (name, _, _), r in zip(SIZE_BINS, recall_size)},
               "prompts": {c: THING_PROMPTS[c] for c in thing_names}}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

    L = [f"# Detection benchmark: `{args.model}`", "",
         f"{hf_id or 'debug model'} · {'FP16' if args.fp16 else 'FP32'} on {device} · every {args.stride}th frame · "
         f"GT boxes from instance masks, ignored if < {args.min_box} px · recall at score ≥ {args.score_thr}", ""]
    if ms:
        L += [f"**Speed:** {np.median(ms):.2f} ms/frame median, {np.percentile(ms, 95):.2f} ms p95 "
              f"(model forward at {summary['input_hw']}, {len(thing_names)} text queries)", ""]
    L += [f"**mAP@0.5: {m_ap*100:.1f}**", "", "| Lighting | mAP@0.5 (%) |", "|---|---|"]
    L += [f"| {k} | {v*100:.1f} |" for k, v in by_light.items()]
    L += ["", "| Class | AP@0.5 (%) ↑ | GT boxes | Prompt |", "|---|---|---|---|"]
    for c, a in sorted(ap_cls.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=-1))):
        note = " (blockout asset)" if c in ("counter", "sink") else ""
        L.append(f"| {c}{note} | {'-' if np.isnan(a) else f'{a*100:.1f}'} | {n_gt[c]} | \"{THING_PROMPTS[c]}\" |")
    L += ["", "| Object distance | Recall (%) |", "|---|---|"]
    L += [f"| {k} | {'-' if v is None else f'{v*100:.1f}'} |" for k, v in summary["recall_by_distance"].items()]
    L += ["", "| Box size | Recall (%) |", "|---|---|"]
    L += [f"| {k} | {'-' if v is None else f'{v*100:.1f}'} |" for k, v in summary["recall_by_size"].items()]
    L.append("")

    frame_quality.sort()
    picked, seen = [], set()
    for q, negtot, walk_id, i in frame_quality:
        if walk_id not in seen and negtot < -1:
            picked.append((q, walk_id, i)); seen.add(walk_id)
        if len(picked) >= args.figures:
            break
    if picked:
        L += ["## Worst frames", ""]
    for q, walk_id, i in picked:
        w = next(x for x in walks if x["walk_id"] == walk_id)
        fr = load_frame(os.path.join(args.data, w["path"]), i)
        gts = gt_boxes(fr["instance"], fr["semantic"], fr["depth"], inst_class, thing_ids, args.min_box, inst_actor)
        dets = [(sc, thing_ids[k], b) for sc, k, b in detect(fr, gts)]     # prompt index -> class id
        name = f"worst_{walk_id}_{fr['frame']:06d}.png"
        save_figure(os.path.join(out_dir, "figures", name), fr["rgb"], gts, dets, classes,
                    f"{args.model} · {walk_id} frame {fr['frame']} · {w['lighting']} · recall {q*100:.0f} %",
                    args.score_thr)
        L += [f"![{walk_id} frame {fr['frame']}](figures/{name})", ""]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nWrote {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
