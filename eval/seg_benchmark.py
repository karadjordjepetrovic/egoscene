"""EgoScene semantic segmentation benchmark: score segmentation models against perfect ground truth.

Usage (from the repo root, venv active):
    python eval/seg_benchmark.py --model segformer-b2-ade --fp16
    python eval/seg_benchmark.py --model clipseg --fp16
    python eval/seg_benchmark.py --model gt-noise            # no GPU: sanity check of this script

Models (--model):
    segformer-b0-ade / segformer-b2-ade / segformer-b5-ade   closed-set, ADE20K (150 classes),
                         mapped to EgoScene classes by label name
    clipseg              open-vocabulary (CLIPSeg), prompted with one text per EgoScene class
    gt-noise             ground truth with random label noise (debug)

Scored classes: every EgoScene class except unlabelled, door and window (no visible pixels in v1).
Metrics: IoU per class, mIoU (mean over scored classes), pixel accuracy, all overall and per
lighting, plus model forward ms/frame. ADE20K classes with no EgoScene counterpart count as
"other", which is always wrong.

Outputs in eval/results/seg/<model>/: summary.md, summary.json, confusion.csv, figures/.
"""
import argparse, csv, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataset.egoscene import list_walks, load_meta, load_frame  # noqa: E402

EXCLUDED = {"unlabelled", "door", "window"}

# EgoScene class -> ADE20K label synonyms (matched against model.config.id2label names)
ADE_SYNONYMS = {
    "wall": ["wall"], "floor": ["floor", "flooring", "rug", "carpet"], "ceiling": ["ceiling"],
    "table": ["table", "coffee table", "desk"], "chair": ["chair", "armchair", "swivel chair", "seat"],
    "sofa": ["sofa", "couch", "lounge"], "shelf": ["shelf", "bookcase", "cabinet", "chest of drawers"],
    "plant": ["plant", "flora", "plant life", "flower", "pot", "flowerpot", "vase"],
    "lamp": ["lamp", "light", "light source", "sconce", "chandelier"],
    "counter": ["counter", "countertop", "kitchen island", "buffet"],
    "sink": ["sink"], "stove": ["stove", "kitchen stove", "range", "oven", "microwave"],
    "tv": ["television receiver", "television", "tv", "crt screen", "screen", "monitor", "computing machine"],
    "sign": ["signboard", "sign", "poster", "painting", "picture", "bulletin board"],
    "outdoor": ["sky", "tree", "building", "edifice", "road", "route", "car", "grass", "sidewalk",
                "house", "streetlight", "street lamp", "windowpane", "window"],
}
CLIP_PROMPTS = {
    "wall": "a wall", "floor": "the floor", "ceiling": "the ceiling", "table": "a table",
    "chair": "a chair", "sofa": "a sofa", "shelf": "a shelf", "plant": "a potted plant",
    "lamp": "a lamp", "counter": "a kitchen counter", "sink": "a sink", "stove": "a stove",
    "tv": "a television", "sign": "a sign with text", "outdoor": "the street outside a window",
}


# ----------------------------------------------------------------------------- models
class _HFBase:
    def _timed(self, fn):
        torch = self.torch
        if self.device == "cuda":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); out = fn(); e.record(); torch.cuda.synchronize(); self.times_ms.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); out = fn(); self.times_ms.append((time.perf_counter() - t0) * 1000)
        return out


class SegFormerModel(_HFBase):
    def __init__(self, hf_id, classes, fp16, device):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        self.torch, self.device, self.times_ms = torch, device, []
        self.dtype = torch.float16 if fp16 else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(hf_id).to(device, self.dtype).eval()
        # ADE id -> EgoScene class id (or 255 = "other")
        self.lut = np.full(len(self.model.config.id2label), 255, np.uint8)
        self.mapping = {}
        for ade_id, name in self.model.config.id2label.items():
            names = {n.strip().lower() for n in name.split(",")} | {name.strip().lower()}
            for cls, syns in ADE_SYNONYMS.items():
                if cls in classes and names & set(syns):
                    self.lut[int(ade_id)] = classes.index(cls)
                    self.mapping.setdefault(cls, []).append(name)
                    break

    def __call__(self, rgb):
        torch = self.torch
        pix = self.processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        self.input_hw = tuple(pix.shape[-2:])
        with torch.inference_mode():
            logits = self._timed(lambda: self.model(pixel_values=pix).logits)
            logits = torch.nn.functional.interpolate(logits.float(), size=rgb.shape[:2], mode="bilinear",
                                                     align_corners=False)
        self.last_raw = logits.argmax(1)[0].cpu().numpy()       # raw ADE20K ids, for diagnostics
        return self.lut[self.last_raw]


class CLIPSegModel(_HFBase):
    def __init__(self, hf_id, classes, fp16, device):
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self.torch, self.device, self.times_ms = torch, device, []
        self.dtype = torch.float16 if fp16 else torch.float32
        self.processor = CLIPSegProcessor.from_pretrained(hf_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(hf_id).to(device, self.dtype).eval()
        self.cls_ids = [classes.index(c) for c in CLIP_PROMPTS if c in classes]
        self.prompts = [CLIP_PROMPTS[classes[i]] for i in self.cls_ids]
        self.mapping = {classes[i]: [p] for i, p in zip(self.cls_ids, self.prompts)}

    def __call__(self, rgb):
        torch = self.torch
        inp = self.processor(text=self.prompts, images=[rgb] * len(self.prompts), padding=True, return_tensors="pt")
        inp = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)) for k, v in inp.items()}
        self.input_hw = tuple(inp["pixel_values"].shape[-2:])
        with torch.inference_mode():
            logits = self._timed(lambda: self.model(**inp).logits)          # (P, 352, 352), one image batch
            logits = torch.nn.functional.interpolate(logits[None].float(), size=rgb.shape[:2], mode="bilinear",
                                                     align_corners=False)[0]
        return np.array(self.cls_ids, np.uint8)[logits.argmax(0).cpu().numpy()]


class GTNoiseModel:
    def __init__(self, classes):
        self.rng, self.times_ms, self.n = np.random.default_rng(0), [], len(classes)
        self.mapping = {}

    def __call__(self, rgb, gt):
        out = gt.copy()
        flip = self.rng.random(gt.shape) < 0.1
        out[flip] = self.rng.integers(1, self.n, flip.sum())
        return out


MODELS = {
    "segformer-b0-ade": ("nvidia/segformer-b0-finetuned-ade-512-512", SegFormerModel),
    "segformer-b2-ade": ("nvidia/segformer-b2-finetuned-ade-512-512", SegFormerModel),
    "segformer-b5-ade": ("nvidia/segformer-b5-finetuned-ade-640-640", SegFormerModel),
    "clipseg": ("CIDAS/clipseg-rd64-refined", CLIPSegModel),
    "gt-noise": (None, GTNoiseModel),
}


# ----------------------------------------------------------------------------- figures
def save_figure(path, rgb, gt, pred, classes, scored, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20", len(classes))
    wrong = (pred != gt) & np.isin(gt, scored)
    fig, ax = plt.subplots(1, 4, figsize=(20, 3.4))
    ax[0].imshow(rgb); ax[0].set_title("RGB")
    ax[1].imshow(gt, cmap=cmap, vmin=0, vmax=len(classes) - 1, interpolation="nearest"); ax[1].set_title("Ground truth")
    ax[2].imshow(np.where(pred == 255, 0, pred), cmap=cmap, vmin=0, vmax=len(classes) - 1, interpolation="nearest")
    ax[2].set_title("Prediction")
    ax[3].imshow(rgb); ax[3].imshow(np.ma.masked_where(~wrong, wrong), cmap="autumn", alpha=0.6); ax[3].set_title("Wrong pixels")
    for a in ax: a.axis("off")
    present = [c for c in np.unique(gt) if c in scored]
    ax[1].legend([plt.Rectangle((0, 0), 1, 1, color=cmap(c)) for c in present], [classes[c] for c in present],
                 loc="lower left", fontsize=7, ncol=2, framealpha=0.8)
    fig.suptitle(title); plt.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


# ----------------------------------------------------------------------------- main
def iou_table(conf, scored):
    tp = np.diag(conf).astype(float)
    iou = {c: tp[c] / max(conf[c, :].sum() + conf[:, c].sum() - tp[c], 1) for c in scored if conf[c, :].sum() > 0}
    acc = tp[scored].sum() / max(conf[scored, :].sum(), 1)
    return iou, float(np.mean(list(iou.values()))) if iou else float("nan"), float(acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--walks", nargs="*")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--figures", type=int, default=4, help="number of worst frames to save")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "seg"))
    args = ap.parse_args()

    walks = [w for w in list_walks(args.data) if not args.walks or w["walk_id"] in args.walks]
    classes = load_meta(os.path.join(args.data, walks[0]["path"]))["classes"]
    scored = [i for i, c in enumerate(classes) if c not in EXCLUDED]
    nC = len(classes) + 1                    # last row/col = "other" (prediction outside EgoScene classes)
    OTHER = nC - 1

    hf_id, cls = MODELS[args.model]
    if hf_id is None:
        model, device = cls(classes), "cpu"
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: no CUDA GPU found, running on CPU")
        model = cls(hf_id, classes, args.fp16 and device == "cuda", device)
    if model.mapping:
        print("Class mapping:", {k: v for k, v in model.mapping.items()})

    out_dir = os.path.join(args.out, args.model)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    conf = np.zeros((nC, nC), np.int64)
    conf_light = {}
    ade_hist = None                          # SegFormer only: GT class x raw ADE20K label counts
    frame_scores = []                        # (pixel acc, walk, index)
    t0, warmed = time.time(), False
    for w in walks:
        wdir = os.path.join(args.data, w["path"])
        meta = load_meta(wdir)
        idx = list(range(0, len(meta["frames"]), args.stride))
        print(f"== {w['walk_id']} ({w['lighting']}): {len(idx)} frames", flush=True)
        cl = conf_light.setdefault(w["lighting"], np.zeros((nC, nC), np.int64))
        for i in idx:
            fr = load_frame(wdir, i, meta)
            gt = fr["semantic"]
            if isinstance(model, GTNoiseModel):
                pred = model(fr["rgb"], gt)
            else:
                if not warmed:
                    for _ in range(5): model(fr["rgb"])
                    model.times_ms.clear(); warmed = True
                pred = model(fr["rgb"])
            pred = np.where(pred == 255, OTHER, pred)
            if getattr(model, "last_raw", None) is not None:
                nA = len(model.model.config.id2label)
                if ade_hist is None:
                    ade_hist = np.zeros((len(classes), nA), np.int64)
                for c in np.unique(gt):
                    if c in scored:
                        ade_hist[c] += np.bincount(model.last_raw[gt == c].ravel(), minlength=nA)
            m = np.isin(gt, scored)
            c = np.bincount(gt[m].astype(np.int64) * nC + pred[m], minlength=nC * nC).reshape(nC, nC)
            conf += c; cl += c
            frame_scores.append((float((gt[m] == pred[m]).mean()) if m.any() else 1.0, w["walk_id"], i))
        print(f"   done, {time.time() - t0:.0f}s", flush=True)

    iou, miou, acc = iou_table(conf, scored)
    by_light = {}
    for L in ("day", "overcast", "evening", "night"):
        if L in conf_light:
            li, lm, la = iou_table(conf_light[L], scored)
            by_light[L] = {"miou": lm, "pixel_acc": la, "iou": {classes[k]: v for k, v in li.items()}}
    # most common confusion per class
    confused = {}
    for c in scored:
        row = conf[c].copy(); row[c] = 0
        if row.sum():
            j = int(row.argmax())
            confused[classes[c]] = ("other" if j == OTHER else classes[j], float(row[j] / conf[c].sum()))
    top_ade = {}
    if ade_hist is not None:
        id2label = model.model.config.id2label
        for c in scored:
            tot = ade_hist[c].sum()
            if tot:
                top = np.argsort(ade_hist[c])[::-1][:3]
                top_ade[classes[c]] = [(id2label[int(t)], float(ade_hist[c, t] / tot)) for t in top if ade_hist[c, t]]
    ms = model.times_ms
    summary = {"model": args.model, "hf_id": hf_id, "fp16": bool(args.fp16), "device": device,
               "input_hw": getattr(model, "input_hw", None), "stride": args.stride,
               "ms_per_frame": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95))} if ms else None,
               "miou": miou, "pixel_acc": acc, "iou": {classes[k]: v for k, v in iou.items()},
               "by_lighting": by_light, "most_confused_with": confused, "mapping": model.mapping,
               "top_ade_labels": top_ade}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "confusion.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); names = classes + ["other"]
        wr.writerow(["gt \\ pred"] + names)
        for r in range(nC):
            wr.writerow([names[r]] + conf[r].tolist())

    L = [f"# Segmentation benchmark: `{args.model}`", "",
         f"{hf_id or 'debug model'} · {'FP16' if args.fp16 else 'FP32'} on {device} · every {args.stride}th frame · "
         f"scored classes: {len(scored)} (door, window excluded)", ""]
    if ms:
        L += [f"**Speed:** {np.median(ms):.2f} ms/frame median, {np.percentile(ms, 95):.2f} ms p95 "
              f"(model forward at {summary['input_hw']})", ""]
    L += ["| Split | mIoU (%) ↑ | Pixel acc (%) ↑ |", "|---|---|---|",
          f"| **all** | {miou*100:.1f} | {acc*100:.1f} |"]
    L += [f"| {k} | {v['miou']*100:.1f} | {v['pixel_acc']*100:.1f} |" for k, v in by_light.items()]
    hdr = "| Class | IoU (%) ↑ | Most confused with |" + (" Top ADE20K labels predicted |" if top_ade else "")
    L += ["", hdr, "|---|---|---|" + ("---|" if top_ade else "")]
    for k, v in sorted(iou.items(), key=lambda kv: kv[1]):
        cw = confused.get(classes[k])
        line = f"| {classes[k]} | {v*100:.1f} | {f'{cw[0]} ({cw[1]*100:.0f} %)' if cw else '-'} |"
        if top_ade:
            line += " " + ", ".join(f"{n.split(',')[0]} {p*100:.0f}%" for n, p in top_ade.get(classes[k], [])) + " |"
        L.append(line)
    L.append("")

    frame_scores.sort()
    picked, seen = [], set()
    for e in frame_scores:
        if e[1] not in seen:
            picked.append(e); seen.add(e[1])
        if len(picked) >= args.figures:
            break
    if picked:
        L += ["## Worst frames", ""]
    for a, walk_id, i in picked:
        w = next(x for x in walks if x["walk_id"] == walk_id)
        fr = load_frame(os.path.join(args.data, w["path"]), i)
        pred = model(fr["rgb"], fr["semantic"]) if isinstance(model, GTNoiseModel) else model(fr["rgb"])
        name = f"worst_{walk_id}_{fr['frame']:06d}.png"
        save_figure(os.path.join(out_dir, "figures", name), fr["rgb"], fr["semantic"], pred, classes, scored,
                    f"{args.model} · {walk_id} frame {fr['frame']} · {w['lighting']} · pixel acc {a*100:.1f} %")
        L += [f"![{walk_id} frame {fr['frame']}](figures/{name})", ""]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[: 8 + len(by_light) + len(iou) + 4]))
    print(f"\nWrote {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
