"""EgoScene depth benchmark: score monocular depth models against perfect ground truth.

Usage (from the repo root, venv active):
    python eval/depth_benchmark.py --model da2-small --fp16
    python eval/depth_benchmark.py --model da2-metric-indoor-small --fp16
    python eval/depth_benchmark.py --model da2-small --stride 30 --walks walk_01 walk_04   # quick run

Models (--model):
    da2-small                 Depth Anything V2 Small      relative (affine-invariant disparity)
    da2-base                  Depth Anything V2 Base       relative
    da2-metric-indoor-small   Depth Anything V2 Metric Indoor Small (metres)
    gt-noise                  ground truth + noise, no GPU needed: sanity check of this script

Relative models are aligned to GT per frame (least-squares scale + shift in disparity
space, the standard protocol). Metric models are scored as-is ("raw") and after
per-frame median scaling ("scaled").

Metrics on valid pixels (GT depth > 0 and <= --max-depth; the outdoor backdrop has GT 0):
    AbsRel  mean |pred - gt| / gt          (lower is better)
    RMSE    sqrt(mean (pred - gt)^2), m    (lower is better)
    d1      % of pixels with max(pred/gt, gt/pred) < 1.25   (higher is better)
    Flicker median change, between consecutive frames, of the per-pixel error ratio
            pred/gt at the same surface point (correspondences from GT geometry), in %
    ms      model forward time per frame (CUDA events, batch 1, after warm-up)

Outputs in eval/results/depth/<model>/: per_frame.csv, summary.json, summary.md,
figures/worst_*.png (RGB | GT | prediction | AbsRel error map).
"""
import argparse, csv, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataset.egoscene import list_walks, load_meta, load_frame  # noqa: E402

MODELS = {
    "da2-small": ("depth-anything/Depth-Anything-V2-Small-hf", "relative"),
    "da2-base": ("depth-anything/Depth-Anything-V2-Base-hf", "relative"),
    "da2-metric-indoor-small": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "metric"),
    "gt-noise": (None, "metric"),
}
MIN_DEPTH = 0.01
DIST_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 100.0)]


# ----------------------------------------------------------------------------- models
class HFDepthModel:
    def __init__(self, hf_id, fp16, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id).to(device, self.dtype).eval()
        self.times_ms = []

    def __call__(self, rgb, timed=True):
        torch = self.torch
        inputs = self.processor(images=rgb, return_tensors="pt")
        pix = inputs["pixel_values"].to(self.device, self.dtype)
        self.input_hw = tuple(pix.shape[-2:])
        with torch.inference_mode():
            if self.device == "cuda" and timed:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); out = self.model(pixel_values=pix).predicted_depth; end.record()
                torch.cuda.synchronize(); self.times_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter(); out = self.model(pixel_values=pix).predicted_depth
                if timed: self.times_ms.append((time.perf_counter() - t0) * 1000)
            out = torch.nn.functional.interpolate(out[:, None].float(), size=rgb.shape[:2],
                                                  mode="bicubic", align_corners=False)[0, 0]
        return out.cpu().numpy()

    def warmup(self, rgb, n=10):
        for _ in range(n):
            self(rgb, timed=False)


class GTNoiseModel:
    """Ground truth with a global scale error and 5 % per-pixel noise: tests the metric code."""
    def __init__(self, seed=0):
        self.rng, self.times_ms = np.random.default_rng(seed), []

    def __call__(self, rgb, gt):
        return gt * 1.1 * (1 + 0.05 * self.rng.standard_normal(gt.shape)).astype(np.float32)

    def warmup(self, rgb, n=0):
        pass


# ----------------------------------------------------------------------------- metrics
def align_disparity(pred_disp, gt, mask):
    """Least-squares scale+shift of predicted disparity to 1/gt on valid pixels -> depth (m)."""
    x, y = pred_disp[mask].astype(np.float64), 1.0 / gt[mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], 1)
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    disp = s * pred_disp + t
    return (1.0 / np.clip(disp, 1.0 / 100.0, None)).astype(np.float32)


def frame_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / g, g / p)
    return {"absrel": float(np.mean(np.abs(p - g) / g)),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "d1": float(np.mean(ratio < 1.25) * 100)}


def correspondences(gt_i, gt_j, c2w_i, c2w_j, K):
    """Pixels of frame i visible in frame j (by GT geometry): returns (vi, ui, vj, uj)."""
    v, u = np.nonzero(gt_i > 0)
    z = gt_i[v, u]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)])
    Q = np.linalg.inv(c2w_j) @ c2w_i @ P
    zj = Q[2]
    uj = np.round(Q[0] / zj * K[0, 0] + K[0, 2]).astype(int)
    vj = np.round(Q[1] / zj * K[1, 1] + K[1, 2]).astype(int)
    H, W = gt_i.shape
    ok = (zj > 0.1) & (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
    v, u, vj, uj, zj = v[ok], u[ok], vj[ok], uj[ok], zj[ok]
    seen = gt_j[vj, uj]
    vis = (seen > 0) & (np.abs(zj - seen) / np.maximum(seen, 1e-6) < 0.01)   # not occluded
    return v[vis], u[vis], vj[vis], uj[vis]


# ----------------------------------------------------------------------------- figures
def save_failure_figure(path, rgb, gt, pred, mask, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err = np.where(mask, np.abs(pred - gt) / np.maximum(gt, 1e-6), np.nan)
    vmax = np.percentile(gt[gt > 0], 99) if (gt > 0).any() else 5
    fig, ax = plt.subplots(1, 4, figsize=(20, 3.4))
    ax[0].imshow(rgb); ax[0].set_title("RGB")
    ax[1].imshow(np.ma.masked_equal(gt, 0), cmap="turbo", vmin=0, vmax=vmax); ax[1].set_title("Ground truth (m)")
    im = ax[2].imshow(np.ma.masked_where(~mask, pred), cmap="turbo", vmin=0, vmax=vmax); ax[2].set_title("Prediction (m)")
    fig.colorbar(im, ax=ax[2], fraction=0.03)
    im = ax[3].imshow(err * 100, cmap="magma", vmin=0, vmax=50); ax[3].set_title("AbsRel error (%)")
    fig.colorbar(im, ax=ax[3], fraction=0.03)
    for a in ax: a.axis("off")
    fig.suptitle(title); plt.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--walks", nargs="*", help="subset of walk ids (default: all)")
    ap.add_argument("--stride", type=int, default=10, help="evaluate every Nth frame (default 10)")
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-flicker", action="store_true", help="skip the extra frame per sample used for flicker")
    ap.add_argument("--worst", type=int, default=4, help="number of worst frames to save as figures")
    ap.add_argument("--implied-fov", type=float, default=74.0,
                    help="horizontal FOV (deg) the metric model implicitly assumes; used by the fov_corrected "
                         "variant (74 measured for DA V2 Metric Indoor: scale 0.75 at 90, 1.00 at 74, 1.30 at 60)")
    ap.add_argument("--crop-fov", type=float, default=None,
                    help="centre-crop frames to this horizontal FOV in degrees (e.g. 60) before inference; "
                         "tests whether a metric model's scale error comes from the wide glasses FOV")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "depth"))
    args = ap.parse_args()

    hf_id, kind = MODELS[args.model]
    if hf_id is None:
        model, device = GTNoiseModel(), "cpu"
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: no CUDA GPU found, running on CPU (slow, and ms/frame won't be meaningful)")
        model = HFDepthModel(hf_id, args.fp16 and device == "cuda", device)
    # calibrated: one scale factor per walk, estimated on the walk's first evaluated frame and
    # then kept fixed, like a one-time calibration when the user puts the glasses on.
    # fov_corrected: raw x tan(implied/2) / tan(actual/2), a constant from the camera intrinsics only,
    # no ground truth. Valid if the model's metric output behaves as if every image had the implied FOV.
    variants = ["aligned"] if kind == "relative" else ["raw", "fov_corrected", "calibrated", "scaled"]
    cal = {"scale": 1.0, "fov_factor": 1.0}

    def predict(fr):
        if isinstance(model, GTNoiseModel):
            return model(fr["rgb"], fr["depth"])
        return model(fr["rgb"])

    def to_depth(raw, gt, mask, variant):
        if kind == "relative":
            pred = align_disparity(raw, gt, mask)
        elif variant == "fov_corrected":
            pred = raw * cal["fov_factor"]
        elif variant == "calibrated":
            pred = raw * cal["scale"]
        elif variant == "scaled":
            pred = raw * float(np.median(gt[mask] / np.maximum(raw[mask], 1e-6)))
        else:
            pred = raw
        # Clamp to the evaluation range, the standard protocol (NYU/KITTI). Without it, a few
        # pixels the model calls "infinitely far" (disparity ~ 0) dominate AbsRel and RMSE.
        return np.clip(pred, MIN_DEPTH, args.max_depth).astype(np.float32)

    out_dir = os.path.join(args.out, args.model + (f"_crop{args.crop_fov:g}" if args.crop_fov else ""))
    crop = None   # (x0, y0, w, h), set per walk from the intrinsics

    def load(wdir, i, meta):
        fr = load_frame(wdir, i, meta)
        if crop is not None:
            x0, y0, cw, ch = crop
            for k in ("rgb", "depth", "semantic", "instance"):
                fr[k] = np.ascontiguousarray(fr[k][y0:y0 + ch, x0:x0 + cw])
        return fr
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    walks = [w for w in list_walks(args.data) if not args.walks or w["walk_id"] in args.walks]

    rows, worst = [], []          # worst: (absrel, walk, index)
    acc = {v: {"dist": np.zeros((len(DIST_BINS), 3)), "cls": None} for v in variants}  # sum absrel, sum d1, count
    warmed = False
    t_start = time.time()
    for w in walks:
        wdir = os.path.join(args.data, w["path"])
        meta = load_meta(wdir)
        classes = meta["classes"]
        for v in variants:
            if acc[v]["cls"] is None:
                acc[v]["cls"] = np.zeros((len(classes), 3))
        K = np.array([[meta["intrinsics"]["fx"], 0, meta["intrinsics"]["cx"]],
                      [0, meta["intrinsics"]["fy"], meta["intrinsics"]["cy"]], [0, 0, 1]])
        if args.crop_fov:
            cw = int(round(2 * K[0, 0] * np.tan(np.radians(args.crop_fov) / 2)))
            ch = int(round(cw * meta["height"] / meta["width"]))
            x0, y0 = (meta["width"] - cw) // 2, (meta["height"] - ch) // 2
            crop = (x0, y0, cw, ch)
            K = K.copy(); K[0, 2] -= x0; K[1, 2] -= y0
        img_w = crop[2] if (args.crop_fov and crop) else meta["width"]
        cal["fov_factor"] = float(np.tan(np.radians(args.implied_fov) / 2) / ((img_w / 2) / K[0, 0]))
        n = len(meta["frames"])
        idx = list(range(0, n - 1, args.stride))
        print(f"== {w['walk_id']} ({w['lighting']}): {len(idx)} frames", flush=True)
        for i in idx:
            fr = load(wdir, i, meta)
            if not warmed:
                model.warmup(fr["rgb"]); warmed = True
            gt = fr["depth"]
            mask = (gt > 0) & (gt <= args.max_depth)
            raw = predict(fr)
            if i == idx[0]:
                cal["scale"] = float(np.median(gt[mask] / np.maximum(raw[mask], 1e-6)))
            nxt = None
            if not args.no_flicker:
                fr2 = load(wdir, i + 1, meta)
                mask2 = (fr2["depth"] > 0) & (fr2["depth"] <= args.max_depth)
                nxt = (fr2, mask2, predict(fr2))
            for v in variants:
                pred = to_depth(raw, gt, mask, v)
                m = frame_metrics(pred, gt, mask)
                row = {"walk_id": w["walk_id"], "lighting": w["lighting"], "speed_cm_s": w["speed_cm_s"],
                       "frame": fr["frame"], "variant": v, **m}
                if kind == "metric" and v == "raw":
                    # > 1 means the model predicts things closer than they are, < 1 farther
                    row["scale_gt_over_pred"] = float(np.median(gt[mask] / np.maximum(raw[mask], 1e-6)))
                # share of valid pixels the model pushes to the far end of the range ("sky" mistakes)
                row["far_clipped_pct"] = float(np.mean(pred[mask] >= args.max_depth * 0.999) * 100)
                # error by distance and by class (pixel-weighted)
                rel = np.abs(pred - gt) / np.maximum(gt, 1e-6)
                good = np.maximum(pred / np.maximum(gt, 1e-6), gt / np.maximum(pred, 1e-6)) < 1.25
                for b, (lo, hi) in enumerate(DIST_BINS):
                    mb = mask & (gt >= lo) & (gt < hi)
                    acc[v]["dist"][b] += [rel[mb].sum(), good[mb].sum(), mb.sum()]
                sem = fr["semantic"]
                for c in np.unique(sem[mask]):
                    mc = mask & (sem == c)
                    acc[v]["cls"][c] += [rel[mc].sum(), good[mc].sum(), mc.sum()]
                if nxt is not None:
                    fr2, mask2, raw2 = nxt
                    pred2 = to_depth(raw2, fr2["depth"], mask2, v)
                    vi, ui, vj, uj = correspondences(gt, fr2["depth"], fr["c2w"], fr2["c2w"], K)
                    keep = mask[vi, ui] & mask2[vj, uj]
                    r1 = pred[vi[keep], ui[keep]] / gt[vi[keep], ui[keep]]
                    r2 = pred2[vj[keep], uj[keep]] / fr2["depth"][vj[keep], uj[keep]]
                    row["flicker"] = float(np.median(np.abs(r1 - r2)) * 100) if keep.any() else float("nan")
                rows.append(row)
                if v == variants[0]:
                    worst.append((m["absrel"], w["walk_id"], i))
        print(f"   done, {time.time() - t_start:.0f}s elapsed", flush=True)

    # ---- write per-frame CSV
    keys = ["walk_id", "lighting", "speed_cm_s", "frame", "variant", "absrel", "rmse", "d1", "flicker",
            "scale_gt_over_pred", "far_clipped_pct"]
    with open(os.path.join(out_dir, "per_frame.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)

    # ---- summaries
    def agg(rs):
        out = {k: float(np.nanmean([r[k] for r in rs])) for k in ("absrel", "rmse", "d1")}
        if rs and "flicker" in rs[0]:
            out["flicker"] = float(np.nanmedian([r["flicker"] for r in rs]))
        if rs and "scale_gt_over_pred" in rs[0]:
            out["scale_gt_over_pred"] = float(np.median([r["scale_gt_over_pred"] for r in rs]))
        if rs:
            out["far_clipped_pct"] = float(np.mean([r["far_clipped_pct"] for r in rs]))
        out["frames"] = len(rs)
        return out

    ms = model.times_ms
    summary = {"model": args.model, "hf_id": hf_id, "kind": kind, "fp16": bool(args.fp16),
               "device": device, "input_hw": getattr(model, "input_hw", None), "stride": args.stride, "max_depth_m": args.max_depth,
               "ms_per_frame": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95))} if ms else None,
               "variants": {}}
    lightings = ["day", "overcast", "evening", "night"]
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        d, c = acc[v]["dist"], acc[v]["cls"]
        summary["variants"][v] = {
            "overall": agg(rs),
            "by_lighting": {L: agg([r for r in rs if r["lighting"] == L]) for L in lightings
                            if any(r["lighting"] == L for r in rs)},
            "by_distance": {f"{lo:g}-{hi:g} m" if hi < 100 else f">{lo:g} m":
                            {"absrel": float(d[b, 0] / d[b, 2]), "d1": float(d[b, 1] / d[b, 2] * 100),
                             "pixels": int(d[b, 2])}
                            for b, (lo, hi) in enumerate(DIST_BINS) if d[b, 2] > 0},
            "by_class": {classes[k]: {"absrel": float(c[k, 0] / c[k, 2]), "d1": float(c[k, 1] / c[k, 2] * 100),
                                      "pixels": int(c[k, 2])}
                         for k in range(len(classes)) if c[k, 2] > 0},
        }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

    # ---- markdown
    L = [f"# Depth benchmark: `{args.model}`", "",
         f"{hf_id or 'debug model'} · {kind} · {'FP16' if args.fp16 else 'FP32'} on {device} · "
         f"every {args.stride}th frame · GT ≤ {args.max_depth:g} m"
         + (f" · centre-cropped to {args.crop_fov:g}° HFOV" if args.crop_fov else " · full 90° HFOV"), ""]
    if ms:
        L += [f"**Speed:** {np.median(ms):.2f} ms/frame median, {np.percentile(ms, 95):.2f} ms p95 "
              f"(model forward at {summary['input_hw']}, batch 1)", ""]
    for v in variants:
        s = summary["variants"][v]
        L += [f"## {v}", "", "| Split | AbsRel ↓ | RMSE (m) ↓ | δ1 (%) ↑ | Flicker (%) ↓ | Far-clipped px (%) | Frames |",
              "|---|---|---|---|---|---|---|"]
        fmt = lambda name, a: (f"| {name} | {a['absrel']:.3f} | {a['rmse']:.3f} | {a['d1']:.1f} | "
                               f"{a.get('flicker', float('nan')):.2f} | {a.get('far_clipped_pct', float('nan')):.1f} | {a['frames']} |")
        L.append(fmt("**all**", s["overall"]))
        if "scale_gt_over_pred" in s["overall"]:
            L.insert(len(L) - 3, f"Median scale error GT/pred: **{s['overall']['scale_gt_over_pred']:.2f}** "
                                 f"(1.0 = correct metric scale; < 1 means the model predicts too far)\n")
        L += [fmt(k, a) for k, a in s["by_lighting"].items()]
        L += ["", "| Distance | AbsRel ↓ | δ1 (%) ↑ |", "|---|---|---|"]
        L += [f"| {k} | {a['absrel']:.3f} | {a['d1']:.1f} |" for k, a in s["by_distance"].items()]
        L += ["", "| Class | AbsRel ↓ | δ1 (%) ↑ | Pixels |", "|---|---|---|---|"]
        L += [f"| {k} | {a['absrel']:.3f} | {a['d1']:.1f} | {a['pixels']:,} |"
              for k, a in sorted(s["by_class"].items(), key=lambda kv: -kv[1]["absrel"])]
        L.append("")

    # ---- worst-frame figures (re-run the model on just those frames)
    worst.sort(reverse=True)
    picked, seen_walks = [], set()
    for e in worst:                          # prefer one frame per walk for variety
        if e[1] not in seen_walks:
            picked.append(e); seen_walks.add(e[1])
        if len(picked) == args.worst:
            break
    if picked:
        L += ["## Worst frames", ""]
    for absrel, walk_id, i in picked:
        w = next(x for x in walks if x["walk_id"] == walk_id)
        wdir = os.path.join(args.data, w["path"])
        meta_w = load_meta(wdir)
        if args.crop_fov:
            cw = int(round(2 * meta_w["intrinsics"]["fx"] * np.tan(np.radians(args.crop_fov) / 2)))
            ch = int(round(cw * meta_w["height"] / meta_w["width"]))
            crop = ((meta_w["width"] - cw) // 2, (meta_w["height"] - ch) // 2, cw, ch)
        fr = load(wdir, i, meta_w)
        gt = fr["depth"]; mask = (gt > 0) & (gt <= args.max_depth)
        pred = to_depth(predict(fr), gt, mask, variants[0])
        name = f"worst_{walk_id}_{fr['frame']:06d}.png"
        save_failure_figure(os.path.join(out_dir, "figures", name), fr["rgb"], gt, pred, mask,
                            f"{args.model} ({variants[0]}) · {walk_id} frame {fr['frame']} · {w['lighting']} · AbsRel {absrel:.3f}")
        L += [f"![{walk_id} frame {fr['frame']}](figures/{name})", ""]

    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:12 + 2 * len(lightings)]))
    print(f"\nWrote {out_dir}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
