"""Sim-to-real check: run the Phase 3 models on a real first-person phone video (no ground truth).

Usage (from the repo root, venv active):
    python eval/real_video.py data/real/home_walk_01.mp4 --long-side-fov 72 --fp16

    --long-side-fov  FOV across the frame's LONG side, in degrees (Samsung S25 main camera, 16:9
                     video: ~72, whether you hold the phone in portrait or landscape).
                     The FOV correction is applied along the long side because that's the axis it
                     was measured on in EgoScene (landscape frames, 90 deg wide -> model assumes ~74).
    --every-sec  sample one frame every N seconds (default 2)

For each sampled frame it runs DA V2 Small (relative depth), DA V2 Metric Indoor (raw and
FOV-corrected with the tan(37 deg)/tan(hfov/2) factor measured on EgoScene), SegFormer-B2
(mapped to EgoScene classes), OWLv2 (EgoScene object prompts) and EasyOCR, and saves a panel
figure per frame plus stats.json.

PRIVACY: everything is written to data/real/_out/<video name>/ (git-ignored). Look through the
figures and copy only the ones you're happy to publish to eval/results/real/.
"""
import argparse, json, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import depth_benchmark as DB   # noqa: E402
import seg_benchmark as SB     # noqa: E402
import det_benchmark as TB     # noqa: E402

MODEL_IMPLIED_FOV = 74.0       # long-side FOV DA V2 Metric Indoor implicitly assumes (measured on EgoScene)


def read_frames(path, every_sec, max_side):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Can't open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_sec)), 1)
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = bgr.shape[:2]
            s = min(1.0, max_side / max(h, w))
            if s < 1:
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append((i / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        i += 1
    return frames, fps


def disparity_to_vis(d):
    lo, hi = np.percentile(d, [2, 98])
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--long-side-fov", type=float, required=True, help="FOV across the frame's long side (deg)")
    ap.add_argument("--every-sec", type=float, default=2.0)
    ap.add_argument("--max-side", type=int, default=1280, help="downscale frames so the long side is at most this")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--score-thr", type=float, default=0.2, help="OWLv2 score threshold for drawing")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16 and device == "cuda"

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(REPO, "data", "real", "_out", name)
    os.makedirs(out_dir, exist_ok=True)

    frames, fps = read_frames(args.video, args.every_sec, args.max_side)
    H, W = frames[0][1].shape[:2]
    print(f"{len(frames)} frames ({W}x{H}) from {args.video} at {fps:.0f} fps, one every {args.every_sec:g}s")

    labels = json.load(open(os.path.join(REPO, "dataset", "labels", "apartment_01_labels.json")))
    classes = ["unlabelled"] + labels["classes"]
    fov_factor = float(np.tan(np.radians(MODEL_IMPLIED_FOV) / 2) / np.tan(np.radians(args.long_side_fov) / 2))
    # alternative hypothesis: the assumption is tied to the SHORT side (can't be told apart on landscape-only data)
    ar = min(H, W) / max(H, W)
    short_model = 2 * np.degrees(np.arctan(np.tan(np.radians(MODEL_IMPLIED_FOV) / 2) * 9 / 16))
    short_video = 2 * np.degrees(np.arctan(np.tan(np.radians(args.long_side_fov) / 2) * ar))
    fov_factor_short = float(np.tan(np.radians(short_model) / 2) / np.tan(np.radians(short_video) / 2))
    print(f"FOV correction for DA V2 Metric: x{fov_factor:.2f} along the long side "
          f"(model ~{MODEL_IMPLIED_FOV:g} deg, video {args.long_side_fov:g} deg); x{fov_factor_short:.2f} if it's tied to the short side")

    print("Loading models ...")
    rel = DB.HFDepthModel(DB.MODELS["da2-small"][0], fp16, device)
    met = DB.HFDepthModel(DB.MODELS["da2-metric-indoor-small"][0], fp16, device)
    seg = SB.SegFormerModel(SB.MODELS["segformer-b2-ade"][0], classes, fp16, device)
    thing_names = list(TB.THING_PROMPTS)
    det = TB.OWLv2(TB.MODELS["owlv2-base"], [TB.THING_PROMPTS[c] for c in thing_names], fp16, device)
    ocr = None
    if not args.no_ocr:
        from text_benchmark import EasyOCRModel
        ocr = EasyOCRModel(gpu=device == "cuda")

    cmap_seg = plt.get_cmap("tab20", len(classes))
    stats = []
    t0 = time.time()
    for k, (t, rgb) in enumerate(frames):
        disp = rel(rgb)
        raw = met(rgb)
        corr = raw * fov_factor
        sem = seg(rgb)
        dets = det(rgb, 0.05)
        texts = ocr(rgb) if ocr else []
        ch, cw = H // 2, W // 2
        centre = (slice(ch - H // 10, ch + H // 10), slice(cw - W // 10, cw + W // 10))
        present = {classes[c]: float(np.mean(sem == c) * 100) for c in np.unique(sem) if c < len(classes)}
        stats.append({"t_sec": round(t, 2), "centre_depth_raw_m": float(np.median(raw[centre])),
                      "centre_depth_fov_corrected_m": float(np.median(corr[centre])),
                      "segformer_classes_pct": {c: round(v, 1) for c, v in sorted(present.items(), key=lambda kv: -kv[1])},
                      "detections": [{"class": thing_names[l], "score": round(s, 2)} for s, l, b in dets if s >= args.score_thr],
                      "ocr": [{"text": tx, "conf": round(c, 2)} for _, tx, c in texts]})

        vmax = float(np.percentile(corr, 98))
        ph = 6.0                                    # panel height in inches
        fig, ax = plt.subplots(1, 5, figsize=(5 * ph * W / H + 1.5, ph + 0.9))
        ax[0].imshow(rgb); ax[0].set_title(f"RGB, t={t:.0f}s", fontsize=9)
        ax[1].imshow(disparity_to_vis(disp), cmap="turbo_r"); ax[1].set_title("DA V2 Small\n(relative)", fontsize=9)
        im = ax[2].imshow(corr, cmap="turbo", vmin=0, vmax=vmax)
        ax[2].set_title(f"DA V2 Metric (m), x{fov_factor:.2f}\ncentre {stats[-1]['centre_depth_fov_corrected_m']:.2f} m "
                        f"(raw {stats[-1]['centre_depth_raw_m']:.2f})", fontsize=9)
        fig.colorbar(im, ax=ax[2], fraction=0.05)
        ax[3].imshow(sem, cmap=cmap_seg, vmin=0, vmax=len(classes) - 1, interpolation="nearest"); ax[3].set_title("SegFormer-B2", fontsize=9)
        top = [c for c, v in stats[-1]["segformer_classes_pct"].items() if v > 2 and c != "unlabelled"][:6]
        ax[3].legend([plt.Rectangle((0, 0), 1, 1, color=cmap_seg(classes.index(c))) for c in top], top,
                     loc="lower left", fontsize=7, framealpha=0.8)
        ax[4].imshow(rgb); ax[4].set_title("OWLv2 (red)\nEasyOCR (cyan)", fontsize=9)
        for s, l, b in dets:
            if s >= args.score_thr:
                ax[4].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=1.5))
                ax[4].text(b[0], b[1] - 4, f"{thing_names[l]} {s:.2f}", color="red", fontsize=7)
        for bx, tx, c in texts:
            ax[4].add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="cyan", lw=1.2))
            ax[4].text(bx[0], bx[3] + 12, tx, color="cyan", fontsize=7)
        for a in ax: a.axis("off")
        plt.tight_layout(); fig.savefig(os.path.join(out_dir, f"frame_{k:03d}_t{int(t):03d}s.png"), dpi=70); plt.close(fig)
        print(f"  frame {k + 1}/{len(frames)}  t={t:.0f}s  centre depth {stats[-1]['centre_depth_fov_corrected_m']:.2f} m  "
              f"{len(stats[-1]['detections'])} objects, {len(texts)} text boxes", flush=True)

    ms = lambda m: float(np.median(m.times_ms)) if m.times_ms else None
    summary = {"video": os.path.basename(args.video), "frames": len(frames), "size": [W, H],
               "long_side_fov": args.long_side_fov, "fov_factor": fov_factor, "fov_factor_if_short_side": fov_factor_short,
               "ms_per_frame": {"da2_small": ms(rel), "da2_metric": ms(met), "segformer_b2": ms(seg),
                                "owlv2": ms(det), "easyocr": ms(ocr) if ocr else None},
               "per_frame": stats}
    json.dump(summary, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(f"\nWrote {len(frames)} figures + stats.json to {out_dir} ({time.time() - t0:.0f}s)")
    print("Review them, then copy the ones you want to publish to eval/results/real/.")


if __name__ == "__main__":
    main()
