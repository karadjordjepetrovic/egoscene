"""EgoScene scene-text benchmark: can an OCR model find and read the wall signs?

Usage (from the repo root, venv active; `pip install easyocr` once):
    python eval/text_benchmark.py --model easyocr
    python eval/text_benchmark.py --model gt-oracle      # no OCR needed: sanity check of this script

Ground truth: the four wall signs (Sign_01..04) and their known text. A sign counts as visible in a
frame if at least --min-pixels of its instance mask are on screen.
  detected  an OCR box overlaps the sign's box (IoU >= 0.1, or the OCR box centre lies inside it)
  read      the matched OCR text, upper-cased and stripped to letters/digits, equals the sign text
  CER       character error rate of the matched text vs the sign text (1.0 if nothing matched)
Splits: by distance to the sign, by the sign's on-screen height in pixels, and by lighting.

Outputs in eval/results/text/<model>/: summary.md, summary.json, per_sign.csv, figures/legibility.png
"""
import argparse, csv, json, os, re, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataset.egoscene import list_walks, load_meta, load_frame  # noqa: E402

SIGN_TEXT = {"Sign_01": "EXIT", "Sign_02": "Kitchen", "Sign_03": "Welcome Home", "Sign_04": "OCTOBER"}
DIST_BINS = [(0, 1), (1, 2), (2, 3), (3, 100)]
HEIGHT_BINS = [(0, 20), (20, 40), (40, 80), (80, 10000)]


def norm(t):
    return re.sub(r"[^A-Z0-9]", "", t.upper())


def cer(pred, gt):
    p, g = norm(pred), norm(gt)
    d = list(range(len(g) + 1))
    for i, pc in enumerate(p, 1):
        prev, d[0] = d[0], i
        for j, gc in enumerate(g, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (pc != gc))
    return min(d[len(g)] / max(len(g), 1), 1.0)


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    ar = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / max(ar(a) + ar(b) - inter, 1e-9)


class EasyOCRModel:
    def __init__(self, gpu):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        self.times_ms = []

    def __call__(self, rgb):
        t0 = time.perf_counter()
        res = self.reader.readtext(rgb)
        self.times_ms.append((time.perf_counter() - t0) * 1000)
        out = []
        for pts, text, conf in res:
            pts = np.array(pts, float)
            out.append(([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], text, float(conf)))
        return out


class GTOracle:
    """Reads a sign perfectly when its on-screen height is >= 30 px (debug)."""
    times_ms = []

    def __call__(self, rgb, signs):
        return [(s["box"], s["text"], 1.0) for s in signs if s["height"] >= 30]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["easyocr", "gt-oracle"])
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--walks", nargs="*")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-pixels", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "text"))
    args = ap.parse_args()

    walks = [w for w in list_walks(args.data) if not args.walks or w["walk_id"] in args.walks]
    meta0 = load_meta(os.path.join(args.data, walks[0]["path"]))
    sign_ids = {int(i): v["actor"] for i, v in meta0["instances"].items() if v.get("actor") in SIGN_TEXT}
    if not sign_ids:
        sys.exit("No Sign_0X instances found in meta.json")
    if args.model == "easyocr":
        import torch
        model = EasyOCRModel(gpu=torch.cuda.is_available())
    else:
        model = GTOracle()

    out_dir = os.path.join(args.out, args.model)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    rows, crops = [], []
    t0 = time.time()
    for w in walks:
        wdir = os.path.join(args.data, w["path"]); meta = load_meta(wdir)
        n_vis = 0
        for i in range(0, len(meta["frames"]), args.stride):
            fr = load_frame(wdir, i, meta)
            inst = fr["instance"]
            signs = []
            for iid, actor in sign_ids.items():
                m = inst == iid
                if m.sum() < args.min_pixels:
                    continue
                ys, xs = np.nonzero(m)
                d = fr["depth"][m]; d = d[d > 0]
                signs.append({"actor": actor, "text": SIGN_TEXT[actor], "box": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                              "height": int(ys.max() + 1 - ys.min()), "depth": float(np.median(d)) if len(d) else 0.0})
            if not signs:
                continue
            n_vis += 1
            dets = model(fr["rgb"], signs) if isinstance(model, GTOracle) else model(fr["rgb"])
            for s in signs:
                b = s["box"]
                hit = [(t, c) for db, t, c in dets
                       if box_iou(b, db) >= 0.1 or (b[0] <= (db[0] + db[2]) / 2 <= b[2] and b[1] <= (db[1] + db[3]) / 2 <= b[3])]
                text = " ".join(t for t, _ in hit)
                row = {"walk_id": w["walk_id"], "lighting": w["lighting"], "frame": fr["frame"], "sign": s["actor"],
                       "gt_text": s["text"], "ocr_text": text, "detected": bool(hit),
                       "read": norm(text) == norm(s["text"]), "cer": cer(text, s["text"]) if hit else 1.0,
                       "distance_m": s["depth"], "height_px": s["height"]}
                rows.append(row)
                x0, y0, x1, y1 = b; pad = max(6, (y1 - y0) // 3)
                crops.append((row, fr["rgb"][max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad].copy()))
        print(f"== {w['walk_id']} ({w['lighting']}): signs visible in {n_vis} sampled frames", flush=True)

    def agg(rs):
        if not rs:
            return None
        return {"n": len(rs), "detected": float(np.mean([r["detected"] for r in rs]) * 100),
                "read": float(np.mean([r["read"] for r in rs]) * 100), "cer": float(np.mean([r["cer"] for r in rs]))}

    summary = {"model": args.model, "stride": args.stride, "overall": agg(rows),
               "ms_per_frame": {"median": float(np.median(model.times_ms)), "p95": float(np.percentile(model.times_ms, 95))}
               if model.times_ms else None,
               "by_sign": {a: agg([r for r in rows if r["sign"] == a]) for a in SIGN_TEXT},
               "by_lighting": {L: agg([r for r in rows if r["lighting"] == L]) for L in ("day", "overcast", "evening", "night")},
               "by_distance": {(f"{lo}-{hi} m" if hi < 100 else f">{lo} m"): agg([r for r in rows if lo <= r["distance_m"] < hi])
                               for lo, hi in DIST_BINS},
               "by_height_px": {(f"{lo}-{hi} px" if hi < 10000 else f">{lo} px"): agg([r for r in rows if lo <= r["height_px"] < hi])
                                for lo, hi in HEIGHT_BINS}}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "per_sign.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    # legibility figure: 10 crops from near to far
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    crops.sort(key=lambda c: c[0]["distance_m"])
    pick = [crops[int(k)] for k in np.linspace(0, len(crops) - 1, min(10, len(crops)))]
    fig, ax = plt.subplots(2, 5, figsize=(16, 5))
    for a, (r, im) in zip(ax.ravel(), pick):
        a.imshow(im); a.axis("off")
        ok = "✓" if r["read"] else ("~" if r["detected"] else "✗")
        a.set_title(f"{r['distance_m']:.1f} m · {r['height_px']} px · {r['lighting']}\nGT \"{r['gt_text']}\" → \"{r['ocr_text'] or '—'}\" {ok}",
                    fontsize=9, color="green" if r["read"] else ("orange" if r["detected"] else "red"))
    for a in ax.ravel()[len(pick):]:
        a.axis("off")
    fig.suptitle(f"{args.model}: sign legibility from near to far"); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "legibility.png"), dpi=80); plt.close(fig)

    def table(title, d):
        T = [f"| {title} | Signs | Detected (%) ↑ | Read exactly (%) ↑ | CER ↓ |", "|---|---|---|---|---|"]
        for k, a in d.items():
            if a:
                T.append(f"| {k} | {a['n']} | {a['detected']:.1f} | {a['read']:.1f} | {a['cer']:.2f} |")
        return T + [""]

    o = summary["overall"]
    L = [f"# Scene-text benchmark: `{args.model}`", "",
         f"Four wall signs, every {args.stride}th frame, sign visible if ≥ {args.min_pixels} px", ""]
    if summary["ms_per_frame"]:
        L += [f"**Speed:** {summary['ms_per_frame']['median']:.0f} ms/frame median (detection + recognition, full frame)", ""]
    L += [f"**Overall:** {o['n']} sign views · detected {o['detected']:.1f} % · read exactly {o['read']:.1f} % · CER {o['cer']:.2f}", ""]
    L += table("Distance", summary["by_distance"]) + table("Sign height on screen", summary["by_height_px"])
    L += table("Lighting", summary["by_lighting"]) + table("Sign", {f'{k} "{SIGN_TEXT[k]}"': v for k, v in summary["by_sign"].items()})
    L += ["![Legibility from near to far](figures/legibility.png)", ""]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"Wrote {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
