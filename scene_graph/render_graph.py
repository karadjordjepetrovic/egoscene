"""Visualize the scene graph: a top-down map and a video with the graph's 3D boxes drawn into the walk.

Usage (from the repo root):
    python scene_graph/render_graph.py walk_04                 # GT inputs, live graph growing over the walk
    python scene_graph/render_graph.py walk_04 --stride 2 --fps 15

Outputs in eval/results/scene_graph/render/: <walk>_map.png and <walk>_graph.mp4
"""
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from dataset.egoscene import list_walks, load_meta, load_frame, backproject   # noqa: E402
from scene_graph.graph import SceneGraph                                     # noqa: E402
from scene_graph.eval_graph import K_of, object_groups, gt_detections         # noqa: E402

PALETTE = {"table": (230, 159, 0), "chair": (86, 180, 233), "sofa": (0, 158, 115), "shelf": (240, 228, 66),
           "plant": (0, 114, 178), "lamp": (213, 94, 0), "counter": (204, 121, 167), "sink": (120, 120, 255),
           "stove": (255, 80, 80), "tv": (180, 180, 180), "sign": (255, 255, 255)}
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(bmin, bmax):
    return np.array([[x, y, z] for z in (bmin[2], bmax[2]) for y in (bmin[1], bmax[1]) for x in (bmin[0], bmax[0])])


def draw_boxes(rgb, objects, K, c2w, t, min_obs=2):
    img = rgb.copy()
    w2c = np.linalg.inv(c2w)
    H, W = img.shape[:2]
    for o in objects:
        if o.n_obs < min_obs:
            continue
        P = corners(o.box_min, o.box_max) @ w2c[:3, :3].T + w2c[:3, 3]
        if (P[:, 2] < 0.1).any():
            continue
        uv = (P[:, :2] / P[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        if (uv[:, 0].max() < 0) or (uv[:, 0].min() > W) or (uv[:, 1].max() < 0) or (uv[:, 1].min() > H):
            continue
        col = PALETTE.get(o.cls, (255, 255, 255))
        fresh = t - o.last_seen < 0.2
        for a, b in EDGES:
            cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), col, 2 if fresh else 1, cv2.LINE_AA)
        top = uv[np.argmin(uv[:, 1])]
        label = f"#{o.id} {o.cls} {o.centroid[2]:.2f}m"
        cv2.putText(img, label, (int(top[0]), max(int(top[1]) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (int(top[0]), max(int(top[1]) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


def topdown(objects, traj, wall_pts, size=720, margin=0.4, cur=None):
    pts = np.concatenate([wall_pts[:, :2], traj[:, :2]])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    s = (size - 20) / max(hi - lo)
    to_px = lambda xy: np.int32((np.asarray(xy) - lo) * s + 10) * [1, 1]
    img = np.full((int((hi[1] - lo[1]) * s) + 20, int((hi[0] - lo[0]) * s) + 20, 3), 250, np.uint8)
    wp = to_px(wall_pts[:, :2])
    img[np.clip(wp[:, 1], 0, img.shape[0] - 1), np.clip(wp[:, 0], 0, img.shape[1] - 1)] = (150, 150, 150)
    img = cv2.flip(img, 0)                                     # +y up
    flipy = lambda p: (int(p[0]), img.shape[0] - 1 - int(p[1]))
    tp = [flipy(p) for p in to_px(traj[:, :2])]
    for a, b in zip(tp, tp[1:]):
        cv2.line(img, a, b, (60, 60, 220), 2, cv2.LINE_AA)
    if cur is not None:
        cv2.circle(img, flipy(to_px(cur[:2])), 6, (60, 60, 220), -1)
    for o in objects:
        if o.n_obs < 2:
            continue
        col = PALETTE.get(o.cls, (0, 0, 0)); col = tuple(int(c * 0.8) for c in col)
        a, b = flipy(to_px(o.box_min[:2])), flipy(to_px(o.box_max[:2]))
        cv2.rectangle(img, a, b, col, 2)
        cv2.putText(img, o.cls, (min(a[0], b[0]), min(a[1], b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walk")
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "scene_graph", "render"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    w = next(x for x in list_walks(args.data) if x["walk_id"] == args.walk)
    wdir = os.path.join(args.data, w["path"]); meta = load_meta(wdir)
    K, groups, fps = K_of(meta), object_groups(meta), meta.get("fps", 30)

    # wall outline for the map: slice of fused depth between 0.5 and 2.0 m height
    wall = []
    for i in range(0, len(meta["frames"]), 25):
        fr = load_frame(wdir, i, meta)
        p, _ = backproject(fr["depth"], K, fr["c2w"])
        p = p[(p[:, 2] > 0.5) & (p[:, 2] < 2.0)]
        wall.append(p[np.random.default_rng(i).choice(len(p), min(len(p), 15000), replace=False)])
    wall = np.concatenate(wall)

    sg = SceneGraph()
    traj = []
    tmp = tempfile.mkdtemp(prefix=f"egoscene_{args.walk}_")      # frames go to a local temp dir, not the repo
    n = 0
    for i in range(0, len(meta["frames"]), args.stride):
        fr = load_frame(wdir, i, meta)
        t = fr["frame"] / fps
        sg.update(gt_detections(fr, groups, 300), fr["depth"], K, fr["c2w"], t)
        traj.append(fr["c2w"][:3, 3])
        view = draw_boxes(fr["rgb"], sg.objects, K, fr["c2w"], t)
        m = topdown(sg.objects, np.array(traj), wall, size=view.shape[0], cur=traj[-1])
        m = cv2.resize(m, (int(m.shape[1] * view.shape[0] / m.shape[0]), view.shape[0]))
        cv2.putText(m, f"{args.walk} ({w['lighting']}) t={t:4.1f}s  objects: {sum(o.n_obs >= 2 for o in sg.objects)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(tmp, f"{n:05d}.png"), cv2.cvtColor(np.hstack([view, m]), cv2.COLOR_RGB2BGR))
        n += 1
    final = topdown(sg.objects, np.array(traj), wall, size=900)
    cv2.imwrite(os.path.join(args.out, f"{args.walk}_map.png"), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    mp4 = os.path.join(args.out, f"{args.walk}_graph.mp4")
    ff = shutil.which("ffmpeg")
    if ff:
        subprocess.run([ff, "-v", "error", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", mp4], check=True)
    else:
        first = cv2.imread(os.path.join(tmp, "00000.png"))
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (first.shape[1], first.shape[0]))
        for k in range(n):
            vw.write(cv2.imread(os.path.join(tmp, f"{k:05d}.png")))
        vw.release()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {mp4} ({n} frames) and {args.walk}_map.png")


if __name__ == "__main__":
    main()
