"""Camera poses for a real video with COLMAP (pycolmap): frames -> SIFT -> sequential matching ->
incremental structure-from-motion. Poses come out up to an unknown scale; metric_scale() fixes it
with a metric depth model.

    frames = extract_frames("data/real/home_walk_01.mp4", every_sec=0.33, max_side=1280, out_dir=...)
    rec = run_colmap(img_dir, work_dir, f_init)        # -> {name: {"c2w": 4x4, "uv": (N,2), "z": (N,)}}, K
"""
import os, shutil
import numpy as np
import cv2


def extract_frames(video, every_sec, max_side, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_sec)), 1)
    names, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = bgr.shape[:2]
            s = min(1.0, max_side / max(h, w))
            if s < 1:
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            name = f"{len(names):05d}_t{i / fps:06.2f}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(name)
        i += 1
    return names, fps


def _matrix(rigid):
    r = rigid() if callable(rigid) else rigid
    M = np.asarray(r.matrix())
    T = np.eye(4); T[:3, :4] = M[:3, :4]
    return T


def run_colmap(img_dir, work_dir, f_init, overlap=15, verbose=False, low_texture=True):
    """low_texture: settings for plain indoor walls (more, weaker SIFT features; easier registration)."""
    import pycolmap
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    db = os.path.join(work_dir, "database.db")
    first = cv2.imread(os.path.join(img_dir, sorted(os.listdir(img_dir))[0]))
    h, w = first.shape[:2]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{f_init},{w / 2},{h / 2},0"
    ext = pycolmap.FeatureExtractionOptions()
    if low_texture:
        ext.max_image_size = 1600
        ext.sift.max_num_features = 16384
        ext.sift.peak_threshold = 0.002          # default 0.0067: keep weaker features on plain walls
        ext.sift.domain_size_pooling = True
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    if hasattr(pairing, "loop_detection"):
        pairing.loop_detection = False
    pycolmap.match_sequential(db, pairing_options=pairing)
    opts = pycolmap.IncrementalPipelineOptions()
    if low_texture:
        opts.min_num_matches = 10
        opts.mapper.init_min_num_inliers = 40
        opts.mapper.abs_pose_min_num_inliers = 15
        opts.mapper.abs_pose_min_inlier_ratio = 0.15
        opts.mapper.init_min_tri_angle = 4
        opts.min_model_size = 5
    recs = pycolmap.incremental_mapping(db, img_dir, os.path.join(work_dir, "sparse"), options=opts)
    if not recs:
        raise RuntimeError("COLMAP could not reconstruct the video (too little texture / motion?)")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    cam = next(iter(rec.cameras.values()))
    p = np.asarray(cam.params)               # SIMPLE_RADIAL: f, cx, cy, k
    K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
    out = {}
    for img in rec.images.values():
        if hasattr(img, "has_pose") and not img.has_pose:
            continue
        w2c = _matrix(img.cam_from_world)
        uv, z = [], []
        for p2 in img.points2D:
            if p2.has_point3D():
                X = rec.points3D[p2.point3D_id].xyz
                zc = (w2c[:3, :3] @ X + w2c[:3, 3])[2]
                if zc > 0:
                    uv.append(p2.xy); z.append(zc)
        out[img.name] = {"c2w": np.linalg.inv(w2c), "uv": np.array(uv).reshape(-1, 2), "z": np.array(z)}
    if verbose:
        print(f"COLMAP: {len(out)} of {len(os.listdir(img_dir))} frames registered, "
              f"{rec.num_points3D()} 3D points, focal {p[0]:.0f} px (init {f_init:.0f})")
    return out, K


def metric_scales(poses, depths, max_depth=5.0, min_pts=30):
    """Per-frame scale (metric depth / SfM depth) at the SfM keypoints, plus a robust global scale.
    depths: {name: (H, W) metric depth}. Returns (global_scale, {name: per-frame scale or None})."""
    per, allr = {}, []
    for name, d in depths.items():
        if name not in poses:
            continue
        uv, z = poses[name]["uv"], poses[name]["z"]
        if len(z) == 0:
            per[name] = None; continue
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, d.shape[1] - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, d.shape[0] - 1)
        m = d[v, u]
        ok = (m > 0.2) & (m < max_depth)
        r = m[ok] / z[ok]
        per[name] = float(np.median(r)) if len(r) >= min_pts else None
        allr.append(r)
    allr = np.concatenate(allr) if allr else np.array([1.0])
    return float(np.median(allr)), per


def frame_time(name):
    """'00012_t004.20.jpg' -> 4.2"""
    return float(name.split("_t")[1].rsplit(".", 1)[0])


def load_chunks(work_dir):
    """All COLMAP models in work_dir/sparse as chunks: [{"poses": {name: {...}}, "K": 3x3, "t0", "t1"}]."""
    import pycolmap
    base = os.path.join(work_dir, "sparse")
    chunks = []
    for d in sorted(os.listdir(base)):
        rec = pycolmap.Reconstruction(os.path.join(base, d))
        cam = next(iter(rec.cameras.values())); p = np.asarray(cam.params)
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
        poses = {}
        for img in rec.images.values():
            if hasattr(img, "has_pose") and not img.has_pose:
                continue
            w2c = _matrix(img.cam_from_world)
            uv, z = [], []
            for p2 in img.points2D:
                if p2.has_point3D():
                    X = rec.points3D[p2.point3D_id].xyz
                    zc = (w2c[:3, :3] @ X + w2c[:3, 3])[2]
                    if zc > 0:
                        uv.append(p2.xy); z.append(zc)
            poses[img.name] = {"c2w": np.linalg.inv(w2c), "uv": np.array(uv).reshape(-1, 2), "z": np.array(z)}
        if len(poses) >= 5:
            ts = [frame_time(n) for n in poses]
            chunks.append({"poses": poses, "K": K, "t0": min(ts), "t1": max(ts)})
    return chunks


def _rigid_from_pairs(A_pts, B_pts):
    """Rotation + translation (no scale) mapping B_pts onto A_pts (Kabsch)."""
    ca, cb = A_pts.mean(0), B_pts.mean(0)
    U, _, Vt = np.linalg.svd((B_pts - cb).T @ (A_pts - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = ca - Rm @ cb
    return T


def stitch_chunks(chunks, scales, pair_dt=0.15):
    """Put metric-scaled chunks into one world frame, in time order.
    scales: one metric scale per chunk (applied to its translations and sparse depths).
    Chunks fully inside a bigger chunk's time span are dropped. Each further chunk is linked to the
    frames already placed by (1) a shared frame, else (2) frames taken within `pair_dt` seconds of each
    other in both (camera centres aligned with Kabsch when there are enough, well spread), else
    (3) the temporally nearest pair of frames (assumes the camera barely moved in between)."""
    order = sorted(range(len(chunks)), key=lambda k: -len(chunks[k]["poses"]))
    kept = []
    for k in order:
        c = chunks[k]
        if any(chunks[j]["t0"] <= c["t0"] and c["t1"] <= chunks[j]["t1"] for j in kept):
            continue
        kept.append(k)
    kept.sort(key=lambda k: chunks[k]["t0"])
    world, info = {}, []
    for k in kept:
        c, s = chunks[k], scales[k]
        local = {}
        for n, p in c["poses"].items():
            T = p["c2w"].copy(); T[:3, 3] *= s
            local[n] = T
        if not world:
            A = np.eye(4); how = "first"
        else:
            shared = [n for n in local if n in world]
            wt = np.array([frame_time(n) for n in world]); wn = list(world)
            pairs = []
            for n in local:
                j = int(np.argmin(np.abs(wt - frame_time(n))))
                dt = abs(wt[j] - frame_time(n))
                pairs.append((dt, wn[j], n))
            pairs.sort()
            close = [p for p in pairs if p[0] <= pair_dt]
            if shared:
                n = sorted(shared)[-1]; A = world[n] @ np.linalg.inv(local[n]); how = f"shared frame {n}"
            elif len(close) >= 5 and np.ptp(np.array([world[a][:3, 3] for _, a, _ in close]), axis=0).max() > 0.3:
                A = _rigid_from_pairs(np.array([world[a][:3, 3] for _, a, _ in close]),
                                      np.array([local[b][:3, 3] for _, _, b in close]))
                how = f"aligned on {len(close)} near-simultaneous frame pairs"
            else:
                dt, a, b = pairs[0]
                A = world[a] @ np.linalg.inv(local[b]); how = f"bridged nearest frames {dt:.2f}s apart"
        for n, T in local.items():
            if n not in world:
                world[n] = A @ T
        info.append({"chunk": k, "frames": len(c["poses"]), "t0": c["t0"], "t1": c["t1"], "scale": s, "link": how})
    return world, info
