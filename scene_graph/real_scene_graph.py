"""Scene graph of a REAL first-person video: COLMAP camera path + metric depth + OWLv2 + SegFormer.

Usage (from the repo root, venv active; `pip install pycolmap imageio-ffmpeg` once):
    python scene_graph/real_scene_graph.py data/real/home_walk_01.mp4 --long-side-fov 72 --fp16
    python scene_graph/real_scene_graph.py data/real/home_walk_01.mp4 --long-side-fov 72 --fp16 --reuse-sfm

Steps
 1. Frames every 0.1 s -> COLMAP (SIFT, sequential matching, incremental SfM). Plain walls break the
    reconstruction into chunks; each chunk has its own unknown scale.
 2. DA V2 Metric (FOV-corrected) at the SfM keypoints -> one metric scale per chunk; chunks are then
    stitched in time order (shared frame, or bridging the short gap between them).
 3. Each frame's depth map is rescaled to agree with the SfM points it sees (a per-frame fix for the
    scale jitter that broke the synthetic scene graph in 4.2).
 4. Floor plane from SegFormer floor pixels -> world rotated so the floor is z = 0, longest wall along X.
 5. OWLv2 with home prompts -> depth-based object masks -> scene graph; walls; support relations.
 6. Video (frame with 3D boxes | live top-down map), final map PNG, graph JSON.

PRIVACY: everything is written to data/real/_out/<video>/scene_graph/ (git-ignored). Copy what you
want to publish into eval/results/real/ yourself.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "eval"))
from scene_graph.sfm import extract_frames, run_colmap, load_chunks, stitch_chunks, frame_time   # noqa: E402
from scene_graph.graph import SceneGraph, Detection, mask_from_box                              # noqa: E402
from scene_graph.layout import fit_plane_ransac, plane_to_z0, yaw_to_wall, estimate_walls, support_relations  # noqa: E402
import scene_graph.render_graph as R                                                             # noqa: E402

HOME_PROMPTS = {"bed": "a bed", "wardrobe": "a wardrobe", "desk": "a desk", "chair": "a chair",
                "armchair": "an armchair", "table": "a table", "shelf": "a shelf", "dresser": "a chest of drawers",
                "lamp": "a lamp", "plant": "a potted plant", "tv": "a television", "door": "a door",
                "window": "a window", "rug": "a rug"}
# near-synonyms OWLv2 flips between on the same object -> one class in the graph
CANONICAL = {"armchair": "chair", "desk": "table", "dresser": "cabinet", "wardrobe": "cabinet", "shelf": "cabinet"}
MODEL_IMPLIED_FOV = 74.0


def color_for(cls):
    if cls in R.PALETTE:
        return R.PALETTE[cls]
    h = (abs(hash(cls)) % 180)
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 200, 210]]]), cv2.COLOR_HSV2RGB)[0, 0])


def backproject(depth, K, c2w, mask, stride=4):
    m = mask[::stride, ::stride] & (depth[::stride, ::stride] > 0)
    v, u = np.nonzero(m); z = depth[::stride, ::stride][v, u]; v, u = v * stride, u * stride
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return P @ c2w[:3, :3].T + c2w[:3, 3]


def write_video(frames_dir, n, fps, out):
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = None
    if ff:
        subprocess.run([ff, "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "24", out], check=True)
    else:
        first = cv2.imread(os.path.join(frames_dir, "00000.png"))
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
        for k in range(n):
            vw.write(cv2.imread(os.path.join(frames_dir, f"{k:05d}.png")))
        vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--long-side-fov", type=float, required=True)
    ap.add_argument("--sfm-every-sec", type=float, default=0.1)
    ap.add_argument("--graph-every", type=int, default=3, help="run the models on every Nth registered frame")
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--camera-height", type=float, default=None,
                    help="known camera height above the floor (m), e.g. 1.5 for a phone held at eye level; "
                         "sets the metric scale instead of the depth model")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--reuse-sfm", action="store_true", help="skip COLMAP if it already ran")
    ap.add_argument("--fps", type=int, default=8, help="output video frame rate")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    out = os.path.join(REPO, "data", "real", "_out", name, "scene_graph"); os.makedirs(out, exist_ok=True)
    img_dir, work = os.path.join(out, "frames"), os.path.join(out, "colmap")
    t0 = time.time()

    # ---- 1. COLMAP
    if not (args.reuse_sfm and os.path.isdir(os.path.join(work, "sparse"))):
        shutil.rmtree(img_dir, ignore_errors=True)
        names, _ = extract_frames(args.video, args.sfm_every_sec, args.max_side, img_dir)
        h, w = cv2.imread(os.path.join(img_dir, names[0])).shape[:2]
        f_init = (max(h, w) / 2) / np.tan(np.radians(args.long_side_fov) / 2)
        print(f"[{time.time() - t0:.0f}s] {len(names)} frames -> COLMAP (this is the slow part) ...", flush=True)
        run_colmap(img_dir, work, f_init, verbose=True)
    all_names = sorted(os.listdir(img_dir))
    chunks = load_chunks(work)
    print(f"[{time.time() - t0:.0f}s] COLMAP chunks: " + ", ".join(f"{len(c['poses'])} frames {c['t0']:.1f}-{c['t1']:.1f}s" for c in chunks), flush=True)

    # ---- models
    import torch
    import depth_benchmark as DB, seg_benchmark as SB, det_benchmark as TB
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16 and device == "cuda"
    depth_model = DB.HFDepthModel(DB.MODELS["da2-metric-indoor-small"][0], fp16, device)
    seg_classes = ["unlabelled", "wall", "floor", "ceiling"]
    seg = SB.SegFormerModel(SB.MODELS["segformer-b2-ade"][0], seg_classes, fp16, device)
    cls_list = list(HOME_PROMPTS)
    det = TB.OWLv2(TB.MODELS["owlv2-base"], [HOME_PROMPTS[c] for c in cls_list], fp16, device)
    # FOV correction for the metric depth model: use COLMAP's self-calibrated focal length, not a guess
    Kc = max(chunks, key=lambda c: len(c["poses"]))["K"]
    h0, w0 = cv2.imread(os.path.join(img_dir, all_names[0])).shape[:2]
    fov_long = float(2 * np.degrees(np.arctan((max(h0, w0) / 2) / Kc[0, 0])))
    fov_factor = float(np.tan(np.radians(MODEL_IMPLIED_FOV) / 2) / np.tan(np.radians(fov_long) / 2))
    print(f"COLMAP focal {Kc[0, 0]:.0f} px -> long-side FOV {fov_long:.1f} deg (you passed {args.long_side_fov:g}); "
          f"metric depth correction x{fov_factor:.2f}", flush=True)

    # ---- 2. per-frame inference on every Nth registered frame + chunk scales
    registered = sorted({n for c in chunks for n in c["poses"]}, key=frame_time)
    use = registered[::args.graph_every]
    data, chunk_ratios = {}, [[] for _ in chunks]
    for k, n in enumerate(use):
        bgr = cv2.imread(os.path.join(img_dir, n)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        d = (depth_model(rgb) * fov_factor).astype(np.float32)
        sem = seg(rgb)
        boxes = [(CANONICAL.get(cls_list[l], cls_list[l]), s, b) for s, l, b in det(rgb, args.score_thr)]
        data[n] = {"rgb": rgb, "depth": d, "sem": sem, "boxes": boxes, "frame_ratio": {}}
        for ci, c in enumerate(chunks):
            if n in c["poses"]:
                uv, z = c["poses"][n]["uv"], c["poses"][n]["z"]
                if len(z):
                    u = np.clip(np.round(uv[:, 0]).astype(int), 0, d.shape[1] - 1)
                    v = np.clip(np.round(uv[:, 1]).astype(int), 0, d.shape[0] - 1)
                    m = d[v, u]; ok = (m > 0.2) & (m < 5.0)
                    r = m[ok] / z[ok]
                    if len(r) >= 30:
                        chunk_ratios[ci].append(r)
                        data[n]["frame_ratio"][ci] = float(np.median(r))
        if (k + 1) % 25 == 0:
            print(f"  models {k + 1}/{len(use)}", flush=True)
    scales = [float(np.median(np.concatenate(r))) if r else 1.0 for r in chunk_ratios]
    world, link_info = stitch_chunks(chunks, scales)
    for li in link_info:
        print(f"  chunk {li['chunk']}: {li['frames']} frames {li['t0']:.1f}-{li['t1']:.1f}s, metric scale {li['scale']:.3f}, {li['link']}")

    # ---- 3. per-frame depth rescaled to agree with the SfM points
    K = Kc
    frames = [n for n in use if n in world]
    for n in frames:
        ci = next(i for i, c in enumerate(chunks) if n in c["poses"])
        fr = data[n]["frame_ratio"].get(ci)
        if fr:
            data[n]["depth"] *= scales[ci] / fr

    # ---- 4. floor -> z = 0, longest wall along X
    F = np.concatenate([backproject(data[n]["depth"], K, world[n], data[n]["sem"] == 2) for n in frames])
    nrm, c, inl = fit_plane_ransac(F[np.random.default_rng(0).choice(len(F), min(len(F), 60000), replace=False)])
    cams = np.array([world[n][:3, 3] for n in frames])
    if np.median((cams - c) @ nrm) < 0:
        nrm = -nrm
    T = plane_to_z0(nrm, c)
    world = {n: T @ P for n, P in world.items()}

    # drop badly registered frames: camera height far from the median, or faster than walking allows
    def plausible(fr_list, max_dz=0.35, max_speed=2.0):
        z = np.array([world[n][2, 3] for n in fr_list]); med = np.median(z)
        keep = [n for n, zz in zip(fr_list, z) if abs(zz - med) < max_dz]
        out_ = [keep[0]] if keep else []
        for n in keep[1:]:
            dt = max(frame_time(n) - frame_time(out_[-1]), 0.05)
            if np.linalg.norm(world[n][:3, 3] - world[out_[-1]][:3, 3]) <= max_speed * dt + 0.15:
                out_.append(n)
        return out_
    good = plausible(frames)
    print(f"  pose sanity check: kept {len(good)} of {len(frames)} frames", flush=True)
    frames = good
    F = np.concatenate([backproject(data[n]["depth"], K, world[n], data[n]["sem"] == 2) for n in frames])
    nrm, c, inl = fit_plane_ransac(F[np.random.default_rng(1).choice(len(F), min(len(F), 60000), replace=False)])
    if np.median((np.array([world[n][:3, 3] for n in frames]) - c) @ nrm) < 0:
        nrm = -nrm
    T2 = plane_to_z0(nrm, c)
    world = {n: T2 @ P for n, P in world.items()}
    eye_data = float(np.median([world[n][2, 3] for n in frames]))
    if args.camera_height:
        k = args.camera_height / eye_data
        for n in world:
            world[n][:3, 3] *= k
        for n in frames:
            data[n]["depth"] *= k
        print(f"  camera height from depth model {eye_data:.2f} m -> rescaled x{k:.2f} to the given {args.camera_height:.2f} m", flush=True)
    Wp = np.concatenate([backproject(data[n]["depth"], K, world[n], data[n]["sem"] == 1) for n in frames])
    walls = estimate_walls(Wp, {"height": 0.0})
    Y = yaw_to_wall(walls)
    world = {n: Y @ P for n, P in world.items()}
    Wp = Wp @ Y[:3, :3].T
    walls = estimate_walls(Wp, {"height": 0.0})
    eye = float(np.median([world[n][2, 3] for n in frames]))
    print(f"[{time.time() - t0:.0f}s] floor fitted ({inl.sum()} inliers); camera height above floor {eye:.2f} m "
          f"(depth-model scale: {eye_data:.2f} m); {len(walls)} walls")

    # ---- 5./6. scene graph + rendering
    R.PALETTE.update({c: color_for(c) for c in set(cls_list) | set(CANONICAL.values()) if c not in R.PALETTE})
    sg = SceneGraph(match_dist=0.5)
    tmp = tempfile.mkdtemp(prefix="egoscene_real_")
    traj = []
    for k, n in enumerate(frames):
        dd = data[n]
        dets = [Detection(cls=c, mask=mask_from_box(b, dd["depth"]), score=s) for c, s, b in dd["boxes"]]
        t = frame_time(n)
        sg.update(dets, dd["depth"], K, world[n], t)
        traj.append(world[n][:3, 3])
        view = R.draw_boxes(dd["rgb"], sg.objects, K, world[n], t)
        m = R.topdown(sg.objects, np.array(traj), Wp[::5], size=view.shape[0], cur=traj[-1])
        m = cv2.resize(m, (int(m.shape[1] * view.shape[0] / m.shape[0]), view.shape[0]))
        cv2.putText(m, f"t={t:4.1f}s  objects: {sum(o.n_obs >= 2 for o in sg.objects)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(tmp, f"{k:05d}.png"), cv2.cvtColor(np.hstack([view, m]), cv2.COLOR_RGB2BGR))
    mp4 = os.path.join(out, f"{name}_graph.mp4")
    write_video(tmp, len(frames), args.fps, mp4)
    shutil.rmtree(tmp, ignore_errors=True)
    final = R.topdown(sg.objects, np.array(traj), Wp[::3], size=900)
    cv2.imwrite(os.path.join(out, f"{name}_map.png"), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    # coloured point cloud of the room (2 cm voxels) for MeshLab / CloudCompare / Unreal
    pts, cols = [], []
    for n in frames:
        dd = data[n]; m = (dd["depth"] > 0.2) & (dd["depth"] < 6.0)
        p = backproject(dd["depth"], K, world[n], m, stride=3)
        vv, uu = np.nonzero(m[::3, ::3]); cols.append(dd["rgb"][vv * 3, uu * 3]); pts.append(p)
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    key = np.floor(pts / 0.02).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    pts, cols = pts[first], cols[first]
    ply = os.path.join(out, f"{name}_room.ply")
    with open(ply, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                  f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        fh.write(rec.tobytes())
    # same cloud in Unreal coordinates (centimetres, left-handed: Y flipped) for the LiDAR Point Cloud plugin
    rec_ue = rec.copy(); rec_ue["x"] = pts[:, 0] * 100; rec_ue["y"] = -pts[:, 1] * 100; rec_ue["z"] = pts[:, 2] * 100
    ply_ue = os.path.join(out, f"{name}_room_ue.ply")
    with open(ply_ue, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                  f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        fh.write(rec_ue.tobytes())
    # ASCII fallback for the LiDAR plugin (x y z r g b, Unreal cm), thinned to 4 cm voxels to keep it small
    k4 = np.floor(pts / 0.04).astype(np.int64); _, f4 = np.unique(k4, axis=0, return_index=True)
    txt = os.path.join(out, f"{name}_room_ue.txt")
    np.savetxt(txt, np.column_stack([pts[f4, 0] * 100, -pts[f4, 1] * 100, pts[f4, 2] * 100, cols[f4]]),
               fmt="%.1f %.1f %.1f %d %d %d")
    print(f"Point cloud: {len(pts):,} points -> {ply} (+ Unreal versions {os.path.basename(ply_ue)}, "
          f"{os.path.basename(txt)} with {len(f4):,} points)")
    objs = [o for o in sg.objects if o.n_obs >= 3]
    rel = support_relations(objs, {"height": 0.0})
    json.dump({"video": os.path.basename(args.video), "frames_used": len(frames), "frames_total": len(all_names),
               "registered": len(registered), "chunks": link_info, "camera_height_m": eye,
               "camera_height_from_depth_model_m": eye_data, "camera_height_given_m": args.camera_height, "walls": walls,
               "colmap_focal_px": float(Kc[0, 0]), "long_side_fov_deg": fov_long, "depth_fov_factor": fov_factor,
               "camera_path": [world[n][:3, 3].round(3).tolist() for n in frames],
               "objects": [{**o.to_dict(), "support": rel[o.id]} for o in objs],
               "ms_per_frame": {"depth": float(np.median(depth_model.times_ms)), "segformer": float(np.median(seg.times_ms)),
                                "owlv2": float(np.median(det.times_ms)),
                                "graph": float(np.median([a + b + c for a, b, c in zip(*sg.timings.values())]))}},
              open(os.path.join(out, f"{name}_graph.json"), "w"), indent=1)
    print(f"[{time.time() - t0:.0f}s] {len(objs)} objects: " + ", ".join(f"{o.cls}({rel[o.id]})" for o in objs))
    print(f"Wrote {mp4}, {name}_map.png and {name}_graph.json in {out}")


if __name__ == "__main__":
    main()
