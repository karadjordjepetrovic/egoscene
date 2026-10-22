"""Convert one rendered EgoScene walk (UE5 Movie Render Queue output) into the dataset format.

Usage (one walk):
    python dataset/convert_walk.py <render_dir> <poses_ue.json> <labels.json> <out_dir>

To convert every walk in the manifest, use dataset/convert_all.py instead.
"""
import os, json, glob, shutil, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"   # must be set before importing cv2
import cv2, numpy as np
import OpenEXR

RGB_PASS, DEPTH_PASS, ID_PASS = "FinalImage", "FinalImageGT_Depth", "FinalImageGT_InstanceID"
W, H, HFOV_DEG = 1280, 720, 90.0
NO_DEPTH_CLASSES = ("outdoor",)   # backdrop "sky" geometry: its depth is not real, mark it invalid


def read_exr_r(path):
    """Return the red channel of an EXR as float32."""
    with OpenEXR.File(path, separate_channels=True) as f:
        ch = f.channels()
        key = next((k for k in ch if k == "R" or k.endswith(".R")), None)
        if key is None:
            sys.exit(f"No R channel in {path}. Channels found: {list(ch)}")
        return np.asarray(ch[key].pixels, dtype=np.float32)


def convert_walk(render_dir, poses_path, labels_path, out_dir, extra_meta=None, log_every=100):
    fx = fy = W / (2 * np.tan(np.radians(HFOV_DEG) / 2)); cx, cy = W / 2, H / 2
    labels = json.load(open(labels_path))
    classes = labels["classes"]
    id_to_class = np.zeros(256, np.uint8)               # 0 = unlabelled
    known_ids = np.zeros(256, bool); known_ids[0] = True
    no_depth_ids = []
    for iid, info in labels["instances"].items():
        id_to_class[int(iid)] = classes.index(info["class"]) + 1
        known_ids[int(iid)] = True
        if info["class"] in NO_DEPTH_CLASSES:
            no_depth_ids.append(int(iid))

    for d in ("rgb", "depth", "instance", "semantic"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    M = np.diag([1.0, -1.0, 1.0])                        # UE (left-handed) -> right-handed world
    poses = json.load(open(poses_path))
    unknown_ids = set()
    frames_out = []
    prev_depth, static_frames = None, 0
    for n, f in enumerate(poses["frames"]):
        i, name = f["frame"], f"{f['frame']:06d}"
        find = lambda p, ext: sorted(glob.glob(os.path.join(render_dir, p, f"*{i:04d}.{ext}")))[0]

        shutil.copy(find(RGB_PASS, "png"), os.path.join(out_dir, "rgb", name + ".png"))

        ids = np.round(read_exr_r(find(ID_PASS, "exr"))).clip(0, 255).astype(np.uint8)
        unknown_ids.update(np.unique(ids[~known_ids[ids]]).tolist())

        depth_mm = np.round(read_exr_r(find(DEPTH_PASS, "exr")) * 10.0)   # cm -> mm
        depth_mm[(depth_mm <= 0) | (depth_mm > 65535)] = 0               # far = invalid
        if no_depth_ids:
            depth_mm[np.isin(ids, no_depth_ids)] = 0                       # outdoor backdrop = invalid
        cv2.imwrite(os.path.join(out_dir, "depth", name + ".png"), depth_mm.astype(np.uint16))
        if prev_depth is not None and np.array_equal(depth_mm, prev_depth):
            static_frames += 1
        prev_depth = depth_mm

        cv2.imwrite(os.path.join(out_dir, "instance", name + ".png"), ids)
        cv2.imwrite(os.path.join(out_dir, "semantic", name + ".png"), id_to_class[ids])

        fwd, right, up = (np.array(f[k]) for k in ("forward", "right", "up"))
        R = np.stack([M @ right, -(M @ up), M @ fwd], axis=1)  # columns = OpenCV camera X, Y, Z in world
        c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = M @ np.array(f["loc_cm"]) / 100.0
        frames_out.append({"frame": i, "rgb": f"rgb/{name}.png", "depth": f"depth/{name}.png",
                           "instance": f"instance/{name}.png", "semantic": f"semantic/{name}.png",
                           "c2w": c2w.tolist()})
        if log_every and (n + 1) % log_every == 0:
            print(f"  {n + 1}/{len(poses['frames'])} frames", flush=True)

    meta = {"width": W, "height": H, "fps": poses["fps"],
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "depth_unit": "millimetres, 0 = invalid (incl. outdoor backdrop)",
            "convention": "world: right-handed, Z up, metres; camera: OpenCV (x right, y down, z forward); c2w = camera-to-world",
            "classes": ["unlabelled"] + classes, "instances": labels["instances"]}
    for k in ("lighting", "speed_cm_s", "seed"):
        if k in poses:
            meta[k] = poses[k]
    meta.update(extra_meta or {})
    meta["frames"] = frames_out
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)

    if static_frames > len(frames_out) // 10:
        print(f"  WARNING: {static_frames} frames have exactly the same depth as the previous frame. "
              f"The render camera probably didn't move (check the sequence's camera binding).")
    if unknown_ids:
        print(f"  WARNING: instance IDs {sorted(unknown_ids)} appear in the renders but not in {labels_path}. "
              f"Re-run label_objects.py in Unreal and copy the labels JSON again.")
    print(f"Wrote {len(frames_out)} frames to {out_dir}")
    return len(frames_out), sorted(unknown_ids)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    convert_walk(*sys.argv[1:5])
