import os, json, glob, shutil, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"   # must be set before importing cv2
import cv2, numpy as np
import OpenEXR

def read_exr_r(path):
    """Return the red channel of an EXR as float32."""
    with OpenEXR.File(path, separate_channels=True) as f:
        ch = f.channels()
        key = next((k for k in ch if k == "R" or k.endswith(".R")), None)
        if key is None:
            sys.exit(f"No R channel in {path}. Channels found: {list(ch)}")
        return np.asarray(ch[key].pixels, dtype=np.float32)
        
RGB_PASS, DEPTH_PASS, ID_PASS = "FinalImage", "FinalImageGT_Depth", "FinalImageGT_InstanceID"
RENDER_DIR, POSES_UE, LABELS, OUT = sys.argv[1:5]
W, H, HFOV_DEG = 1280, 720, 90.0

fx = fy = W / (2 * np.tan(np.radians(HFOV_DEG) / 2)); cx, cy = W / 2, H / 2
labels = json.load(open(LABELS))
classes = labels["classes"]
id_to_class = np.zeros(256, np.uint8)               # 0 = unlabelled
for iid, info in labels["instances"].items():
    id_to_class[int(iid)] = classes.index(info["class"]) + 1

for d in ("rgb", "depth", "instance", "semantic"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

M = np.diag([1.0, -1.0, 1.0])                        # UE (left-handed) -> right-handed world
poses = json.load(open(POSES_UE))
frames_out = []
for f in poses["frames"]:
    i, name = f["frame"], f"{f['frame']:06d}"
    find = lambda p, ext: sorted(glob.glob(os.path.join(RENDER_DIR, p, f"*{i:04d}.{ext}")))[0]

    shutil.copy(find(RGB_PASS, "png"), os.path.join(OUT, "rgb", name + ".png"))

    depth_cm = read_exr_r(find(DEPTH_PASS, "exr"))
    depth_mm = np.round(depth_cm * 10.0)
    depth_mm[(depth_mm <= 0) | (depth_mm > 65535)] = 0                              # sky/far = invalid
    cv2.imwrite(os.path.join(OUT, "depth", name + ".png"), depth_mm.astype(np.uint16))

    ids = np.round(read_exr_r(find(ID_PASS, "exr"))).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "instance", name + ".png"), ids)
    cv2.imwrite(os.path.join(OUT, "semantic", name + ".png"), id_to_class[ids])

    fwd, right, up = (np.array(f[k]) for k in ("forward", "right", "up"))
    R = np.stack([M @ right, -(M @ up), M @ fwd], axis=1)  # columns = OpenCV camera X, Y, Z in world
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = M @ np.array(f["loc_cm"]) / 100.0
    frames_out.append({"frame": i, "rgb": f"rgb/{name}.png", "depth": f"depth/{name}.png",
                       "instance": f"instance/{name}.png", "semantic": f"semantic/{name}.png",
                       "c2w": c2w.tolist()})

json.dump({"width": W, "height": H, "fps": poses["fps"],
           "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
           "depth_unit": "millimetres, 0 = invalid",
           "convention": "world: right-handed, Z up, metres; camera: OpenCV (x right, y down, z forward); c2w = camera-to-world",
           "classes": ["unlabelled"] + classes, "instances": labels["instances"],
           "frames": frames_out}, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print(f"Wrote {len(frames_out)} frames to {OUT}")