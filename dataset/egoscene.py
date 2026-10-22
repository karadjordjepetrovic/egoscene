"""Loader for the EgoScene dataset.

    from dataset.egoscene import load_walk, list_walks

    walk = load_walk("data/apartment_01/walk_04", frames=range(0, 400, 10))
    walk["rgb"].shape        # (N, 720, 1280, 3) uint8, RGB order
    walk["depth"].shape      # (N, 720, 1280) float32, metres, 0 = invalid
    walk["semantic"].shape   # (N, 720, 1280) uint8 class ids, see walk["classes"]
    walk["instance"].shape   # (N, 720, 1280) uint8 instance ids, see walk["instances"]
    walk["c2w"].shape        # (N, 4, 4) float64 camera-to-world, OpenCV camera, metres
    walk["K"]                # (3, 3) intrinsics

Conventions: world is right-handed, Z up, metres. Camera is OpenCV (x right, y down, z forward).
A full walk is ~1-2 GB in memory, so pass `frames` (indices into the walk) or `stride` for subsets.
"""
import json, os
import numpy as np
import cv2

__all__ = ["list_walks", "load_meta", "load_frame", "load_walk", "backproject", "to_torch"]


def list_walks(data_root="data"):
    """Return the dataset manifest (one dict per walk: walk_id, level, lighting, speed_cm_s, path, ...)."""
    return json.load(open(os.path.join(data_root, "manifest.json")))


def load_meta(walk_dir):
    return json.load(open(os.path.join(walk_dir, "meta.json")))


def _intrinsics(meta):
    k = meta["intrinsics"]
    return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=np.float64)


def load_frame(walk_dir, index, meta=None):
    """Load one frame (index into meta['frames']) as a dict of numpy arrays."""
    meta = meta or load_meta(walk_dir)
    f = meta["frames"][index]
    rd = lambda key, flags: cv2.imread(os.path.join(walk_dir, f[key]), flags)
    bgr = rd("rgb", cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(os.path.join(walk_dir, f["rgb"]))
    return {
        "frame": f["frame"],
        "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "depth": rd("depth", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0,
        "semantic": rd("semantic", cv2.IMREAD_UNCHANGED),
        "instance": rd("instance", cv2.IMREAD_UNCHANGED),
        "c2w": np.array(f["c2w"], dtype=np.float64),
    }


def load_walk(walk_dir, frames=None, stride=1):
    """Load a walk as aligned stacked arrays. `frames`: iterable of indices; default every `stride`-th frame."""
    meta = load_meta(walk_dir)
    idx = list(frames) if frames is not None else list(range(0, len(meta["frames"]), stride))
    per = [load_frame(walk_dir, i, meta) for i in idx]
    out = {k: np.stack([p[k] for p in per]) for k in ("rgb", "depth", "semantic", "instance", "c2w")}
    out.update({
        "frame": np.array([p["frame"] for p in per]),
        "K": _intrinsics(meta),
        "classes": meta["classes"],
        "instances": meta["instances"],
        "meta": {k: v for k, v in meta.items() if k != "frames"},
    })
    return out


def backproject(depth, K, c2w=None, mask=None, rgb=None):
    """Depth (H, W, metres) -> 3D points (N, 3), in camera space, or world space if c2w is given.
    Returns (points, colors) where colors is None unless rgb is given."""
    valid = depth > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1)
    if c2w is not None:
        pts = pts @ c2w[:3, :3].T + c2w[:3, 3]
    colors = rgb[v, u] if rgb is not None else None
    return pts, colors


def to_torch(walk, device="cpu"):
    """Convert a load_walk() result to torch tensors: rgb -> (N,3,H,W) float in [0,1], depth -> (N,1,H,W)."""
    import torch
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)
    return {
        "rgb": t(walk["rgb"], torch.float32).permute(0, 3, 1, 2) / 255.0,
        "depth": t(walk["depth"], torch.float32)[:, None],
        "semantic": t(walk["semantic"], torch.long),
        "instance": t(walk["instance"], torch.long),
        "c2w": t(walk["c2w"], torch.float32),
        "K": t(walk["K"], torch.float32),
    }
