"""EgoScene object-centric 3D scene graph.

A small, persistent list of objects the glasses "know about", updated every frame:

    sg = SceneGraph()
    for frame in walk:
        dets = [Detection(cls="chair", mask=..., score=0.9), ...]   # 2D masks from any model (or GT)
        sg.update(dets, depth_m, K, c2w, t_sec)
    sg.objects   # -> list of SceneObject (id, class, 3D box in world, surface type, last seen, ...)

World frame: right-handed, Z up, metres (EgoScene convention). Boxes are axis-aligned in world,
from robust percentiles of the fused points (indoor furniture is mostly aligned with the room).
"""
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class Detection:
    cls: str
    mask: np.ndarray           # (H, W) bool
    score: float = 1.0
    track_id: int | None = None    # optional 2D tracker id, helps association


@dataclass
class SceneObject:
    id: int
    cls: str
    points: np.ndarray         # (N, 3) reservoir sample of fused world points
    first_seen: float
    last_seen: float
    n_obs: int = 1
    score: float = 1.0
    box_min: np.ndarray = field(default=None)
    box_max: np.ndarray = field(default=None)
    n_seen: int = 0            # total points ever merged (for unbiased reservoir sampling)

    @property
    def centroid(self):
        return (self.box_min + self.box_max) / 2

    @property
    def size(self):
        return self.box_max - self.box_min

    @property
    def confidence(self):
        return float(1 - np.exp(-self.n_obs / 3) ) * self.score

    @property
    def surface(self):
        """'vertical' for thin upright things (TV, sign, wall-mounted), else 'horizontal' (has a usable top)."""
        dx, dy, dz = self.size
        return "vertical" if min(dx, dy) < 0.12 and dz > 0.15 else "horizontal"

    def to_dict(self):
        return {"id": self.id, "class": self.cls, "centroid": np.round(self.centroid, 3).tolist(),
                "size": np.round(self.size, 3).tolist(), "top_z": round(float(self.box_max[2]), 3),
                "surface": self.surface, "n_obs": self.n_obs, "confidence": round(self.confidence, 3),
                "last_seen": round(self.last_seen, 2)}


def aabb(points, lo=2, hi=98):
    return np.percentile(points, lo, axis=0), np.percentile(points, hi, axis=0)


def containment(amin, amax, bmin, bmax):
    """Fraction of the smaller box's volume inside the other (handles partial views of big objects)."""
    inter = np.prod(np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0, None))
    va, vb = np.prod(np.clip(amax - amin, 1e-6, None)), np.prod(np.clip(bmax - bmin, 1e-6, None))
    return float(inter / min(va, vb))


def box_iou_3d(amin, amax, bmin, bmax):
    inter = np.prod(np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0, None))
    va, vb = np.prod(np.clip(amax - amin, 1e-6, None)), np.prod(np.clip(bmax - bmin, 1e-6, None))
    return float(inter / (va + vb - inter))


class SceneGraph:
    def __init__(self, max_points=3000, pixel_stride=4, min_points=40, match_iou=0.1, match_dist=0.6,
                 prune_after_s=3.0, min_obs_keep=2, rng_seed=0):
        self.objects: list[SceneObject] = []
        self.max_points, self.pixel_stride, self.min_points = max_points, pixel_stride, min_points
        self.match_iou, self.match_dist = match_iou, match_dist
        self.prune_after_s, self.min_obs_keep = prune_after_s, min_obs_keep
        self._next_id = 1
        self._rng = np.random.default_rng(rng_seed)
        self.timings = {"lift": [], "associate": [], "update": []}

    # ---------------------------------------------------------------- lifting
    def lift(self, mask, depth, K, c2w):
        """Masked depth pixels -> world points, with depth outliers (mask bleeding) removed."""
        s = self.pixel_stride
        m = mask[::s, ::s] & (depth[::s, ::s] > 0)
        v, u = np.nonzero(m)
        if len(v) < self.min_points:
            return None
        z = depth[::s, ::s][v, u]
        med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-3
        keep = np.abs(z - med) < 4 * 1.4826 * mad + 0.05
        v, u, z = v[keep] * s, u[keep] * s, z[keep]
        if len(z) < self.min_points:
            return None
        pts = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        return pts @ c2w[:3, :3].T + c2w[:3, 3]

    # ---------------------------------------------------------------- one frame
    def update(self, detections, depth, K, c2w, t):
        t0 = time.perf_counter()
        lifted = []
        for d in detections:
            p = self.lift(d.mask, depth, K, c2w)
            if p is not None:
                bmin, bmax = aabb(p)
                lifted.append((d, p, bmin, bmax))
        t1 = time.perf_counter()

        # greedy association: same class, best 3D IoU (or centroid distance as fallback)
        pairs = []
        for di, (d, p, bmin, bmax) in enumerate(lifted):
            c = (bmin + bmax) / 2
            for oi, o in enumerate(self.objects):
                if o.cls != d.cls:
                    continue
                iou = box_iou_3d(bmin, bmax, o.box_min, o.box_max)
                dist = float(np.linalg.norm(c - o.centroid))
                cont = containment(bmin, bmax, o.box_min, o.box_max)
                if iou >= self.match_iou or dist <= self.match_dist or cont >= 0.5:
                    pairs.append((iou - 0.1 * dist, di, oi))
        pairs.sort(reverse=True)
        used_d, used_o, matches = set(), set(), []
        for _, di, oi in pairs:
            if di not in used_d and oi not in used_o:
                matches.append((di, oi)); used_d.add(di); used_o.add(oi)
        t2 = time.perf_counter()

        for di, oi in matches:
            d, p, _, _ = lifted[di]
            self._merge(self.objects[oi], p, d.score, t)
        for di, (d, p, bmin, bmax) in enumerate(lifted):
            if di not in used_d:
                self._add(d, p, t)
        self._prune(t)
        t3 = time.perf_counter()
        for k, a, b in (("lift", t0, t1), ("associate", t1, t2), ("update", t2, t3)):
            self.timings[k].append((b - a) * 1000)

    def _sample(self, pts):
        if len(pts) > self.max_points:
            pts = pts[self._rng.choice(len(pts), self.max_points, replace=False)]
        return pts

    def _add(self, d, p, t):
        n = len(p)
        p = self._sample(p)
        bmin, bmax = aabb(p)
        self.objects.append(SceneObject(self._next_id, d.cls, p, t, t, 1, d.score, bmin, bmax, n_seen=n))
        self._next_id += 1

    def _merge(self, o, p, score, t):
        # Unbiased reservoir: every point ever seen has the same chance to be kept, so the box
        # reflects all views of the object, not just the most recent ones.
        total = o.n_seen + len(p)
        if len(o.points) + len(p) <= self.max_points:
            o.points = np.concatenate([o.points, p])
        else:
            k_new = int(round(self.max_points * len(p) / total))
            k_old = self.max_points - k_new
            keep_old = o.points[self._rng.choice(len(o.points), min(k_old, len(o.points)), replace=False)]
            keep_new = p[self._rng.choice(len(p), min(k_new, len(p)), replace=False)]
            o.points = np.concatenate([keep_old, keep_new])
        o.n_seen = total
        o.box_min, o.box_max = aabb(o.points)
        o.n_obs += 1
        o.score = max(o.score, score)
        o.last_seen = t

    def _prune(self, t):
        """Merge same-class objects that turned out to be parts of one thing (e.g. a counter's
        worktop seen first, its cabinet later), then drop rarely-seen objects that went stale."""
        merged = True
        while merged:
            merged = False
            for i, a in enumerate(self.objects):
                for b in self.objects[i + 1:]:
                    if a.cls == b.cls and containment(a.box_min, a.box_max, b.box_min, b.box_max) >= 0.6:
                        keep, drop = (a, b) if a.n_obs >= b.n_obs else (b, a)
                        self._merge(keep, drop.points, drop.score, max(a.last_seen, b.last_seen))
                        keep.n_obs += drop.n_obs - 1
                        keep.first_seen = min(a.first_seen, b.first_seen)
                        self.objects.remove(drop)
                        merged = True
                        break
                if merged:
                    break
        self.objects = [o for o in self.objects
                        if o.n_obs >= self.min_obs_keep or t - o.last_seen <= self.prune_after_s]

    def snapshot(self, min_obs=1):
        return [o.to_dict() for o in self.objects if o.n_obs >= min_obs]

def mask_from_box(box, depth, inner=0.5, front_pct=10, near=0.3, far_min=0.3, far_rel=0.15):
    """Turn a 2D detection box into an object mask using depth: keep the box pixels whose depth is
    close to the object's front surface (estimated from the box centre), dropping the wall behind it.
    box: [x0, y0, x1, y1] in pixels; depth: (H, W) metres."""
    H, W = depth.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    m = np.zeros((H, W), bool)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return m
    cx0 = int(x0 + (x1 - x0) * (1 - inner) / 2); cx1 = int(x1 - (x1 - x0) * (1 - inner) / 2)
    cy0 = int(y0 + (y1 - y0) * (1 - inner) / 2); cy1 = int(y1 - (y1 - y0) * (1 - inner) / 2)
    c = depth[cy0:cy1 + 1, cx0:cx1 + 1]
    c = c[c > 0]
    if len(c) == 0:
        return m
    z_ref = np.percentile(c, front_pct)
    d = depth[y0:y1, x0:x1]
    m[y0:y1, x0:x1] = (d > 0) & (d > z_ref - near) & (d < z_ref + max(far_min, far_rel * z_ref))
    return m


class ScaleAligner:
    """Temporal scale alignment for monocular metric depth (no ground truth needed, only camera poses).

    Each new depth map is rescaled so that it agrees with the previous (aligned) depth map where the
    two views overlap, then pulled gently back towards scale 1 (the FOV-corrected prior) so chained
    estimates don't drift:   log s_t = (1 - prior_weight) * log s_chain + prior_weight * 0
    """

    def __init__(self, prior_weight=0.1, stride=8, max_rel=0.25):
        self.prior_weight, self.stride, self.max_rel = prior_weight, stride, max_rel
        self.prev = None           # (aligned depth, c2w)
        self.scales = []

    def __call__(self, depth, K, c2w):
        s = 1.0
        if self.prev is not None:
            pd, pc2w = self.prev
            st = self.stride
            v, u = np.mgrid[0:pd.shape[0]:st, 0:pd.shape[1]:st]
            v, u = v.ravel(), u.ravel()
            z = pd[v, u]
            ok = z > 0
            v, u, z = v[ok], u[ok], z[ok]
            P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)])
            Q = np.linalg.inv(c2w) @ pc2w @ P
            zj = Q[2]
            front = zj > 0.1
            uj = np.round(Q[0][front] / zj[front] * K[0, 0] + K[0, 2]).astype(int)
            vj = np.round(Q[1][front] / zj[front] * K[1, 1] + K[1, 2]).astype(int)
            zj = zj[front]
            H, W = depth.shape
            inb = (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
            cur = depth[vj[inb], uj[inb]]
            zj = zj[inb]
            good = cur > 0
            if good.sum() > 200:
                r = zj[good] / cur[good]
                med = np.median(r)
                r = r[np.abs(r / med - 1) < self.max_rel]          # drop occlusions / disocclusions
                if len(r) > 100:
                    s_chain = float(np.median(r))
                    s = float(np.exp((1 - self.prior_weight) * np.log(s_chain)))
        aligned = depth * s
        self.prev = (aligned, c2w)
        self.scales.append(s)
        return aligned
