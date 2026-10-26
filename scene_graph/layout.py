"""Room layout for the scene graph: floor plane, main walls, and support relations between objects.

    floor = estimate_floor(floor_points)            # {"height", "normal", "tilt_deg", "inliers"}
    walls = estimate_walls(wall_points, floor)       # [{"p0", "p1", "normal", "points"}] on the top-down map
    rel   = support_relations(objects, floor)        # {obj id: "floor" | "on <class>#<id>" | "wall" | "free"}

All points in world coordinates (metres). On EgoScene the world is already Z-up; on real video the
floor fit is what defines "up" (see align_to_floor).
"""
import numpy as np


def fit_plane_ransac(P, iters=300, thr=0.02, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    Q = P[best]                                     # least-squares refit on inliers
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    return n, c, best


def estimate_floor(points, up_hint=(0, 0, 1)):
    """Fit the floor plane. Returns height (z of the plane under the origin), unit normal, tilt."""
    n, c, inl = fit_plane_ransac(points)
    if n @ np.asarray(up_hint) < 0:
        n = -n
    height = float(c @ n / n[2]) if abs(n[2]) > 1e-6 else float(c[2])   # plane z at x=y=0
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    return {"height": height, "normal": n.tolist(), "tilt_deg": tilt, "inliers": int(inl.sum())}


def align_to_floor(floor):
    """4x4 transform that makes the fitted floor the plane z = 0 with +Z up (for real video, unknown gravity)."""
    n = np.asarray(floor["normal"], float)
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z); s = np.linalg.norm(v); c = float(n @ z)
    if s < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    T = np.eye(4); T[:3, :3] = R
    T[2, 3] = -float((R @ (np.array([0, 0, floor["height"]])))[2])
    return T


def estimate_walls(points, floor, n_max=12, thr=0.04, min_pts=150, band=(0.3, 2.0), grid=0.02, max_gap=0.4, rng=None):
    """Main wall lines on the top-down map: iterative 2D RANSAC lines on wall points between `band`
    metres above the floor. Each wall: endpoints (robust extent of its inliers), normal, support."""
    rng = rng or np.random.default_rng(0)
    h = points[:, 2] - floor["height"]
    P = points[(h > band[0]) & (h < band[1])][:, :2]
    if len(P) == 0:
        return []
    P = np.unique(np.round(P / grid), axis=0) * grid      # occupancy grid: one point per 2 cm cell
    walls = []
    for _ in range(n_max):
        if len(P) < min_pts:
            break
        best, best_n = None, -1
        for _ in range(300):
            a, b = P[rng.choice(len(P), 2, replace=False)]
            d = b - a
            if np.linalg.norm(d) < 0.3:
                continue
            nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            inl = np.abs((P - a) @ nrm) < thr
            if inl.sum() > best_n:
                best, best_n, best_nrm, best_a = inl, inl.sum(), nrm, a
        if best_n < min_pts:
            break
        Q = P[best]
        c = Q.mean(0); dirv = np.linalg.svd(Q - c, full_matrices=False)[2][0]
        t = np.sort((Q - c) @ dirv)
        # split the line where its points have gaps (doorways, openings), keep pieces >= 0.3 m
        cuts = np.nonzero(np.diff(t) > max_gap)[0]
        for seg in np.split(t, cuts + 1):
            if len(seg) < 20 or seg[-1] - seg[0] < 0.3:
                continue
            lo, hi = np.percentile(seg, 1), np.percentile(seg, 99)
            walls.append({"p0": (c + lo * dirv).tolist(), "p1": (c + hi * dirv).tolist(),
                          "normal": [float(-dirv[1]), float(dirv[0])], "points": int(len(seg)),
                          "length": float(hi - lo)})
        P = P[~best]
    return walls


def support_relations(objects, floor, tol=0.08, floor_tol=0.2):
    """For each scene-graph object: what holds it up. 'floor', 'on <class>#<id>' (bottom rests on
    another object's top with overlapping footprint), 'wall' (high up and thin/vertical) or 'free'."""
    rel = {}
    for o in objects:
        bottom = float(o.box_min[2]) - floor["height"]
        if bottom < floor_tol:        # boxes are robust percentiles, so thin legs / toe-kicks lift the bottom a bit
            rel[o.id] = "floor"; continue
        best = None
        for s in objects:
            if s is o or s.box_max[2] > o.box_max[2]:
                continue
            gap = abs(float(s.box_max[2]) - float(o.box_min[2]))
            ov = np.prod(np.clip(np.minimum(o.box_max[:2], s.box_max[:2]) - np.maximum(o.box_min[:2], s.box_min[:2]), 0, None))
            if gap < tol and ov > 0.3 * np.prod(np.clip(o.box_max[:2] - o.box_min[:2], 1e-6, None)):
                if best is None or gap < best[0]:
                    best = (gap, s)
        if best:
            rel[o.id] = f"on {best[1].cls}#{best[1].id}"
        elif o.surface == "vertical" or bottom > 0.8:
            rel[o.id] = "wall"
        else:
            rel[o.id] = "free"
    return rel


def plane_to_z0(n, point):
    """4x4 transform rotating unit normal n to +Z and moving the plane through `point` to z = 0."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z); s = np.linalg.norm(v); c = float(n @ z)
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    T = np.eye(4); T[:3, :3] = R
    T[2, 3] = -float((R @ np.asarray(point, float))[2])
    return T


def yaw_to_wall(walls):
    """4x4 rotation about Z that makes the longest wall run along the X axis (tidier axis-aligned boxes)."""
    if not walls:
        return np.eye(4)
    w = max(walls, key=lambda w: w["length"])
    d = np.array(w["p1"]) - np.array(w["p0"]); a = -np.arctan2(d[1], d[0])
    T = np.eye(4); T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T
