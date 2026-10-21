import json, sys, cv2, numpy as np

walk, i, j = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
meta = json.load(open(f"{walk}/meta.json"))
K = meta["intrinsics"]; fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
load_d = lambda k: cv2.imread(f"{walk}/{meta['frames'][k]['depth']}", cv2.IMREAD_UNCHANGED) / 1000.0
c2w = lambda k: np.array(meta["frames"][k]["c2w"])

d_i, d_j = load_d(i), load_d(j)
v, u = np.nonzero(d_i > 0)
z = d_i[v, u]
pts_cam_i = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z, np.ones_like(z)])
pts_cam_j = np.linalg.inv(c2w(j)) @ c2w(i) @ pts_cam_i
zj = pts_cam_j[2]
uj = np.round(pts_cam_j[0] / zj * fx + cx).astype(int)
vj = np.round(pts_cam_j[1] / zj * fy + cy).astype(int)
ok = (zj > 0.1) & (uj >= 0) & (uj < meta["width"]) & (vj >= 0) & (vj < meta["height"])
pred, seen = zj[ok], d_j[vj[ok], uj[ok]]
valid = seen > 0
rel = np.abs(pred[valid] - seen[valid]) / seen[valid]
print(f"median depth error {np.median(rel)*100:.2f}%, within 2%: {(rel < 0.02).mean()*100:.1f}% of pixels")