# EgoScene

Ego-centric synthetic data and scene understanding for XR glasses (work in progress).

A game engine as a ground-truth generator: UE5 renders first-person walks with perfect depth, segmentation and camera poses, which are then used to benchmark vision foundation models and build lightweight 3D scene representations.

## Status

- [x] **Phase 1:** UE5 capture rig: glasses camera, natural head motion, GT passes, coordinate conversion. Re-projection check: 0.02 % median depth error.
- [x] **Phase 2:** dataset v1: 10 walks, 3,644 frames, 17 classes, 4 lighting presets. See the [dataset card](dataset/DATASET_CARD.md).
- [ ] **Phase 3:** benchmark depth, segmentation, detection and text models against ground truth. Depth done (see below).
- [ ] **Phase 4:** object-centric 3D scene graph in a real-time budget.
- [ ] **Phase 5:** Gemma interface, contextual triggers and XR panel placement.

## Results: depth (Phase 3, in progress)

Every 10th frame of all 10 walks (370 frames), ground truth ≤ 10 m, RTX 3070, FP16. Full tables: [`eval/results/depth/`](eval/results/depth/).

| Model | Protocol | AbsRel ↓ | δ1 (%) ↑ | 0–1 m AbsRel ↓ | Flicker (%) ↓ | ms/frame* |
|---|---|---|---|---|---|---|
| Depth Anything V2 Small | relative, per-frame scale+shift | **0.060** | **95.7** | 0.070 | 0.69 | 32 |
| DA V2 Metric Indoor Small | raw metric | 0.428 | 21.4 | 1.164 | 1.78 | 28 |
| DA V2 Metric Indoor Small | **FOV-corrected** (intrinsics only, no GT) | 0.139 | 86.1 | 0.654 | 1.34 | 28 |
| DA V2 Metric Indoor Small | first-frame calibration | 0.203 | 61.6 | 0.401 | 1.15 | 28 |
| DA V2 Metric Indoor Small | oracle per-frame scale (upper bound) | 0.083 | 92.2 | 0.353 | 0.76 | 28 |

\* Unoptimized Hugging Face forward pass at 518×924, batch 1. My [optimized pipeline](https://github.com/karadjordjepetrovic/realtime-depth-optimization) runs DA V2 Small in 6.86 ms at 518×518; not yet re-timed at this resolution.

### Findings

**1. Metric depth fails on wide-angle glasses cameras, by a predictable amount.** On the 90° glasses frames, DA V2 Metric Indoor predicts everything about 33 % too far (median GT/pred scale 0.75; δ1 21 %). Centre-cropping the same frames to narrower fields of view shows why: the scale error follows `tan(37°) / tan(FOV/2)` almost exactly, i.e. the model behaves as if every image had a ~74° lens. Correcting for it with the camera intrinsics alone (×0.75 at 90°) lifts δ1 from 21 % to 86 % with no ground truth.

![Scale error vs FOV](eval/results/depth/fov_scale.png)

*Median GT/pred scale (bars: interquartile range across 370 frames) at 90°, 74° and 60° horizontal FOV, against the curve `tan(37°)/tan(FOV/2)`.*

**2. Calibrating on one frame is fragile; calibrating from intrinsics is not.** Estimating the scale on each walk's first frame and keeping it fixed scores worse (δ1 62 %) than the intrinsics-based correction (86 %). Three walks start next to a wall, where the first-frame scale is 0.45 instead of ~0.75, and that one bad frame skews the whole walk.

**3. Near range (0–1 m) is a separate failure.** Even with the right FOV the metric model's error under 1 m stays high (AbsRel 0.65, and 0.35 with a perfect per-frame scale), while beyond 2 m it reaches δ1 98–99.7 %. The relative model doesn't have this problem (0.070 under 1 m), so it's the metric head, not the backbone. For glasses this is the range that matters most: counters, nearby walls, hands, panels.

**4. Objects are harder than surfaces.** For DA V2 Small, sofa, table and chair have about twice the error of walls and floor (AbsRel 0.10–0.13 vs 0.06; δ1 ~81 % for sofa and table). These are exactly the surfaces XR content gets anchored to.

**5. Blank walls far away get read as "sky".** The relative model pushes 0.9 % of valid pixels to the far limit (0.4 % in daylight, 1.4 % at night), mostly plain walls seen through doorways; that's why its error beyond 4 m jumps to 0.15.

**6. Lighting matters less than expected.** Day to night changes AbsRel by 0.049 → 0.064 (relative) and 0.114 → 0.147 (metric, FOV-corrected).

| Far wall read as "sky" (DA V2 Small, night) | Near wall pushed back (DA V2 Metric, raw, evening) |
|---|---|
| ![](eval/results/depth/da2-small/figures/worst_walk_08_000050.png) | ![](eval/results/depth/da2-metric-indoor-small/figures/worst_walk_03_000050.png) |

## Repository

| Path | What it is |
|---|---|
| `ue5/` | Unreal Editor Python scripts: labelling, walk generation |
| `dataset/` | Converter, loader (`egoscene.py`), re-projection check, demo notebook, dataset card |
| `dataset/labels/` | Class and instance label maps |
| `eval/` | Model benchmarks (Phase 3) |
| `scene_graph/`, `placement/`, `report/` | Later phases |

## Quick start

```bash
pip install numpy opencv-python OpenEXR matplotlib jupyter
python dataset/convert_all.py                       # UE renders -> data/
python dataset/reproject_check.py data/apartment_01/walk_04 100 105
jupyter notebook dataset/demo.ipynb
```
