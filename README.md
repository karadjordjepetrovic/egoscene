# EgoScene

Ego-centric synthetic data and scene understanding for XR glasses (work in progress).

A game engine as a ground-truth generator: UE5 renders first-person walks with perfect depth, segmentation and camera poses, which are then used to benchmark vision foundation models and build lightweight 3D scene representations.

## Key findings so far

Synthetic first-person walks with perfect ground truth, used to score off-the-shelf vision models the way XR glasses would use them:

1. **Metric depth is ~33 % off on a wide-angle glasses camera, and the error follows the lens.** DA V2 Metric Indoor behaves as if every image had a ~74° lens (scale error = `tan(37°)/tan(FOV/2)`, confirmed at 60°, 74° and 90°). Correcting from the camera intrinsics alone takes δ1 from 21 % to 86 %. On a phone (~72° lens) the same model needs almost no correction.
2. **Under 1 m is the blind spot across tasks.** Depth error, detection recall (53 % vs 84 % at 2–4 m) and text reading all get worse up close, exactly where you interact with things.
3. **Small things are invisible at 720p.** Detection recall is 10 % for objects under 32×32 px; signs under 20 px tall are read correctly 2.5 % of the time.
4. **Open-vocabulary detection beats closed-set segmentation for the objects XR triggers need.** Sofa: 93.7 AP with OWLv2 vs 12 IoU with SegFormer (it calls it "armchair").
5. **Lighting hurts reading most.** Day → night barely moves depth or detection, but exact text reads drop from 48 % to 28 %.

## Status

- [x] **Phase 1:** UE5 capture rig: glasses camera, natural head motion, GT passes, coordinate conversion. Re-projection check: 0.02 % median depth error.
- [x] **Phase 2:** dataset v1: 10 walks, 3,644 frames, 17 classes, 4 lighting presets. See the [dataset card](dataset/DATASET_CARD.md).
- [x] **Phase 3:** benchmark depth, segmentation, detection and text models against ground truth, plus a real-video check.
- [ ] **Phase 4:** object-centric 3D scene graph in a real-time budget.
- [ ] **Phase 5:** Gemma interface, contextual triggers and XR panel placement.

## Results: depth (Phase 3)

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

## Results: segmentation, detection, text (Phase 3)

Same 370 frames (every 10th; every 5th for text), RTX 3070, FP16, unoptimized Hugging Face forward passes. Full tables in [`eval/results/`](eval/results/).

| Task | Model | Score | ms/frame |
|---|---|---|---|
| Semantic segmentation | SegFormer-B2 (ADE20K, mapped to EgoScene classes) | 44.0 mIoU | 31 |
| Semantic segmentation | CLIPSeg (open-vocabulary prompts) | 41.5 mIoU | 80 |
| Object detection | OWLv2-base (open-vocabulary prompts) | **79.2 mAP@0.5** | 51 |
| Scene text | EasyOCR | 87 % found, **38 % read exactly** | 110 |

**Segmentation.** Large surfaces are solved (wall 93, ceiling 97, floor 88 IoU). Objects are not: SegFormer calls the sofa "armchair" (sofa 12 IoU, but 80 IoU if sofa and chair are merged into "seating"); stove 72, table 62, TV 29. Counter and sink score near zero, but they're textured blockouts in v1 and the models fairly call them "cabinet" and "step". The view through the windows is often called "painting", a synthetic artifact (a flat HDRI behind glass) that doesn't happen on the real video below. Evening is the worst split (36 vs 44–47 mIoU), partly because the evening interiors are overexposed.

**Detection.** With text prompts, OWLv2 finds chairs (96 AP), plants (96), sofas (94), shelves (88), tables and TVs (83–84). What limits it is size and range, not lighting (77–85 mAP across lighting):

| Box size | Recall | | Object distance | Recall |
|---|---|---|---|---|
| < 32² px | 10 % | | 0–1 m | 53 % |
| 32²–96² px | 56 % | | 1–2 m | 77 % |
| > 96² px | 93 % | | 2–4 m | 84 % |

**Scene text.** EasyOCR finds 87 % of the sign views but reads only 38 % exactly (character error rate 0.31). Signs under 20 px tall are almost never read (2.5 %); at night exact reads drop to 28 % (day 48 %). Up close is no better (42 %), because signs are cut off by the frame edge or seen at a steep angle.

![Sign legibility from near to far](eval/results/text/easyocr/figures/legibility.png)

## Sim-to-real check

The same models on a 73 s first-person phone video of a real home (Samsung S25 main camera, portrait; one frame every 2 s). No ground truth, so this is qualitative, but it checks that the synthetic findings aren't artifacts:

- **Depth scale transfers as predicted.** The phone's lens (~72° along the long side) is close to the ~74° the metric model assumes, so the FOV correction is only ×1.04 and raw metric depth looks right (room depths 1.4–6.4 m, median 2.8 m). On the 90° glasses camera the same model needs ×0.75.
- **Real windows are recognised as windows/outdoor**, unlike the synthetic HDRI view, confirming that failure is a dataset artifact.
- **OWLv2 transfers well** (chairs, armchair, dressers as shelves). **EasyOCR hallucinates** text on a patterned rug: on real footage false positives matter too.
- Same speed as on synthetic frames: depth 19 ms, SegFormer 26 ms, OWLv2 54 ms, EasyOCR 96 ms at 720×1280.

| | |
|---|---|
| ![](eval/results/real/frame_018_t036s.png) | ![](eval/results/real/frame_036_t072s.png) |
| ![](eval/results/real/frame_013_t026s.png) | ![](eval/results/real/frame_015_t030s.png) |

## Repository

| Path | What it is |
|---|---|
| `ue5/` | Unreal Editor Python scripts: labelling, walk generation |
| `dataset/` | Converter, loader (`egoscene.py`), re-projection check, demo notebook, dataset card |
| `dataset/labels/` | Class and instance label maps |
| `eval/` | Benchmarks: `depth_benchmark.py`, `seg_benchmark.py`, `det_benchmark.py`, `text_benchmark.py`, `real_video.py`; results in `eval/results/` |
| `scene_graph/`, `placement/`, `report/` | Later phases |

## Quick start

```bash
pip install numpy opencv-python OpenEXR matplotlib jupyter
python dataset/convert_all.py                       # UE renders -> data/
python dataset/reproject_check.py data/apartment_01/walk_04 100 105
jupyter notebook dataset/demo.ipynb
```
