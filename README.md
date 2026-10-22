# EgoScene

Ego-centric synthetic data and scene understanding for XR glasses (work in progress).

A game engine as a ground-truth generator: UE5 renders first-person walks with perfect depth, segmentation and camera poses, which are then used to benchmark vision foundation models and build lightweight 3D scene representations.

## Status

- [x] **Phase 1:** UE5 capture rig: glasses camera, natural head motion, GT passes, coordinate conversion. Re-projection check: 0.02 % median depth error.
- [x] **Phase 2:** dataset v1: 10 walks, 3,644 frames, 17 classes, 4 lighting presets. See the [dataset card](dataset/DATASET_CARD.md).
- [ ] **Phase 3:** benchmark depth, segmentation, detection and text models against ground truth.
- [ ] **Phase 4:** object-centric 3D scene graph in a real-time budget.
- [ ] **Phase 5:** Gemma interface, contextual triggers and XR panel placement.

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
