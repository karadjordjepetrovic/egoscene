# EgoScene v1 – dataset card

Synthetic **first-person (ego-centric) walks** through an indoor apartment, rendered in Unreal Engine 5 to look like what a pair of XR glasses would see, with **pixel-perfect ground truth** for every frame. Built as a benchmark for evaluating vision foundation models (depth, segmentation, detection, text) on ego-centric footage, and as input for lightweight 3D scene representations.

## At a glance

| | |
|---|---|
| Walks | 10 (one environment: `Apartment_01`, living room + kitchen + hallway) |
| Frames | 3,644 at 30 fps (≈ 2 minutes of walking) |
| Resolution | 1280 × 720, 90° horizontal FOV, pinhole (fx = fy = 640, cx = 640, cy = 360) |
| Per frame | RGB, metric depth, semantic segmentation, instance segmentation, camera pose |
| Classes | 17 (+ unlabelled), 50 object instances |
| Variation | 4 lighting presets × 3 walking speeds, 10 different routes |
| Ground-truth check | multi-view re-projection: **0.02–0.03 % median depth error, 98.4 % of pixels within 2 %** |

| Walk | Lighting | Speed | Frames |
|---|---|---|---|
| walk_01 | day | 110 cm/s | 296 |
| walk_02 | overcast | 80 cm/s | 392 |
| walk_03 | evening | 140 cm/s | 315 |
| walk_04 | night | 80 cm/s | 402 |
| walk_05 | day | 140 cm/s | 277 |
| walk_06 | overcast | 110 cm/s | 358 |
| walk_07 | evening | 80 cm/s | 393 |
| walk_08 | night | 140 cm/s | 377 |
| walk_09 | day | 110 cm/s | 382 |
| walk_10 | overcast | 80 cm/s | 452 |

## Format

```
data/
  manifest.json                  # one entry per walk: walk_id, level, lighting, speed_cm_s, seed, num_frames, path
  apartment_01/walk_XX/
    rgb/000000.png               # 8-bit RGB
    depth/000000.png             # uint16, millimetres, 0 = invalid
    semantic/000000.png          # uint8 class id (index into meta.json "classes", 0 = unlabelled)
    instance/000000.png          # uint8 instance id (key into meta.json "instances", 0 = none)
    meta.json                    # intrinsics, classes, instances, lighting/speed/seed, per-frame c2w pose
```

**Conventions.** World: right-handed, Z up, metres (floor at Z = 0, ceiling at 2.7 m). Camera: OpenCV (x right, y down, z forward). `c2w` is the 4×4 camera-to-world matrix. Depth is planar z-depth (distance along the optical axis), not ray length.

**Loading.**

```python
from dataset.egoscene import list_walks, load_walk, backproject
walk = load_walk("data/apartment_01/walk_04", stride=10)   # aligned numpy arrays
pts, cols = backproject(walk["depth"][0], walk["K"], walk["c2w"][0], rgb=walk["rgb"][0])
```

`dataset/demo.ipynb` shows one frame's ground truth side by side, a back-projected point cloud and a whole walk fused into a top-down floor plan.

## Classes

Share of pixels and of frames in which the class appears (sampled every 40th frame across all walks):

| Class | Pixels | Frames | | Class | Pixels | Frames |
|---|---|---|---|---|---|---|
| wall | 68.1 % | 100 % | | chair | 0.74 % | 43 % |
| floor | 9.5 % | 91 % | | stove | 0.75 % | 25 % |
| ceiling | 6.8 % | 86 % | | tv | 0.49 % | 34 % |
| shelf | 4.9 % | 58 % | | sign | 0.15 % | 35 % |
| outdoor | 3.5 % | 56 % | | sink | 0.07 % | 15 % |
| table | 1.6 % | 61 % | | lamp | 0.03 % | 17 % |
| counter | 1.2 % | 20 % | | door | 0 % | 0 % |
| plant | 1.2 % | 59 % | | window | 0 % | 0 % |
| sofa | 1.1 % | 27 % | | | | |

`outdoor` is the view through the windows (an HDRI backdrop). Its depth is set to 0 (invalid), because the backdrop's distance isn't real geometry.

## How it was made

1. **Scene.** UE5.8 level built with Claude Code driving the editor through the Unreal MCP server. Furniture, plants and textures are **CC0 assets from Poly Haven**; counter and sink are textured blockouts. Four wall signs carry readable text ("EXIT", "Kitchen", "Welcome Home", "OCTOBER") for text-recognition tests.
2. **Labels.** Every object gets a Custom Depth Stencil value (1–255) from regex rules on actor and mesh names (`ue5/label_objects.py`), mapped to a class.
3. **Walks.** Waypoint routes interpolated with Catmull-Rom splines, resampled at the walking speed, with natural head motion: yaw glances, pitch, roll and head bob at about 1.6 m eye height (`ue5/make_walk.py`). Each route is line-traced for collisions.
4. **Lighting.** Each preset (day, overcast, evening, night) is baked into its own copy of the level: sun, lamps, sky light, exposure, and a matching CC0 outdoor HDRI (street, overcast street, sunset, night street).
5. **Rendering.** Movie Render Queue, no anti-aliasing, motion blur or depth of field, auto-exposure off; GT passes from post-process materials (scene depth, custom stencil) written as 32-bit EXR.
6. **Conversion.** `dataset/convert_all.py` converts the UE coordinates (left-handed, Z up, cm) to the conventions above and writes the PNG/JSON format.
7. **Verification.** `dataset/reproject_check.py` projects frame *i*'s depth into frame *i+5* using the poses and compares with frame *i+5*'s depth.

## Licence

- Third-party assets: Poly Haven models, textures and HDRIs, all **CC0**.
- Rendered with Unreal Engine 5 under the Epic EULA, which permits distributing rendered output.
- Dataset licence: **CC BY 4.0** (proposed). Code: see repository licence.

## Known gaps and limitations (v1)

- **One environment.** All walks are in the same apartment; results don't yet show generalisation across scenes. Office and street environments are planned.
- **Wall-dominated.** Walls are 68 % of pixels and fairly bare, which makes per-pixel averages optimistic; report per-class metrics.
- **`door` and `window` are never visible as labelled pixels.** The front door isn't in view on the routes, and the translucent window glass doesn't write a stencil value (the view through it is labelled `outdoor`). Treat both classes as absent in v1.
- **Synthetic look.** No sensor noise, motion blur, rolling shutter, lens distortion or auto-exposure. Models trained on real data face a sim-to-real gap; that gap is part of what the benchmark measures, not a bug.
- **Counter and sink are textured blockouts**, not real models: segmentation models fairly call them "cabinet", "step" or "bathtub". Flag them in results.
- **Evening interiors are overexposed** (bright, washed-out walls), so the evening split partly measures exposure, not dusk lighting: segmentation drops ~8 mIoU points in evening vs ~1 at night.
- **No people or dynamic objects.**
- **Ground-truth poses.** Poses are exact; there is no simulated tracking error.
- **IDs are per version.** Class and instance IDs may change in v2; always read them from each walk's `meta.json`.
- **Depth range.** uint16 millimetres caps depth at 65.5 m (indoor scenes stay under 10 m).
