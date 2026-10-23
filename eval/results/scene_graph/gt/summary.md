# Scene graph vs ground truth: `gt` inputs

Every 3th frame (10 Hz) · object match = same class and 3D IoU ≥ 0.25

**Recall 100.0 % · precision 99.4 % · 1 duplicates · mean 3D IoU 0.98 · centroid error 0.3 cm · size error 0.3 cm**

**Update time:** 4.66 ms/frame median, 7.93 ms p95 (lift 2.51 · associate 0.38 · update 1.80; CPU, numpy)

| Walk | Lighting | GT objects | In graph | Matched | Duplicates | 3D IoU | Centroid err (cm) | Missed |
|---|---|---|---|---|---|---|---|---|
| walk_01 | day | 15 | 15 | 15 | 0 | 0.98 | 0.4 | - |
| walk_02 | overcast | 15 | 15 | 15 | 0 | 0.99 | 0.3 | - |
| walk_03 | evening | 18 | 18 | 18 | 0 | 0.98 | 0.3 | - |
| walk_04 | night | 13 | 13 | 13 | 0 | 0.99 | 0.3 | - |
| walk_05 | day | 16 | 16 | 16 | 0 | 0.99 | 0.2 | - |
| walk_06 | overcast | 16 | 16 | 16 | 0 | 0.98 | 0.4 | - |
| walk_07 | evening | 16 | 16 | 16 | 0 | 0.98 | 0.4 | - |
| walk_08 | night | 17 | 18 | 17 | 1 | 0.98 | 0.4 | - |
| walk_09 | day | 17 | 17 | 17 | 0 | 0.98 | 0.5 | - |
| walk_10 | overcast | 15 | 15 | 15 | 0 | 0.98 | 0.3 | - |
