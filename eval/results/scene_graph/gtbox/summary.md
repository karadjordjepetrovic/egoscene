# Scene graph vs ground truth: `gtbox`

Every 3th frame (10 Hz) · match = same class and 3D IoU ≥ 0.25 · GT poses

**Recall 93.0 % · precision 80.3 % · 17 duplicates · 19 false positives · 3D IoU 0.54 · centroid error 11.3 cm · size error 11.9 cm**

**Graph update:** 5.08 ms/frame median, 8.95 ms p95 (CPU, numpy)

| Class | Recall (%) |
|---|---|
| chair | 60 |
| counter | 67 |
| table | 95 |
| tv | 100 |
| plant | 100 |
| shelf | 100 |
| lamp | 100 |
| sign | 100 |
| stove | 100 |
| sink | 100 |
| sofa | 100 |

| Walk | Lighting | GT objects | In graph | Matched | Dup. | False pos. | Missed |
|---|---|---|---|---|---|---|---|
| walk_01 | day | 15 | 17 | 14 | 2 | 1 | SM_Chair_02 |
| walk_02 | overcast | 15 | 15 | 14 | 0 | 1 | SM_Chair_02 |
| walk_03 | evening | 18 | 21 | 16 | 2 | 3 | SM_Chair_02, SM_Counter_01 |
| walk_04 | night | 13 | 17 | 13 | 2 | 2 | - |
| walk_05 | day | 16 | 16 | 15 | 1 | 0 | SM_Chair_02 |
| walk_06 | overcast | 16 | 20 | 16 | 3 | 1 | - |
| walk_07 | evening | 16 | 18 | 15 | 1 | 2 | SM_Chair_02 |
| walk_08 | night | 17 | 20 | 15 | 2 | 3 | SM_Chair_02, SM_Counter_01 |
| walk_09 | day | 17 | 20 | 15 | 1 | 4 | SM_CoffeeTable_01, SM_Chair_02 |
| walk_10 | overcast | 15 | 19 | 14 | 3 | 2 | SM_Chair_02 |
