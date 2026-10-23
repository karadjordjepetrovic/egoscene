# Scene graph vs ground truth: `gtmask_modeldepth`

Every 3th frame (10 Hz) · match = same class and 3D IoU ≥ 0.25 · GT poses

**Recall 61.4 % · precision 51.1 % · 12 duplicates · 81 false positives · 3D IoU 0.41 · centroid error 15.2 cm · size error 17.4 cm**

**Graph update:** 4.96 ms/frame median, 8.33 ms p95 (CPU, numpy) · models: depth 19 ms, OWLv2 59 ms

| Class | Recall (%) |
|---|---|
| sign | 0 |
| lamp | 11 |
| chair | 60 |
| plant | 64 |
| sink | 67 |
| stove | 75 |
| table | 80 |
| shelf | 85 |
| tv | 90 |
| counter | 100 |
| sofa | 100 |

| Walk | Lighting | GT objects | In graph | Matched | Dup. | False pos. | Missed |
|---|---|---|---|---|---|---|---|
| walk_01 | day | 15 | 17 | 11 | 1 | 5 | SM_Lamp_01, SM_CoffeeTable_01, Sign_04, Sign_02 |
| walk_02 | overcast | 15 | 16 | 8 | 1 | 7 | SM_Lamp_01, Shelf_02, SM_Chair_02, Sign_02, Plant_03, Sign_03, Sign_04 |
| walk_03 | evening | 18 | 22 | 11 | 0 | 11 | Plant_03, Sign_04, SM_Chair_02, Sign_02, Sign_03, SM_TV_01, SM_Lamp_01 |
| walk_04 | night | 13 | 16 | 7 | 1 | 8 | SM_CoffeeTable_01, SM_Chair_02, Sign_02, SM_Sink_01, SM_Plant_02, SM_Shelf_01 |
| walk_05 | day | 16 | 18 | 9 | 0 | 9 | SM_Lamp_01, SM_Chair_02, Sign_02, Plant_03, Sign_03, SM_Stove_01, Sign_04 |
| walk_06 | overcast | 16 | 20 | 10 | 2 | 8 | SM_CoffeeTable_01, SM_Chair_02, Plant_03, Sign_02, SM_Lamp_01, SM_Sink_01 |
| walk_07 | evening | 16 | 21 | 8 | 3 | 10 | SM_Lamp_01, SM_Plant_02, SM_Chair_02, Plant_03, Sign_02, Sign_03, SM_Stove_01, Sign_04 |
| walk_08 | night | 17 | 21 | 12 | 1 | 8 | SM_CoffeeTable_01, Plant_03, Sign_04, Sign_02, SM_Chair_02 |
| walk_09 | day | 17 | 20 | 13 | 2 | 5 | Plant_03, Sign_02, Sign_03, SM_Lamp_01 |
| walk_10 | overcast | 15 | 19 | 8 | 1 | 10 | Plant_03, Sign_04, Sign_02, SM_Chair_02, Shelf_02, Sign_03, SM_Lamp_01 |
