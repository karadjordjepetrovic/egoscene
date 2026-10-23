# Scene graph vs ground truth: `gtmask_modeldepth_aligned`

Every 3th frame (10 Hz) · match = same class and 3D IoU ≥ 0.25 · GT poses

**Recall 50.6 % · precision 39.0 % · 15 duplicates · 110 false positives · 3D IoU 0.44 · centroid error 15.7 cm · size error 15.5 cm**

**Graph update:** 5.02 ms/frame median, 8.24 ms p95 (CPU, numpy) · models: depth 19 ms, OWLv2 59 ms

| Class | Recall (%) |
|---|---|
| sign | 0 |
| lamp | 11 |
| chair | 35 |
| stove | 50 |
| sink | 50 |
| table | 60 |
| plant | 64 |
| shelf | 75 |
| tv | 80 |
| counter | 83 |
| sofa | 88 |

| Walk | Lighting | GT objects | In graph | Matched | Dup. | False pos. | Missed |
|---|---|---|---|---|---|---|---|
| walk_01 | day | 15 | 17 | 9 | 0 | 8 | Shelf_02, SM_Chair_02, SM_Lamp_01, SM_CoffeeTable_01, Sign_04, Sign_02 |
| walk_02 | overcast | 15 | 17 | 8 | 3 | 6 | SM_Lamp_01, Shelf_02, SM_Chair_02, Sign_02, Plant_03, Sign_03, Sign_04 |
| walk_03 | evening | 18 | 23 | 5 | 2 | 16 | Plant_03, Sign_04, SM_Table_01, SM_Chair_02, SM_Chair_01, Sign_02, Shelf_02, Sign_03, SM_TV_01, SM_Lamp_01, SM_Stove_01, SM_Counter_01, SM_Sink_01 |
| walk_04 | night | 13 | 16 | 6 | 0 | 10 | SM_CoffeeTable_01, SM_Chair_02, Sign_02, SM_Sink_01, SM_TV_01, SM_Plant_02, SM_Shelf_01 |
| walk_05 | day | 16 | 20 | 3 | 0 | 17 | SM_Lamp_01, SM_Plant_02, Shelf_02, SM_Table_01, SM_Chair_01, SM_Chair_02, Sign_02, SM_Sofa_01, SM_CoffeeTable_01, Plant_03, Sign_03, SM_Stove_01, Sign_04 |
| walk_06 | overcast | 16 | 22 | 12 | 4 | 6 | SM_CoffeeTable_01, Plant_03, Sign_02, SM_Lamp_01 |
| walk_07 | evening | 16 | 20 | 7 | 0 | 13 | SM_CoffeeTable_01, SM_Lamp_01, SM_Chair_01, SM_Chair_02, Plant_03, Sign_02, Sign_03, SM_Stove_01, Sign_04 |
| walk_08 | night | 17 | 21 | 13 | 1 | 7 | Plant_03, Sign_04, Sign_02, SM_Chair_02 |
| walk_09 | day | 17 | 25 | 9 | 3 | 13 | Plant_03, Sign_02, Sign_03, SM_Table_01, SM_Chair_01, SM_Lamp_01, SM_Stove_01, SM_Sink_01 |
| walk_10 | overcast | 15 | 24 | 8 | 2 | 14 | Plant_03, Sign_04, Sign_02, SM_Chair_02, SM_Chair_01, Sign_03, SM_Lamp_01 |
