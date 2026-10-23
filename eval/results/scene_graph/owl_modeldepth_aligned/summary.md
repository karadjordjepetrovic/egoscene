# Scene graph vs ground truth: `owl_modeldepth_aligned`

Every 3th frame (10 Hz) · match = same class and 3D IoU ≥ 0.25 · GT poses

**Recall 42.4 % · precision 33.8 % · 15 duplicates · 116 false positives · 3D IoU 0.38 · centroid error 17.1 cm · size error 19.2 cm**

**Graph update:** 5.30 ms/frame median, 11.13 ms p95 (CPU, numpy) · models: depth 19 ms, OWLv2 59 ms

| Class | Recall (%) |
|---|---|
| sign | 0 |
| lamp | 11 |
| chair | 20 |
| sink | 33 |
| table | 45 |
| stove | 50 |
| counter | 50 |
| plant | 64 |
| shelf | 65 |
| tv | 70 |
| sofa | 75 |

| Walk | Lighting | GT objects | In graph | Matched | Dup. | False pos. | Missed |
|---|---|---|---|---|---|---|---|
| walk_01 | day | 15 | 14 | 8 | 0 | 6 | SM_Table_01, SM_Chair_01, SM_Chair_02, SM_Lamp_01, SM_CoffeeTable_01, Sign_04, Sign_02 |
| walk_02 | overcast | 15 | 12 | 7 | 0 | 5 | SM_Lamp_01, Shelf_02, SM_Table_01, SM_Chair_02, Sign_02, Plant_03, Sign_03, Sign_04 |
| walk_03 | evening | 18 | 28 | 5 | 3 | 20 | Plant_03, Sign_04, SM_Table_01, SM_Chair_02, SM_Chair_01, Sign_02, Shelf_02, Sign_03, SM_TV_01, SM_Lamp_01, SM_Stove_01, SM_Counter_01, SM_Sink_01 |
| walk_04 | night | 13 | 21 | 7 | 4 | 10 | SM_CoffeeTable_01, Sign_02, SM_Sink_01, SM_TV_01, SM_Plant_02, SM_Shelf_01 |
| walk_05 | day | 16 | 15 | 3 | 0 | 12 | SM_TV_01, SM_Lamp_01, Shelf_02, SM_Table_01, SM_Chair_01, SM_Chair_02, Sign_02, SM_Sofa_01, SM_CoffeeTable_01, Plant_03, Sign_03, SM_Stove_01, Sign_04 |
| walk_06 | overcast | 16 | 24 | 9 | 3 | 12 | SM_CoffeeTable_01, SM_Chair_01, Plant_03, Shelf_02, Sign_02, SM_Lamp_01, SM_Sink_01 |
| walk_07 | evening | 16 | 19 | 4 | 0 | 15 | SM_CoffeeTable_01, SM_Lamp_01, SM_Table_01, SM_Chair_01, SM_Chair_02, Plant_03, Shelf_02, Sign_02, SM_Sofa_01, Sign_03, SM_Stove_01, Sign_04 |
| walk_08 | night | 17 | 19 | 10 | 0 | 9 | Plant_03, Sign_04, Sign_02, SM_Chair_02, SM_Chair_01, SM_Plant_02, SM_Counter_01 |
| walk_09 | day | 17 | 24 | 7 | 2 | 15 | SM_CoffeeTable_01, Plant_03, Sign_02, Sign_03, SM_Chair_02, SM_Chair_01, SM_Lamp_01, SM_Stove_01, SM_Counter_01, SM_Sink_01 |
| walk_10 | overcast | 15 | 22 | 7 | 3 | 12 | Plant_03, Sign_04, Sign_02, SM_Chair_02, SM_Chair_01, Shelf_02, Sign_03, SM_Lamp_01 |
