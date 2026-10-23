# Scene graph vs ground truth: `owl_modeldepth`

Every 3th frame (10 Hz) · match = same class and 3D IoU ≥ 0.25 · GT poses

**Recall 51.9 % · precision 44.8 % · 19 duplicates · 82 false positives · 3D IoU 0.36 · centroid error 19.9 cm · size error 22.1 cm**

**Graph update:** 5.22 ms/frame median, 10.85 ms p95 (CPU, numpy) · models: depth 19 ms, OWLv2 59 ms

| Class | Recall (%) |
|---|---|
| lamp | 0 |
| sign | 0 |
| sink | 0 |
| chair | 40 |
| counter | 50 |
| table | 65 |
| tv | 70 |
| plant | 71 |
| stove | 75 |
| shelf | 85 |
| sofa | 100 |

| Walk | Lighting | GT objects | In graph | Matched | Dup. | False pos. | Missed |
|---|---|---|---|---|---|---|---|
| walk_01 | day | 15 | 15 | 7 | 1 | 7 | SM_TV_01, SM_Table_01, SM_Chair_02, SM_Lamp_01, SM_CoffeeTable_01, Sign_04, Sign_02, SM_Sink_01 |
| walk_02 | overcast | 15 | 13 | 8 | 2 | 3 | SM_Lamp_01, Shelf_02, SM_Chair_02, Sign_02, Plant_03, Sign_03, Sign_04 |
| walk_03 | evening | 18 | 22 | 9 | 2 | 11 | Plant_03, Sign_04, SM_Chair_02, SM_Chair_01, Sign_02, Sign_03, SM_Lamp_01, SM_Counter_01, SM_Sink_01 |
| walk_04 | night | 13 | 24 | 9 | 3 | 12 | SM_CoffeeTable_01, Sign_02, SM_Sink_01, SM_Shelf_01 |
| walk_05 | day | 16 | 13 | 8 | 1 | 4 | SM_TV_01, SM_Lamp_01, SM_Chair_02, Sign_02, Plant_03, Sign_03, SM_Stove_01, Sign_04 |
| walk_06 | overcast | 16 | 23 | 12 | 6 | 5 | Plant_03, Sign_02, SM_Lamp_01, SM_Sink_01 |
| walk_07 | evening | 16 | 17 | 6 | 0 | 11 | SM_Shelf_01, SM_Lamp_01, SM_Table_01, SM_Chair_01, SM_Chair_02, Plant_03, Sign_02, Sign_03, SM_Stove_01, Sign_04 |
| walk_08 | night | 17 | 18 | 7 | 1 | 10 | SM_CoffeeTable_01, Plant_03, Sign_04, Sign_02, SM_Chair_02, SM_Chair_01, SM_TV_01, SM_Lamp_01, SM_Counter_01, SM_Sink_01 |
| walk_09 | day | 17 | 21 | 9 | 1 | 11 | SM_CoffeeTable_01, Plant_03, Sign_02, Sign_03, SM_Chair_01, SM_Lamp_01, SM_Counter_01, SM_Sink_01 |
| walk_10 | overcast | 15 | 17 | 7 | 2 | 8 | Plant_03, Sign_04, SM_Table_01, Sign_02, SM_Chair_02, SM_Chair_01, Sign_03, SM_Lamp_01 |
