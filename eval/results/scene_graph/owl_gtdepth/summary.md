# Scene graph vs ground truth: `owl_gtdepth`

Every 3th frame (10 Hz) · match = same class and 3D IoU ≥ 0.25 · GT poses

**Recall 72.2 % · precision 70.8 % · 12 duplicates · 35 false positives · 3D IoU 0.54 · centroid error 13.3 cm · size error 14.2 cm**

**Graph update:** 5.17 ms/frame median, 10.36 ms p95 (CPU, numpy) · models: depth 19 ms, OWLv2 59 ms

| Class | Recall (%) |
|---|---|
| sign | 0 |
| sink | 33 |
| chair | 60 |
| counter | 67 |
| stove | 75 |
| table | 80 |
| lamp | 89 |
| tv | 100 |
| plant | 100 |
| shelf | 100 |
| sofa | 100 |

| Walk | Lighting | GT objects | In graph | Matched | Dup. | False pos. | Missed |
|---|---|---|---|---|---|---|---|
| walk_01 | day | 15 | 15 | 10 | 1 | 4 | SM_Chair_02, SM_CoffeeTable_01, Sign_04, Sign_02, SM_Sink_01 |
| walk_02 | overcast | 15 | 12 | 11 | 0 | 1 | SM_Chair_02, Sign_02, Sign_03, Sign_04 |
| walk_03 | evening | 18 | 20 | 12 | 2 | 6 | Sign_04, SM_Chair_02, Sign_02, Sign_03, SM_Counter_01, SM_Sink_01 |
| walk_04 | night | 13 | 18 | 10 | 1 | 7 | SM_CoffeeTable_01, Sign_02, SM_Sink_01 |
| walk_05 | day | 16 | 12 | 11 | 1 | 0 | SM_Chair_02, Sign_02, Sign_03, SM_Stove_01, Sign_04 |
| walk_06 | overcast | 16 | 20 | 14 | 4 | 2 | SM_CoffeeTable_01, Sign_02 |
| walk_07 | evening | 16 | 15 | 10 | 0 | 5 | SM_Lamp_01, SM_Chair_02, Sign_02, Sign_03, SM_Stove_01, Sign_04 |
| walk_08 | night | 17 | 17 | 12 | 1 | 4 | Sign_04, Sign_02, SM_Chair_02, SM_Counter_01, SM_Sink_01 |
| walk_09 | day | 17 | 18 | 13 | 1 | 4 | SM_CoffeeTable_01, Sign_02, Sign_03, SM_Chair_02 |
| walk_10 | overcast | 15 | 14 | 11 | 1 | 2 | Sign_04, Sign_02, SM_Chair_02, Sign_03 |
