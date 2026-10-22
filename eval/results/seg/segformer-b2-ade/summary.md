# Segmentation benchmark: `segformer-b2-ade`

nvidia/segformer-b2-finetuned-ade-512-512 · FP16 on cuda · every 10th frame · scored classes: 15 (door, window excluded)

**Speed:** 32.11 ms/frame median, 42.12 ms p95 (model forward at (512, 512))

| Split | mIoU (%) ↑ | Pixel acc (%) ↑ |
|---|---|---|
| **all** | 42.7 | 88.4 |
| day | 43.5 | 88.7 |
| overcast | 44.1 | 90.3 |
| evening | 35.3 | 85.8 |
| night | 43.7 | 87.6 |

| Class | IoU (%) ↑ | Most confused with | Top ADE20K labels predicted |
|---|---|---|---|
| sink | 0.4 | other (48 %) | step 30%, cabinet 28%, wall 18% |
| counter | 1.4 | shelf (61 %) | cabinet 59%, bathtub 20%, wall 9% |
| sign | 3.4 | wall (64 %) | wall 64%, signboard 35%, door 1% |
| lamp | 8.5 | wall (72 %) | wall 72%, sconce 22%, ceiling 3% |
| sofa | 12.2 | chair (84 %) | armchair 56%, chair 28%, sofa 12% |
| outdoor | 23.8 | sign (41 %) | painting 39%, windowpane 19%, mirror 12% |
| tv | 28.7 | other (25 %) | television receiver 29%, table 21%, box 20% |
| chair | 33.8 | table (5 %) | chair 86%, table 5%, floor 3% |
| shelf | 48.2 | wall (17 %) | shelf 69%, wall 17%, curtain 4% |
| table | 62.2 | chair (8 %) | coffee table 42%, table 41%, chair 8% |
| plant | 67.5 | wall (9 %) | plant 39%, pot 22%, vase 13% |
| stove | 72.3 | wall (10 %) | stove 65%, microwave 13%, wall 10% |
| floor | 87.8 | wall (4 %) | floor 90%, wall 4%, table 3% |
| wall | 93.2 | shelf (2 %) | wall 96%, shelf 2%, door 1% |
| ceiling | 96.5 | wall (2 %) | ceiling 98%, wall 2%, curtain 0% |

## Worst frames

![walk_09 frame 380](figures/worst_walk_09_000380.png)

![walk_08 frame 260](figures/worst_walk_08_000260.png)

![walk_03 frame 220](figures/worst_walk_03_000220.png)

![walk_07 frame 260](figures/worst_walk_07_000260.png)
