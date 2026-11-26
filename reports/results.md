# Results

Lab test: 7,759 PlantVillage images (grouped split). Field test: 445 PlantDoc leaf crops
(27 of the 38 classes). Mean ± sample std over seeds, in %.

| Training data | Model | Seeds | Lab accuracy | Lab macro-F1 | Field accuracy | Field macro-F1 | Field accuracy (masked) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PlantVillage only | Custom CNN | 3 | 99.59 ± 0.04 | 99.34 ± 0.10 | 19.10 ± 0.81 | 21.28 ± 1.96 | 25.47 ± 1.87 |
| PlantVillage only | EfficientNet-B0 | 3 | 99.60 ± 0.01 | 99.42 ± 0.01 | 29.06 ± 1.01 | 26.22 ± 1.18 | 30.94 ± 1.28 |
| PlantVillage + PlantDoc | Custom CNN | 3 | 99.02 ± 0.10 | 98.68 ± 0.12 | 57.45 ± 1.70 | 54.37 ± 1.92 | 57.98 ± 2.17 |
| PlantVillage + PlantDoc | EfficientNet-B0 | 3 | 99.02 ± 0.32 | 98.73 ± 0.42 | 66.52 ± 0.81 | 63.55 ± 0.15 | 66.74 ± 0.81 |
| + PlantDoc, no background swap | EfficientNet-B0 | 1 | 99.43 | 99.23 | 68.99 | 66.78 | 69.21 |

Masked: predictions restricted to the 27 classes PlantDoc contains.

## Training cost

| Training data | Model | Epochs run | s / epoch | Train min | Params (M) |
| --- | --- | ---: | ---: | ---: | ---: |
| PlantVillage only | Custom CNN | 19.33 ± 1.15 | 37.17 ± 0.21 | 12.11 ± 0.74 | 2.36 |
| PlantVillage only | EfficientNet-B0 | 10.00 ± 0.00 | 40.10 ± 0.10 | 6.85 ± 0.02 | 4.06 |
| PlantVillage + PlantDoc | Custom CNN | 20.00 ± 0.00 | 44.23 ± 0.31 | 14.89 ± 0.11 | 2.36 |
| PlantVillage + PlantDoc | EfficientNet-B0 | 9.33 ± 0.58 | 48.03 ± 0.06 | 7.63 ± 0.47 | 4.06 |

## Leakage check (EfficientNet-B0, seed 0, PlantVillage only)

| Split | Lab accuracy | Lab macro-F1 |
| --- | ---: | ---: |
| Grouped by near-duplicates | 99.61 | 99.41 |
| Random per image | 99.52 | 99.35 |

Per-run numbers: [results.csv](results.csv).
