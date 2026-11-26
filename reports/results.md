# Results (PlantVillage test set, grouped split)

Mean ± sample std over seeds. Accuracy and F1 in %.

| Model | Seeds | Test accuracy | Test macro-F1 | Test weighted F1 | Epochs run | s / epoch | Train min | Params (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 3 | 99.59 ± 0.03 | 99.34 ± 0.10 | 99.59 ± 0.03 | 19.33 ± 1.15 | 37.17 ± 0.21 | 12.11 ± 0.74 | 2.36 |
| EfficientNet-B0 | 3 | 99.60 ± 0.01 | 99.42 ± 0.02 | 99.60 ± 0.01 | 10.00 ± 0.00 | 40.10 ± 0.10 | 6.85 ± 0.02 | 4.06 |

## Leakage check (EfficientNet-B0, seed 0)

| Split | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Grouped by near-duplicates | 99.61 | 99.40 |
| Random per image | 99.52 | 99.35 |

Per-run numbers: [results.csv](results.csv).
