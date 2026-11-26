# Explainability results

Grad-CAM and Grad-CAM++ on the last 7x7 stage, 500 random PlantVillage test images per model (491 with a leaf mask from the segmented twin, the rest from an HSV fallback). The leaf covers 59.5 % of the input on average: the chance level for the energy ratio.

| Training data | Model (run) | CAM mass on leaf: Grad-CAM | Grad-CAM++ | CAM peak on leaf | Accuracy: normal | background blacked out | Random-head CAM corr. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PlantVillage only | Custom CNN (custom_s0) | 72.3 % | 72.2 % | 89.8 % | 99.6 % | 76.8 % | 0.20 |
| PlantVillage only | EfficientNet-B0 (effb0_s2) | 69.2 % | 79.5 % | 86.4 % | 99.4 % | 82.3 % | 0.19 |
| + PlantDoc | Custom CNN (custom_field_s0) | 76.2 % | 75.3 % | 97.4 % | 98.6 % | 91.6 % | 0.11 |
| + PlantDoc | EfficientNet-B0 (effb0_field_s1) | 71.3 % | 83.4 % | 90.6 % | 98.8 % | 92.7 % | 0.15 |

## Reading

- Lab-only models partly used the PlantVillage background: blacking it out dropped accuracy to 77-82 %.
- After adding PlantDoc field photos, the same test drops accuracy only to 92-93 %, and the CAM peak lands on the leaf more often (custom CNN 90 -> 97 %, EfficientNet-B0 86 -> 91 %). Training on cluttered field photos made the models rely less on the uniform lab background.
- Part of the remaining drop is plain distribution shift: no training image had a pure black background.
- CAMs from a randomly re-initialised head correlate only 0.1-0.2 with the trained ones, so the maps depend on what the model learned (sanity check passed).

Figures: `figures/cam_<run>_correct.png`, `figures/cam_<run>_wrong.png`. Per-image scores: `explain_<run>.json`.
