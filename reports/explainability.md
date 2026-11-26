# Explainability results

Grad-CAM and Grad-CAM++ on the last 7x7 stage, 500 random test images per model (491 with a leaf mask from the PlantVillage segmented twin, the rest from an HSV fallback).
On average the leaf covers 59.5% of the 224x224 input, so that is the chance level for the energy ratio.

| Model (run) | CAM energy on leaf: Grad-CAM | Grad-CAM++ | Pointing game: Grad-CAM | Grad-CAM++ | Accuracy: normal image | black-background image | Random-head CAM correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom CNN (custom_s0) | 72.3 % | 72.2 % | 89.8 % | 89.0 % | 99.6 % | 76.8 % | 0.20 |
| EfficientNet-B0 (effb0_s2) | 69.2 % | 79.5 % | 86.4 % | 82.2 % | 99.4 % | 82.3 % | 0.19 |

## Reading

- Both models put more CAM mass on the leaf than chance, but 20-30 % of it falls on the background, and for 10-14 % of images the CAM peak lies off the leaf.
- Accuracy drops sharply when the same test leaves are shown on a black background (about 99.5 % to 77-82 %). The models partly rely on background or lighting context. Part of the drop is also plain distribution shift, since no training image had a black background.
- CAMs from a randomly re-initialised head correlate only about 0.2 with the trained ones, so the maps depend on what the model learned (sanity check passed).
- Most test errors are Corn gray leaf spot vs Northern leaf blight; the CAMs sit on the lesions, which look alike.

Figures: `figures/cam_<run>_correct.png`, `figures/cam_<run>_wrong.png`. Per-image scores: `explain_<run>.json`.
