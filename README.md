# Crop Disease Detection

Explainable crop-disease classifier for leaf photos. Two CNNs are trained on
PlantVillage (38 classes, 14 crops) and compared:

| Model | Init | Params | Role |
| --- | --- | ---: | --- |
| Custom CNN (5 conv blocks + GAP head) | random, trained from scratch | 2.36M | baseline |
| EfficientNet-B0 | ImageNet, two-phase fine-tuning | 4.06M | main model, app default |

Predictions are explained with Grad-CAM / Grad-CAM++, and a local Streamlit app
names the disease from a camera or uploaded photo.

- Full plan: [PLAN.md](PLAN.md)

## Results

Two held-out test sets: 7,759 PlantVillage lab photos and 445 PlantDoc field leaf crops
(27 classes). Mean ± std over 3 seeds, trained on Modal A10G. Details:
[reports/results.md](reports/results.md).

| Training data | Model | Lab accuracy | Field accuracy | Field macro-F1 |
| --- | --- | ---: | ---: | ---: |
| PlantVillage only | Custom CNN | 99.59 ± 0.04 % | 19.1 ± 0.8 % | 21.3 ± 2.0 % |
| PlantVillage only | EfficientNet-B0 | 99.60 ± 0.01 % | 29.1 ± 1.0 % | 26.2 ± 1.2 % |
| PlantVillage + PlantDoc | Custom CNN | 99.02 ± 0.10 % | 57.5 ± 1.7 % | 54.4 ± 1.9 % |
| **PlantVillage + PlantDoc** | **EfficientNet-B0 (app default)** | **99.02 ± 0.32 %** | **66.5 ± 0.8 %** | **63.6 ± 0.2 %** |

- **Lab accuracy alone is misleading.** Models trained only on PlantVillage reach 99.6 % on lab
  photos but 19–29 % on field photos (the PlantDoc paper reports 15–40 % for this setting).
- **Adding 7,065 PlantDoc field leaves** (cropped from detection boxes, oversampled 2x) raises field
  accuracy by about 37 points for both models while lab accuracy drops only ~0.6 points.
  The best published PlantDoc fine-tuning result is 70.5 %.
- **Background reliance dropped.** With the background blacked out, lab-test accuracy went from
  77–82 % to 92–93 %, and the Grad-CAM peak lands on the leaf more often (EfficientNet-B0 86 → 91 %,
  custom CNN 90 → 97 %) ([explainability](reports/explainability.md)).
- **Background-swap augmentation did not help:** a single run without it scored 69.0 % field
  accuracy, within or above the 65.8–67.4 % of the three runs with it.
- **Transfer learning matters most in the field:** EfficientNet-B0 beats the from-scratch CNN by
  9 points on field photos (they tie in the lab), trains in half the time and runs 3x faster on CPU
  (4.3 ms vs 12.9 ms, [efficiency](reports/efficiency.md)).
- Leakage check: a random lab split scored 99.52 % vs 99.61 % grouped, so no measurable leakage.

## How it works

1. **Data.** Lab: PlantVillage color images (54,305) from Kaggle
   [`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
   (CC BY-NC-SA 4.0); the `segmented/` copy supplies leaf masks for the Grad-CAM check.
   Field: [PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset) (CC BY 4.0),
   8,688 single-leaf crops from its bounding boxes, mapped onto the PlantVillage class names, with
   crops that duplicate PlantVillage photos or cross the train/test line removed
   ([`src/cropdoc/plantdoc.py`](src/cropdoc/plantdoc.py)).
2. **Leakage-safe split.** PlantVillage contains several photos of the same leaf.
   Near-duplicates are linked by perceptual hash (pHash, Hamming distance ≤ 6 within
   a class), merged into groups, and split by group with `StratifiedGroupKFold`
   (~71/14/14 train/val/test). No group spans two splits
   ([`src/cropdoc/splits.py`](src/cropdoc/splits.py)).
3. **Training** on Modal A10G GPUs, 3 seeds per model
   ([`src/cropdoc/train.py`](src/cropdoc/train.py)):
   - EfficientNet-B0: 2 epochs head-only (backbone frozen, BN in eval mode), then
     8 epochs full fine-tune with 10x lower backbone LR (1e-4 vs 1e-3).
   - Custom CNN: 20 epochs, LR 1e-3.
   - Both: AdamW (wd 0.01, none on BN/bias), warmup + cosine, label smoothing 0.1,
     bf16, batch 64, early stopping on val macro-F1 (patience 3).
4. **Evaluation:** accuracy, macro-F1, per-class F1, confusion matrix on the test split.
5. **Explainability:** Grad-CAM on the last 7x7 stage; leaf energy ratio and
   pointing game against segmentation masks; accuracy on black-background images;
   randomised-head sanity check ([`src/cropdoc/explain.py`](src/cropdoc/explain.py)).
6. **App:** ONNX Runtime prediction, PyTorch Grad-CAM, top-3, disease card, and an
   "unsure, retake photo" state whose threshold comes from the validation set.

## Repository layout

```text
configs/        base.yaml, custom.yaml, effb0.yaml
src/cropdoc/    data, splits, models, train, evaluate, explain, export, utils
modal_app.py    download, make_splits, smoke, grid, collect, explain (Modal)
scripts/        make_splits, dataset_stats, aggregate, make_figures, latency, export_models, fetch_models
app/            server.py + web/ (main app), inference.py, streamlit_app.py, models/, samples/
data/           disease_info.json, splits/pv_split.csv
reports/        results, efficiency, explainability outputs and figures
tests/          pytest suite (CPU, synthetic data)
```

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt && pip install -e .
pytest -q
```

## Train on Modal

Needs a Modal account (`modal token new`) and a Modal secret named `kaggle`
holding `KAGGLE_API_TOKEN`.

```bash
modal run modal_app.py::download          # PlantVillage -> volume cropdoc-data
modal run modal_app.py::make_splits       # writes data/splits/pv_split.csv
modal run modal_app.py::smoke             # 1 epoch on 5% of data: timing check
modal run --detach modal_app.py::grid     # lab-only: 2 models x 3 seeds (+ random-split leakage run)
modal run modal_app.py::field_data        # PlantDoc crops, backgrounds, combined lab+field split
modal run modal_app.py::field_baseline    # lab-only models on the field test set
modal run --detach modal_app.py::grid_field   # lab+field: 2 models x 3 seeds (+ no-background-swap run)
modal run modal_app.py::collect           # metrics -> runs/, best lab+field weights -> app/models/
modal run modal_app.py::explain --runs custom_field_s0,effb0_field_s1   # Grad-CAM analysis -> reports/
python scripts/export_models.py           # app/models/*.pt -> .onnx
python scripts/aggregate.py && python scripts/latency.py && python scripts/build_site_stats.py
```

## Run the app

```bash
pip install -r requirements-app.txt       # after the CPU torch install above
python scripts/fetch_models.py            # trained weights from the latest models-* release
python app/server.py                      # open http://localhost:8000
```

The web app (FastAPI + a hand-built frontend in `app/web/`) has:

- **Diagnose:** upload, drag-and-drop, paste, or a live in-page camera; eight unseen
  test leaves to try; a specimen view with a Grad-CAM / Grad-CAM++ heatmap you can
  fade in and out; top-3 predictions, a confidence meter with the "unsure" threshold,
  and disease notes.
- **Model report:** test accuracy and macro-F1 (mean ± std over 3 seeds), a
  head-to-head table, validation curves, per-class F1, the most confused pairs,
  Grad-CAM leaf-attention scores and dataset statistics. Numbers come from
  `reports/site_stats.json` (rebuild with `python scripts/build_site_stats.py`).
- Light and dark themes, keyboard accessible, works on phones.
- Deep links: `/?sample=tomato_early_blight.jpg` runs a sample, `/?theme=dark` forces a theme.
- API docs at `/api/docs` (`POST /api/predict`, `GET /api/stats`, `GET /api/models`).

The camera works on `localhost`. A phone on the same Wi-Fi (`python app/server.py --host 0.0.0.0`)
can only use its camera over HTTPS, so upload a photo there instead.
The older Streamlit version still runs with `streamlit run app/streamlit_app.py`.

## Limitations

- Field accuracy is about 66 %, far below the 99 % lab figure, and the field test set is small
  (445 leaves, as few as 4 per class), so per-class field numbers are noisy.
- 11 of the 38 classes have no field photos at all (e.g. Orange HLB, Grape Esca).
- PlantDoc is web-scraped, not farmers' phone photos.
- Knows only the 38 PlantVillage classes; other crops get a wrong or "unsure" answer.
- `data/disease_info.json` is a draft and is not yet verified against agricultural
  extension sources. Do not use it as treatment advice.

## References

- PlantVillage: Hughes & Salathé (2015); Mohanty, Hughes & Salathé (2016).
- EfficientNet: Tan & Le (2019). Grad-CAM: Selvaraju et al. (2017). Grad-CAM++: Chattopadhay et al. (2018).
- Repos reviewed while planning: see [PLAN.md](PLAN.md), section 2.
