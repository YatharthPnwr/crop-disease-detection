# Crop Disease Detection

Explainable crop-disease classifier for leaf photos. Two CNNs are trained on
PlantVillage (38 classes, 14 crops) and compared:

| Model | Init | Params | Role |
| --- | --- | ---: | --- |
| Custom CNN (5 conv blocks + GAP head) | random, trained from scratch | 2.36M | baseline |
| EfficientNet-B0 | ImageNet, two-phase fine-tuning | 4.06M | main model, app default |

Predictions are explained with Grad-CAM / Grad-CAM++, and a local Streamlit app
names the disease from a camera or uploaded photo.

- Results: [reports/results.md](reports/results.md) (filled in after training)
- Full plan: [PLAN.md](PLAN.md)

## How it works

1. **Data.** PlantVillage color images (54,305) from Kaggle
   [`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
   (CC BY-NC-SA 4.0). The `segmented/` copy supplies leaf masks for the Grad-CAM check.
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
app/            streamlit_app.py, inference.py, models/ (weights come from a release)
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
modal run --detach modal_app.py::grid     # 2 models x 3 seeds (+ random-split leakage run)
modal run modal_app.py::collect           # metrics -> runs/, best weights -> app/models/
modal run modal_app.py::explain           # Grad-CAM analysis -> reports/
python scripts/export_models.py           # app/models/*.pt -> .onnx
python scripts/aggregate.py && python scripts/make_figures.py && python scripts/latency.py
```

## Run the app

```bash
pip install -r requirements-app.txt       # after the CPU torch install above
python scripts/fetch_models.py            # trained weights from the GitHub release
streamlit run app/streamlit_app.py
```

The camera works with the laptop webcam on `localhost`. A phone on the same Wi-Fi
can only use its camera over HTTPS, so either upload a photo taken on the phone or
start Streamlit with `--server.sslCertFile` / `--server.sslKeyFile`.

## Limitations

- Trained only on lab photos (single leaf, plain background). Field photos will be
  less accurate; testing on PlantDoc field images is future work.
- Knows only the 38 PlantVillage classes; other crops get a wrong or "unsure" answer.
- `data/disease_info.json` is a draft and is not yet verified against agricultural
  extension sources. Do not use it as treatment advice.

## References

- PlantVillage: Hughes & Salathé (2015); Mohanty, Hughes & Salathé (2016).
- EfficientNet: Tan & Le (2019). Grad-CAM: Selvaraju et al. (2017). Grad-CAM++: Chattopadhay et al. (2018).
- Repos reviewed while planning: see [PLAN.md](PLAN.md), section 2.
