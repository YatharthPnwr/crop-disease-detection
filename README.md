# Crop Disease Detection

Explainable crop-disease classifier for leaf photos. Two CNNs are trained on
PlantVillage (38 classes, 14 crops) and compared:

- **Custom CNN** trained from scratch (baseline)
- **EfficientNet-B0** fine-tuned from ImageNet weights (main model)

Predictions are explained with Grad-CAM, and a local Streamlit app names the
disease from a camera or uploaded photo.

Status: work in progress. See [PLAN.md](PLAN.md) for the full project plan.
