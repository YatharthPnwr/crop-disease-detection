"""Model loading, prediction and Grad-CAM for the Streamlit app (CPU only).

Prediction runs through ONNX Runtime when an .onnx file sits next to the .pt;
Grad-CAM needs gradients and module hooks, so it always uses the eager PyTorch model.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from cropdoc.data import load_image, preprocess  # noqa: E402
from cropdoc.explain import compute_cam, mask_from_hsv, overlay  # noqa: E402
from cropdoc.export import load_checkpoint  # noqa: E402
from cropdoc.models import DISPLAY_NAMES  # noqa: E402

MODELS_DIR = ROOT / "app" / "models"
DISEASE_INFO = ROOT / "data" / "disease_info.json"
ORDER = ("efficientnet_b0", "custom_cnn")  # default model first
MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE = 64


@dataclass
class LoadedModel:
    name: str
    display: str
    model: torch.nn.Module
    classes: list[str]
    session: object | None  # onnxruntime.InferenceSession
    unsure_threshold: float


def available_models(models_dir: Path = MODELS_DIR) -> list[str]:
    present = {p.stem for p in models_dir.glob("*.pt")}
    return [m for m in ORDER if m in present]


def load_model(name: str, models_dir: Path = MODELS_DIR) -> LoadedModel:
    model, classes, _ = load_checkpoint(models_dir / f"{name}.pt")
    session = None
    onnx_path = models_dir / f"{name}.onnx"
    if onnx_path.exists():
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tau = 0.5
    meta = models_dir / "models.json"
    if meta.exists():
        tau = json.loads(meta.read_text()).get(name, {}).get("unsure_threshold", tau)
    return LoadedModel(name, DISPLAY_NAMES.get(name, name), model, classes, session, float(tau))


def load_disease_info(path: Path = DISEASE_INFO) -> dict:
    return json.loads(path.read_text())["classes"]


class InvalidImage(ValueError):
    pass


def read_upload(data: bytes) -> Image.Image:
    """Validate an uploaded or captured image. Raises InvalidImage with a user-facing message."""
    import io

    if len(data) > MAX_BYTES:
        raise InvalidImage("Image is larger than 10 MB. Please use a smaller photo.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # catches truncated / corrupt files
        img = load_image(Image.open(io.BytesIO(data)))  # re-open after verify; EXIF fix; RGB
    except Exception as e:
        raise InvalidImage("That file is not a readable image (jpg, png or webp).") from e
    if min(img.size) < MIN_SIDE:
        raise InvalidImage(f"Image is too small ({img.size[0]}x{img.size[1]}). Use at least {MIN_SIDE} px per side.")
    return img


def predict(lm: LoadedModel, img: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """Class probabilities (n_classes,) and the preprocessed input tensor (1,3,224,224)."""
    x = preprocess(img).unsqueeze(0)
    if lm.session is not None:
        logits = lm.session.run(None, {lm.session.get_inputs()[0].name: x.numpy()})[0][0]
    else:
        with torch.no_grad():
            logits = lm.model(x)[0].numpy()
    e = np.exp(logits - logits.max())
    return e / e.sum(), x


def top_k(probs: np.ndarray, classes: list[str], k: int = 3) -> list[tuple[str, float]]:
    idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in idx]


def unsure_reason(probs: np.ndarray, tau: float, margin: float = 0.15) -> str | None:
    top2 = np.sort(probs)[::-1][:2]
    if top2[0] < tau:
        return f"top confidence {top2[0]:.0%} is below the {tau:.0%} threshold"
    if top2[0] - top2[1] < margin:
        return f"the top two classes are within {margin:.0%} of each other"
    return None


def leaf_fraction(img: Image.Image) -> float:
    """Share of green / yellow / brown saturated pixels: a rough 'is there a leaf?' heuristic."""
    return float(mask_from_hsv(img).mean())


def gradcam_overlay(lm: LoadedModel, x: torch.Tensor, class_idx: int, method: str = "gradcam") -> np.ndarray:
    cam = compute_cam(lm.model, x, [class_idx], method)[0]
    return overlay(x[0], cam)


def pretty(class_name: str) -> tuple[str, str]:
    crop, _, disease = class_name.partition("___")
    crop = crop.replace("_", " ").replace(",", "").replace("(including sour)", "").replace("(maize)", "(maize)").strip()
    disease = disease.replace("_", " ").strip()
    return crop, ("Healthy" if disease.lower() == "healthy" else disease)
