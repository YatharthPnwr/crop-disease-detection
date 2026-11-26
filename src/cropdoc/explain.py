"""Grad-CAM / Grad-CAM++ and a quantitative leaf-vs-background check.

For each test image we compute the CAM of the predicted class and compare it with
a leaf mask taken from the PlantVillage `segmented/` twin of that image:

- leaf energy ratio: share of CAM mass inside the leaf mask (chance = leaf area share)
- pointing game: is the CAM's maximum inside the mask?
- background-bias test: accuracy on the colour image vs its black-background twin
- sanity check: CAMs from a randomly re-initialised head should differ from the trained ones
"""

from __future__ import annotations

import copy
import io
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .data import IMAGE_SIZE, RESIZE, denormalize, geometry_transform, load_image, preprocess
from .models import gradcam_target_layer

CAM_METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}


def compute_cam(model, x: torch.Tensor, class_idx, method: str = "gradcam") -> np.ndarray:
    """CAMs in [0, 1] with shape (B, H, W) for the given target classes."""
    model.eval()
    targets = [ClassifierOutputTarget(int(c)) for c in class_idx]
    with CAM_METHODS[method](model=model, target_layers=[gradcam_target_layer(model)]) as cam:
        return cam(input_tensor=x, targets=targets)


def overlay(x: torch.Tensor, cam: np.ndarray, image_weight: float = 0.55) -> np.ndarray:
    """Heatmap over the (normalised) input tensor (3,H,W). Returns uint8 HxWx3."""
    rgb = denormalize(x.detach().cpu()).permute(1, 2, 0).numpy().astype(np.float32)
    return show_cam_on_image(rgb, cam, use_rgb=True, image_weight=image_weight)


# ------------------------------------------------------------------ leaf masks


def mask_from_segmented(seg: Image.Image, threshold: int = 20) -> np.ndarray:
    """Leaf mask from a black-background segmented image, aligned with the model input."""
    arr = np.asarray(geometry_transform(IMAGE_SIZE, RESIZE)(load_image(seg)))
    return arr.max(axis=-1) > threshold


def mask_from_hsv(img: Image.Image) -> np.ndarray:
    """Approximate leaf mask (green / yellow / brown, saturated pixels). Fallback only."""
    hsv = np.asarray(geometry_transform(IMAGE_SIZE, RESIZE)(load_image(img)).convert("HSV")).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 10) & (h <= 110) & (s > 50) & (v > 40)  # PIL hue is 0-255


def _norm_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"_final_masked$", "", stem)


def segmented_index(seg_root: str | Path) -> dict[tuple[str, str], Path]:
    """(class, normalised stem) -> segmented file. Segmented files end in `_final_masked`."""
    idx = {}
    for f in Path(seg_root).glob("*/*"):
        idx[(f.parent.name, _norm_stem(f.name))] = f
    return idx


def find_segmented(index: dict, rel_path: str) -> Path | None:
    cls, name = rel_path.split("/", 1)
    return index.get((cls, _norm_stem(name)))


def leaf_scores(cam: np.ndarray, mask: np.ndarray) -> dict:
    total = float(cam.sum()) or 1e-8
    return {
        "energy_ratio": float((cam * mask).sum() / total),
        "leaf_fraction": float(mask.mean()),
        "pointing_hit": bool(mask.flat[int(cam.argmax())]),
    }


def _randomised_head(model):
    m = copy.deepcopy(model)
    last = [mod for mod in m.classifier.modules() if isinstance(mod, torch.nn.Linear)][-1]
    torch.nn.init.normal_(last.weight, std=0.01)
    torch.nn.init.zeros_(last.bias)
    return m


# ------------------------------------------------------------------ analysis


def analyze(
    model,
    color_root: str | Path,
    seg_root: str | Path | None,
    test_rows: list[dict],
    classes: list[str],
    device: torch.device,
    n: int = 500,
    seed: int = 0,
    batch: int = 32,
) -> dict:
    """Run the leaf-vs-background analysis on n random test images."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(test_rows), size=min(n, len(test_rows)), replace=False)
    rows = [test_rows[i] for i in sorted(pick)]
    seg_idx = segmented_index(seg_root) if seg_root else {}
    cls_index = {c: i for i, c in enumerate(classes)}
    model = model.to(device).eval()
    rand_model = _randomised_head(model).to(device).eval()

    per_image = []
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        imgs = [load_image(Path(color_root) / r["path"]) for r in chunk]
        x = torch.stack([preprocess(im) for im in imgs]).to(device)
        seg_paths = [find_segmented(seg_idx, r["path"]) for r in chunk]
        with torch.no_grad():
            pred = model(x).argmax(1).cpu().numpy()
            xs = [preprocess(Image.open(p)) if p else None for p in seg_paths]
            seg_pred = [None] * len(chunk)
            have = [i for i, t in enumerate(xs) if t is not None]
            if have:
                sp = model(torch.stack([xs[i] for i in have]).to(device)).argmax(1).cpu().numpy()
                for i, p in zip(have, sp):
                    seg_pred[i] = int(p)
        cams = {m: compute_cam(model, x, pred, m) for m in CAM_METHODS}
        rand_cams = compute_cam(rand_model, x, pred, "gradcam")
        for i, r in enumerate(chunk):
            if seg_paths[i] is not None:
                mask, mask_src = mask_from_segmented(Image.open(seg_paths[i])), "segmented"
            else:
                mask, mask_src = mask_from_hsv(imgs[i]), "hsv"
            rec = {"path": r["path"], "label": cls_index[r["label"]], "pred": int(pred[i]),
                   "seg_pred": seg_pred[i], "mask": mask_src}
            for m in CAM_METHODS:
                rec |= {f"{m}_{k}": v for k, v in leaf_scores(cams[m][i], mask).items()}
            a, b = cams["gradcam"][i].ravel(), rand_cams[i].ravel()
            rec["random_head_corr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
            per_image.append(rec)

    def mean(key, subset=None):
        vals = [r[key] for r in (subset or per_image) if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    seg_rows = [r for r in per_image if r["seg_pred"] is not None]
    summary = {
        "images": len(per_image),
        "mask_from_segmented": sum(r["mask"] == "segmented" for r in per_image),
        "accuracy_color": float(np.mean([r["pred"] == r["label"] for r in per_image])),
        "accuracy_segmented": float(np.mean([r["seg_pred"] == r["label"] for r in seg_rows])) if seg_rows else None,
        "leaf_fraction": mean("gradcam_leaf_fraction"),
        "random_head_corr": mean("random_head_corr"),
    }
    for m in CAM_METHODS:
        summary[f"{m}_energy_ratio"] = mean(f"{m}_energy_ratio")
        summary[f"{m}_pointing_acc"] = mean(f"{m}_pointing_hit")
    return {"summary": summary, "per_image": per_image}


def cam_grid_png(
    model,
    color_root: str | Path,
    items: list[dict],
    classes: list[str],
    device: torch.device,
    title: str = "",
) -> bytes:
    """Figure: one column per image; rows = input, Grad-CAM, Grad-CAM++. Returns PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model.to(device).eval()
    x = torch.stack([preprocess(Path(color_root) / it["path"]) for it in items]).to(device)
    pred = [it["pred"] for it in items]
    cams = {m: compute_cam(model, x, pred, m) for m in CAM_METHODS}
    fig, axes = plt.subplots(3, len(items), figsize=(2.1 * len(items), 6.9), squeeze=False)
    for j, it in enumerate(items):
        ok = it["pred"] == it["label"]
        axes[0, j].imshow(denormalize(x[j].cpu()).permute(1, 2, 0).numpy())
        short = lambda c: classes[c].replace("___", ": ").replace("_", " ")[:28]
        axes[0, j].set_title(f"{'OK' if ok else 'WRONG'}\npred {short(it['pred'])}\ntrue {short(it['label'])}",
                             fontsize=6.5, color="darkgreen" if ok else "firebrick")
        for row, m in enumerate(CAM_METHODS, start=1):
            axes[row, j].imshow(overlay(x[j], cams[m][j]))
        for row in range(3):
            axes[row, j].axis("off")
    axes[1, 0].text(-0.08, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    axes[2, 0].text(-0.08, 0.5, "Grad-CAM++", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="right")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
