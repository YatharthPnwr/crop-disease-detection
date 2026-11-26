"""Grad-CAM and leaf-mask scoring on an untrained model (shapes and ranges only)."""

import numpy as np
import torch
from PIL import Image

from cropdoc.explain import (
    analyze,
    cam_grid_png,
    compute_cam,
    find_segmented,
    leaf_scores,
    mask_from_segmented,
    segmented_index,
)
from cropdoc.models import build_model


def _pair(root_c, root_s, cls, name, rng):
    (root_c / cls).mkdir(parents=True, exist_ok=True)
    (root_s / cls).mkdir(parents=True, exist_ok=True)
    arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(root_c / cls / f"{name}.JPG")
    seg = np.zeros_like(arr)
    seg[64:192, 64:192] = arr[64:192, 64:192]  # leaf = centre square
    Image.fromarray(seg).save(root_s / cls / f"{name}_final_masked.jpg")


def test_scores_and_segmented_lookup(tmp_path):
    rng = np.random.default_rng(0)
    c, s = tmp_path / "color", tmp_path / "segmented"
    for i in range(4):
        _pair(c, s, "A___x" if i % 2 else "B___y", f"img{i}", rng)
    idx = segmented_index(s)
    assert find_segmented(idx, "B___y/img0.JPG") is not None

    mask = mask_from_segmented(Image.open(s / "B___y" / "img0_final_masked.jpg"))
    assert mask.shape == (224, 224) and 0.2 < mask.mean() < 0.4
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[100:120, 100:120] = 1.0
    sc = leaf_scores(cam, mask)
    assert sc["energy_ratio"] > 0.99  # a few random leaf pixels are near-black

    model = build_model("custom_cnn", num_classes=2, pretrained=False)
    rows = [{"path": f"{'A___x' if i % 2 else 'B___y'}/img{i}.JPG", "label": "A___x" if i % 2 else "B___y"} for i in range(4)]
    res = analyze(model, c, s, rows, ["A___x", "B___y"], torch.device("cpu"), n=4, batch=2)
    summ = res["summary"]
    assert summ["images"] == 4 and summ["mask_from_segmented"] == 4
    assert 0.0 <= summ["gradcam_energy_ratio"] <= 1.0

    png = cam_grid_png(model, c, res["per_image"][:2], ["A___x", "B___y"], torch.device("cpu"), "t")
    assert png[:4] == b"\x89PNG"


def test_cam_shape():
    model = build_model("custom_cnn", num_classes=3, pretrained=False)
    cams = compute_cam(model, torch.randn(2, 3, 224, 224), [0, 2], "gradcam++")
    assert cams.shape == (2, 224, 224) and cams.min() >= 0 and cams.max() <= 1
