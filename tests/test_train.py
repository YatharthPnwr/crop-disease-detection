"""End-to-end CPU smoke test of the training loop on a tiny synthetic dataset."""

import json

import numpy as np
from PIL import Image

from cropdoc.splits import make_split
from cropdoc.train import run
from cropdoc.utils import load_config


def _dataset(root, classes=3, per_class=14):
    rng = np.random.default_rng(1)
    for c in range(classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            base = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
            base[..., c] = 255  # make classes separable
            Image.fromarray(base).resize((96, 96), Image.BILINEAR).save(d / f"{i}.png")


def test_custom_cnn_trains_and_resumes(tmp_path):
    root = tmp_path / "color"
    _dataset(root)
    split = tmp_path / "split.csv"
    make_split(root, split, workers=1)

    cfg = load_config("custom")
    cfg["data"].update(batch_size=4, num_workers=0)
    cfg["phases"][0]["epochs"] = 2
    out = tmp_path / "run"
    m = run(cfg, root, split, out, seed=0)

    for f in ("best.pt", "last.pt", "metrics.json", "classes.json", "confusion.npy", "test_per_class.json"):
        assert (out / f).exists(), f
    assert m["epochs_run"] == 2 and 0.0 <= m["test"]["accuracy"] <= 1.0
    assert json.loads((out / "classes.json").read_text()) == ["class_0", "class_1", "class_2"]
    # a finished run is not retrained
    assert run(cfg, root, split, out, seed=0)["history"] == m["history"]
