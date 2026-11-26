"""Grouped split: near-duplicates share a group and never span two splits."""

import csv

import numpy as np
from PIL import Image

from cropdoc.splits import check_no_leakage, make_split


def _make_dataset(root, classes=3, per_class=21, dups=2):
    rng = np.random.default_rng(0)
    for c in range(classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            # smooth random image so pHash is stable under small edits
            base = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
            img = Image.fromarray(base).resize((128, 128), Image.BILINEAR)
            img.save(d / f"img{i}.png")
            for k in range(dups if i % 3 == 0 else 0):
                arr = np.clip(np.asarray(img, dtype=np.int16) + 6 * (k + 1), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"img{i}_dup{k}.png")


def test_duplicates_grouped_and_no_leak(tmp_path):
    root = tmp_path / "color"
    _make_dataset(root)
    out = tmp_path / "split.csv"
    summary = make_split(root, out, max_dist=6, seed=42, workers=2)

    with open(out) as f:
        rows = list(csv.DictReader(f))
    by_path = {r["path"]: r for r in rows}
    # every duplicate is in the same group (and split) as its original
    for r in rows:
        if "_dup" in r["path"]:
            orig = by_path[r["path"].split("_dup")[0] + ".png"]
            assert r["group"] == orig["group"]
            assert r["split"] == orig["split"]

    check_no_leakage(
        [r["path"] for r in rows],
        np.array([int(r["group"]) for r in rows]),
        np.array([r["split"] for r in rows], dtype=object),
    )
    assert set(summary["split_counts"]) == {"train", "val", "test"}
    assert summary["groups"] < summary["images"]
