"""PlantDoc cropping, class mapping, dedupe against PlantVillage, background swap, domain metrics."""

import csv

import numpy as np
from PIL import Image

from cropdoc.data import BackgroundSwap
from cropdoc.evaluate import domain_metrics
from cropdoc.plantdoc import CLASS_MAP, build_plantdoc, write_combined_split


def _smooth(rng, size):
    return Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).resize(size, Image.BILINEAR)


def test_class_map_targets_are_plantvillage_names():
    assert len(CLASS_MAP) == 29
    assert all("___" in v for v in CLASS_MAP.values())


def test_build_plantdoc(tmp_path):
    rng = np.random.default_rng(0)
    det, pv = tmp_path / "det", tmp_path / "color"
    (pv / "Tomato___healthy").mkdir(parents=True)
    lab = _smooth(rng, (256, 256))
    lab.save(pv / "Tomato___healthy" / "lab0.JPG")
    pv_rows = [{"path": "Tomato___healthy/lab0.JPG", "label": "Tomato___healthy", "group": 0, "split": "train"}]
    for split, folder, n in (("train", "TRAIN", 21), ("test", "TEST", 7)):
        (det / folder).mkdir(parents=True)
        with open(det / f"{split}_labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
            for i in range(n):
                img = _smooth(rng, (300, 200))
                name = f"{split} img {i}.jpg"
                img.save(det / folder / name)
                cls = "Tomato leaf" if i % 2 else "Tomato Early blight leaf"
                w.writerow([name.replace(" ", "%20"), 300, 200, cls, 10, 10, 150, 190])
                w.writerow([name.replace(" ", "%20"), 300, 200, cls, 5, 5, 20, 20])  # too small
            # a "field" photo that is really the lab image
            lab.resize((300, 200)).save(det / folder / f"{split}_copy.jpg")
            w.writerow([f"{split}_copy.jpg", 300, 200, "Tomato leaf", 0, 0, 300, 200])
    rows, summary = build_plantdoc(det, tmp_path / "crops", pv, pv_rows, workers=1)
    assert summary["skipped"]["too_small"] == 28
    assert summary["skipped"]["duplicate_of_plantvillage"] == 2
    assert {r["label"] for r in rows} == {"Tomato___healthy", "Tomato___Early_blight"}
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert all((tmp_path / "crops" / r["path"]).exists() for r in rows)
    out = tmp_path / "combined.csv"
    write_combined_split(pv_rows, rows, out)
    lines = list(csv.DictReader(open(out)))
    assert lines[0]["path"].startswith("color/") and lines[-1]["path"].startswith("plantdoc/")


def test_background_swap(tmp_path):
    rng = np.random.default_rng(1)
    seg, bgs = tmp_path / "segmented" / "A___x", tmp_path / "bgs"
    seg.mkdir(parents=True)
    bgs.mkdir()
    leaf = np.zeros((256, 256, 3), np.uint8)
    leaf[64:192, 64:192] = 200
    Image.fromarray(leaf).save(seg / "img0_final_masked.jpg")
    Image.fromarray(np.full((300, 300, 3), 40, np.uint8)).save(bgs / "bg0.jpg")
    swap = BackgroundSwap(tmp_path / "segmented", bgs, p=1.0)
    img = Image.fromarray(np.full((256, 256, 3), 120, np.uint8))
    out = np.asarray(swap("color/A___x/img0.JPG", img))
    assert abs(int(out[128, 128, 0]) - 120) < 5      # leaf kept
    assert abs(int(out[5, 5, 0]) - 40) < 10          # background replaced
    assert swap("plantdoc/A___x/z.jpg", img) is img  # field rows untouched


def test_domain_metrics_masked():
    y = np.array([0, 1, 2, 2])
    probs = np.array([[.9, .05, .05, 0], [.1, .8, .1, 0], [.1, .1, .2, .6], [.1, .1, .7, .1]])
    dm = domain_metrics(y, probs, ["plantvillage", "plantvillage", "plantdoc", "plantdoc"], field_classes=[2])
    assert dm["plantvillage"]["accuracy"] == 1.0
    assert dm["plantdoc"]["accuracy"] == 0.5 and dm["plantdoc"]["accuracy_masked"] == 1.0
