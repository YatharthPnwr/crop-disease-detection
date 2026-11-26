"""PlantDoc field images: crop leaves from bounding boxes, map classes, dedupe, split.

PlantDoc (Singh et al. 2020, CC BY 4.0) is web-scraped field photography. Its
object-detection release gives one box per leaf, so we crop single leaves (the
app's use case) and map its 29 class names onto PlantVillage classes.

Leakage guards:
- crops that are near-duplicates of a PlantVillage image (scraped lab photos) are dropped;
- near-duplicate crops across PlantDoc train/test are removed from train;
- validation is carved from PlantDoc train grouped by source photo.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .splits import compute_phashes, group_near_duplicates

# PlantDoc detection class -> PlantVillage class
CLASS_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def _find_image(folder: Path, name: str) -> Path | None:
    for cand in (name, unquote(name)):
        p = folder / cand
        if p.exists():
            return p
    return None


def crop_boxes(images_dir: str | Path, labels_csv: str | Path, out_dir: str | Path, source_split: str,
               min_side: int = 32, pad: float = 0.05, max_side: int = 512) -> tuple[list[dict], Counter]:
    """Crop every labelled box to out_dir/<PV class>/<split>_<image>_<k>.jpg. Returns rows and skip counts."""
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    rows, skipped = [], Counter()
    with open(labels_csv, newline="") as f:
        boxes = list(csv.DictReader(f))
    by_image: dict[str, list[dict]] = {}
    for b in boxes:
        by_image.setdefault(b["filename"], []).append(b)
    for img_i, (fname, bs) in enumerate(sorted(by_image.items())):
        src = _find_image(images_dir, fname)
        if src is None:
            skipped["missing_image"] += len(bs)
            continue
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                W, H = im.size
                for k, b in enumerate(bs):
                    cls = CLASS_MAP.get(b["class"].strip())
                    if cls is None:
                        skipped["unmapped_class"] += 1
                        continue
                    # boxes were drawn on the image at the CSV's width/height; rescale if the file differs
                    sx, sy = W / float(b["width"] or W), H / float(b["height"] or H)
                    x0, y0, x1, y1 = (float(b["xmin"]) * sx, float(b["ymin"]) * sy, float(b["xmax"]) * sx, float(b["ymax"]) * sy)
                    if min(x1 - x0, y1 - y0) < min_side:
                        skipped["too_small"] += 1
                        continue
                    px, py = (x1 - x0) * pad, (y1 - y0) * pad
                    box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(W, int(x1 + px)), min(H, int(y1 + py)))
                    crop = im.crop(box)
                    crop.thumbnail((max_side, max_side))
                    rel = f"{cls}/{source_split}_{img_i:04d}_{k:02d}.jpg"
                    (out_dir / cls).mkdir(parents=True, exist_ok=True)
                    crop.save(out_dir / rel, quality=92)
                    rows.append({"path": rel, "label": cls, "source_split": source_split, "source_image": f"{source_split}/{fname}"})
        except OSError:
            skipped["unreadable_image"] += len(bs)
    return rows, skipped


def build_plantdoc(det_root: str | Path, crops_dir: str | Path, pv_color_root: str | Path, pv_rows: list[dict],
                   max_dist: int = 4, seed: int = 42, workers: int = 8) -> tuple[list[dict], dict]:
    """Crop, dedupe and split PlantDoc. Returns rows (path relative to crops_dir) and a summary."""
    det_root, crops_dir = Path(det_root), Path(crops_dir)
    rows, skipped = [], Counter()
    for split, folder, csv_name in (("train", "TRAIN", "train_labels.csv"), ("test", "TEST", "test_labels.csv")):
        r, s = crop_boxes(det_root / folder, det_root / csv_name, crops_dir, split)
        rows += r
        skipped.update(s)

    # 1) drop crops that are near-duplicates of PlantVillage lab photos (same class)
    pd_hash = compute_phashes([str(crops_dir / r["path"]) for r in rows], workers)
    pv_classes = {r["label"] for r in rows}
    pv_sel = [r for r in pv_rows if r["label"] in pv_classes]
    pv_hash = compute_phashes([str(Path(pv_color_root) / r["path"]) for r in pv_sel], workers)
    pv_by_cls: dict[str, np.ndarray] = {}
    for r, h in zip(pv_sel, pv_hash):
        pv_by_cls.setdefault(r["label"], []).append(h)
    pv_by_cls = {k: np.array(v, dtype=np.uint64) for k, v in pv_by_cls.items()}
    keep = []
    for r, h in zip(rows, pd_hash):
        ref = pv_by_cls.get(r["label"])
        if ref is not None and np.bitwise_count(ref ^ h).min() <= max_dist:
            skipped["duplicate_of_plantvillage"] += 1
        else:
            keep.append((r, h))
    rows = [r for r, _ in keep]
    hashes = np.array([h for _, h in keep], dtype=np.uint64)

    # 2) near-duplicate groups inside PlantDoc; a crop's group also covers its source photo
    groups, _ = group_near_duplicates(hashes, [r["label"] for r in rows], max_dist=max_dist)
    src_ids = {s: i for i, s in enumerate(sorted({r["source_image"] for r in rows}))}
    parent = list(range(len(groups) + len(src_ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, r in enumerate(rows):  # union near-dup group node with source-photo node
        a, b = find(int(groups[i])), find(len(groups) + src_ids[r["source_image"]])
        if a != b:
            parent[max(a, b)] = min(a, b)
    gid = np.array([find(int(groups[i])) for i in range(len(rows))])

    test_groups = {g for g, r in zip(gid, rows) if r["source_split"] == "test"}
    final = []
    for g, r in zip(gid, rows):
        if r["source_split"] == "train" and g in test_groups:
            skipped["train_crop_near_duplicate_of_test"] += 1
            continue
        final.append({**r, "group": int(g)})

    # 3) PlantDoc val from train, grouped by source photo / near-dup group, stratified by class
    tr = [i for i, r in enumerate(final) if r["source_split"] == "train"]
    y = np.array([final[i]["label"] for i in tr])
    g = np.array([final[i]["group"] for i in tr])
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    _, val_rel = next(sgkf.split(np.zeros(len(tr)), y, g))
    val_set = {tr[i] for i in val_rel}
    for i, r in enumerate(final):
        r["split"] = "test" if r["source_split"] == "test" else ("val" if i in val_set else "train")

    summary = {
        "crops": len(final),
        "classes": len({r["label"] for r in final}),
        "split_counts": dict(Counter(r["split"] for r in final)),
        "source_photos": len({r["source_image"] for r in final}),
        "skipped": dict(skipped),
        "per_class": {c: dict(Counter(r["split"] for r in final if r["label"] == c)) for c in sorted({r["label"] for r in final})},
    }
    return final, summary


def write_combined_split(pv_rows: list[dict], pd_rows: list[dict], out_csv: str | Path) -> None:
    """One CSV for both domains, paths relative to a root holding color/ and plantdoc/."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "split", "domain"])
        for r in pv_rows:
            w.writerow([f"color/{r['path']}", r["label"], f"pv{r['group']}", r["split"], "plantvillage"])
        for r in pd_rows:
            w.writerow([f"plantdoc/{r['path']}", r["label"], f"pd{r['group']}", r["split"], "plantdoc"])
