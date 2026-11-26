"""Shared preprocessing, augmentation and the split-driven dataset.

`preprocess()` is the single eval-time path. Training evaluation, ONNX export and
the Streamlit app all call it, so train and serve see identical inputs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
RESIZE = 256


def load_image(source) -> Image.Image:
    """Open a path, file-like object or PIL image; fix EXIF orientation; return RGB."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = ImageOps.exif_transpose(img)  # phone photos carry an orientation tag
    return img.convert("RGB")


def geometry_transform(image_size: int = IMAGE_SIZE, resize: int = RESIZE) -> Callable:
    """Resize + centre-crop only. Used to align masks and CAM overlays with model inputs."""
    return T.Compose([T.Resize(resize), T.CenterCrop(image_size)])


def eval_transform(image_size: int = IMAGE_SIZE, resize: int = RESIZE) -> Callable:
    """Deterministic eval transform on an RGB PIL image."""
    return T.Compose(
        [
            geometry_transform(image_size, resize),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess(source, image_size: int = IMAGE_SIZE, resize: int = RESIZE) -> torch.Tensor:
    """The one preprocessing path shared by evaluation, export and the app. Returns (3, H, W)."""
    return eval_transform(image_size, resize)(load_image(source))


def train_transform(image_size: int = IMAGE_SIZE) -> Callable:
    """Training augmentation. Hue jitter stays tiny: colour carries the disease signal."""
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25),
        ]
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation for display. Accepts (3,H,W) or (B,3,H,W); returns [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# ---------------------------------------------------------------- split CSV


SPLIT_COLUMNS = ("path", "label", "group", "split")


def read_split_csv(csv_path: str | Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = set(SPLIT_COLUMNS) - set(rows[0].keys()) if rows else set(SPLIT_COLUMNS)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    return rows


def class_names(rows: Iterable[dict]) -> list[str]:
    """Sorted class list: index i in the model output is class_names[i]."""
    return sorted({r["label"] for r in rows})


def domain_of(row: dict) -> str:
    """'plantdoc' (field crops) or 'plantvillage' (lab photos)."""
    return row.get("domain") or ("plantdoc" if row["path"].startswith("plantdoc/") else "plantvillage")


class LeafDataset(Dataset):
    """Images listed in the split CSV, read relative to `root`.

    `pre(rel_path, img) -> img` runs before the transform (used for background swap).
    """

    def __init__(
        self,
        root: str | Path,
        rows: list[dict],
        classes: list[str],
        transform: Callable,
        pre: Callable | None = None,
    ):
        self.root = Path(root)
        index = {c: i for i, c in enumerate(classes)}
        self.items = [(r["path"], index[r["label"]]) for r in rows]
        self.domains = [domain_of(r) for r in rows]
        self.transform = transform
        self.pre = pre

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rel, label = self.items[i]
        img = load_image(self.root / rel)
        if self.pre is not None:
            img = self.pre(rel, img)
        return self.transform(img), label


class BackgroundSwap:
    """Paste a PlantVillage leaf onto a blurred field background (probability p).

    The leaf mask comes from the image's `segmented/` twin. Targets the shortcut found by
    the Grad-CAM analysis: the models partly relied on PlantVillage's uniform backgrounds.
    Only applies to lab rows (paths under `color/`).
    """

    def __init__(self, seg_root: str | Path, bg_dir: str | Path, p: float = 0.3, threshold: int = 25):
        import re

        self.p, self.threshold = p, threshold
        norm = lambda n: re.sub(r"_final_masked$", "", Path(n).stem.lower())  # noqa: E731
        self._norm = norm
        self.seg = {(f.parent.name, norm(f.name)): f for f in Path(seg_root).glob("*/*")}
        self.bgs = sorted(Path(bg_dir).glob("*.jpg"))
        if not self.bgs:
            raise FileNotFoundError(f"no background images in {bg_dir}")

    def __call__(self, rel: str, img: Image.Image) -> Image.Image:
        import random

        from PIL import ImageFilter

        if not rel.startswith("color/") or random.random() >= self.p:
            return img
        _, cls, name = rel.split("/", 2)
        seg_path = self.seg.get((cls, self._norm(name)))
        if seg_path is None:
            return img
        seg = load_image(seg_path).resize(img.size)
        mask = Image.fromarray(((np.asarray(seg).max(axis=-1) > self.threshold) * 255).astype(np.uint8))
        mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2))  # trim halo, feather edge
        bg = load_image(random.choice(self.bgs))
        bw, bh = bg.size
        side = min(bw, bh)
        s = int(side * random.uniform(0.5, 1.0))
        x0, y0 = random.randint(0, bw - s), random.randint(0, bh - s)
        bg = bg.crop((x0, y0, x0 + s, y0 + s)).resize(img.size)
        bg = bg.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 4.0)))
        return Image.composite(img, bg, mask)


def split_rows(rows: list[dict], split: str, fraction: float = 1.0, seed: int = 0) -> list[dict]:
    """Rows of one split, optionally a reproducible random fraction (smoke runs)."""
    out = [r for r in rows if r["split"] == split]
    if fraction < 1.0:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(out), generator=g)[: max(1, int(len(out) * fraction))]
        out = [out[i] for i in sorted(keep.tolist())]
    return out
