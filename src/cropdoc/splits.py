"""Leakage-safe PlantVillage split.

PlantVillage holds several photos of the same physical leaf. A random per-image
split puts siblings in both train and test. We link near-duplicates by
perceptual hash (pHash, Hamming distance <= max_dist, within each class), merge
links into groups with union-find, and split by group with StratifiedGroupKFold
so a group never spans two splits.
"""

from __future__ import annotations

import csv
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: str | Path) -> list[tuple[str, str]]:
    """(relative path, class) for every image in root/<class>/*."""
    root = Path(root)
    items = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f"{cls_dir.name}/{f.name}", cls_dir.name))
    if not items:
        raise FileNotFoundError(f"no images found under {root}/<class>/")
    return items


def _phash(path: str) -> int:
    with Image.open(path) as img:
        return int(str(imagehash.phash(img.convert("RGB"))), 16)


def compute_phashes(paths: list[str], workers: int = 8) -> np.ndarray:
    with Pool(workers) as pool:
        hashes = pool.map(_phash, paths, chunksize=256)
    return np.array(hashes, dtype=np.uint64)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_near_duplicates(
    hashes: np.ndarray, labels: list[str], max_dist: int = 6, chunk: int = 1024
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Group ids per image, plus the linked pairs (i, j, distance) for inspection."""
    n = len(hashes)
    uf = _UnionFind(n)
    labels_arr = np.asarray(labels)
    pairs: list[tuple[int, int, int]] = []
    for cls in np.unique(labels_arr):
        idx = np.flatnonzero(labels_arr == cls)
        h = hashes[idx]
        for start in range(0, len(idx), chunk):
            block = h[start : start + chunk, None] ^ h[None, :]
            dist = np.bitwise_count(block)
            rows, cols = np.nonzero(dist <= max_dist)
            keep = cols > rows + start  # upper triangle only, no self-pairs
            for r, c in zip(rows[keep], cols[keep]):
                i, j = int(idx[start + r]), int(idx[c])
                uf.union(i, j)
                pairs.append((i, j, int(dist[r, c])))
    roots = np.array([uf.find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids, pairs


def assign_splits(labels: list[str], groups: np.ndarray, seed: int = 42) -> np.ndarray:
    """~5/7 train, 1/7 val, 1/7 test (about 71/14/14), stratified by class, grouped."""
    y = np.asarray(labels)
    n = len(y)
    split = np.empty(n, dtype=object)
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    trval_idx, test_idx = next(outer.split(np.zeros(n), y, groups))
    split[test_idx] = "test"
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_rel, val_rel = next(inner.split(np.zeros(len(trval_idx)), y[trval_idx], groups[trval_idx]))
    split[trval_idx[tr_rel]] = "train"
    split[trval_idx[val_rel]] = "val"
    return split


def check_no_leakage(paths: list[str], groups: np.ndarray, split: np.ndarray) -> None:
    if len(set(paths)) != len(paths):
        raise AssertionError("duplicate paths in split")
    seen: dict[int, str] = {}
    for g, s in zip(groups.tolist(), split.tolist()):
        if seen.setdefault(g, s) != s:
            raise AssertionError(f"group {g} appears in both {seen[g]} and {s}")


def make_split(
    root: str | Path,
    out_csv: str | Path,
    max_dist: int = 6,
    seed: int = 42,
    workers: int = 8,
    pairs_csv: str | Path | None = None,
) -> dict:
    """Build and write the grouped split CSV. Returns a summary dict."""
    root = Path(root)
    items = list_images(root)
    paths = [p for p, _ in items]
    labels = [c for _, c in items]
    hashes = compute_phashes([str(root / p) for p in paths], workers=workers)
    groups, pairs = group_near_duplicates(hashes, labels, max_dist=max_dist)
    split = assign_splits(labels, groups, seed=seed)
    check_no_leakage(paths, groups, split)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "split"])
        w.writerows(zip(paths, labels, groups.tolist(), split.tolist()))

    if pairs_csv is not None:
        with open(pairs_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path_a", "path_b", "hamming"])
            w.writerows((paths[i], paths[j], d) for i, j, d in pairs)

    sizes = Counter(Counter(groups.tolist()).values())
    return {
        "images": len(paths),
        "classes": len(set(labels)),
        "groups": int(groups.max()) + 1,
        "linked_pairs": len(pairs),
        "max_dist": max_dist,
        "seed": seed,
        "group_size_histogram": {int(k): int(v) for k, v in sorted(sizes.items())},
        "split_counts": dict(Counter(split.tolist())),
    }
