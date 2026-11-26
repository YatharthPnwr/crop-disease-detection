"""Collect every number the website shows into one committed JSON file.

Reads runs/*/metrics.json (+ confusion / per-class files of the app models) and the
reports/ outputs, writes reports/site_stats.json. Re-run after new training:

    python scripts/build_site_stats.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cropdoc.models import DISPLAY_NAMES

ROOT = Path(__file__).resolve().parents[1]
RUNS, REPORTS = ROOT / "runs", ROOT / "reports"
MODELS = ("custom_cnn", "efficientnet_b0")


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def pretty(c: str) -> dict:
    crop, _, dis = c.partition("___")
    crop = crop.replace("_", " ").replace(",", "").replace("(including sour)", "").strip()
    dis = dis.replace("_", " ").strip()
    return {"class": c, "crop": crop, "disease": "Healthy" if dis.lower() == "healthy" else dis}


def main() -> None:
    app_models = json.loads((ROOT / "app/models/models.json").read_text())
    efficiency = {r["model"]: r for r in json.loads((REPORTS / "efficiency.json").read_text())}
    runs = {}
    for f in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if not m["run_id"].startswith("smoke"):
            runs[m["run_id"]] = m

    models = {}
    for model in MODELS:
        seeds = [m for rid, m in runs.items() if m["model"] == model and "random" not in rid]
        app_run = app_models[model]["run_id"]
        d = RUNS / app_run
        classes = json.loads((d / "classes.json").read_text())
        per_class = json.loads((d / "test_per_class.json").read_text())
        cm = np.load(d / "confusion.npy")
        off = [(int(cm[i, j]), i, j) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
        off.sort(reverse=True)
        eff = efficiency[DISPLAY_NAMES[model]]
        models[model] = {
            "display": DISPLAY_NAMES[model],
            "app_run": app_run,
            "unsure_threshold": app_models[model]["unsure_threshold"],
            "params_m": eff["params_M"],
            "gmacs": eff["gmacs"],
            "onnx_mb": eff.get("onnx_MB"),
            "latency_ms": {"onnx": eff.get("onnx_ms"), "torch": eff["torch_ms"]},
            "test_accuracy": mean_std([m["test"]["accuracy"] for m in seeds]),
            "test_macro_f1": mean_std([m["test"]["macro_f1"] for m in seeds]),
            "test_weighted_f1": mean_std([m["test"]["weighted_f1"] for m in seeds]),
            "epochs_run": mean_std([m["epochs_run"] for m in seeds]),
            "train_minutes": mean_std([m["train_minutes"] for m in seeds]),
            "seconds_per_epoch": mean_std([m["mean_epoch_seconds"] for m in seeds]),
            "test_errors": int(cm.sum() - np.trace(cm)),
            "curves": [
                {"seed": m["seed"], "points": [
                    {"epoch": h["global_epoch"] + 1, "phase": h["phase"], "val_macro_f1": h["val_macro_f1"],
                     "val_acc": h["val_acc"], "train_acc": h["train_acc"]} for h in m["history"]]}
                for m in sorted(seeds, key=lambda m: m["seed"])
            ],
            "per_class": [{**pretty(c), **per_class[c]} for c in classes],
            "top_confusions": [
                {"true": pretty(classes[i]), "pred": pretty(classes[j]), "count": n} for n, i, j in off[:6]
            ],
        }
        ex = REPORTS / f"explain_{app_run}.json"
        if ex.exists():
            models[model]["explain"] = json.loads(ex.read_text())["summary"]

    rows = list(csv.DictReader(open(ROOT / "data/splits/pv_split.csv")))
    split_counts = Counter(r["split"] for r in rows)
    class_counts = Counter(r["label"] for r in rows)
    groups = Counter(r["group"] for r in rows)
    split_summary = json.loads((REPORTS / "split_summary.json").read_text())
    leak = {}
    for rid, key in (("effb0_s0", "grouped"), ("effb0_randomsplit_s0", "random")):
        if rid in runs:
            leak[key] = {"accuracy": runs[rid]["test"]["accuracy"], "macro_f1": runs[rid]["test"]["macro_f1"]}

    stats = {
        "dataset": {
            "name": "PlantVillage",
            "images": len(rows),
            "classes": len(class_counts),
            "crops": len({c.split("___")[0] for c in class_counts}),
            "split": {k: split_counts[k] for k in ("train", "val", "test")},
            "groups": len(groups),
            "linked_images": sum(n for n in groups.values() if n > 1),
            "phash_threshold": split_summary["max_dist"],
            "largest_class": {**pretty(max(class_counts, key=class_counts.get)), "count": max(class_counts.values())},
            "smallest_class": {**pretty(min(class_counts, key=class_counts.get)), "count": min(class_counts.values())},
            "license": "CC BY-NC-SA 4.0",
        },
        "leakage_check": leak,
        "models": models,
        "training": {"gpu": "NVIDIA A10G (Modal)", "seeds": 3, "batch_size": 64, "image_size": 224},
    }
    out = REPORTS / "site_stats.json"
    out.write_text(json.dumps(stats, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
