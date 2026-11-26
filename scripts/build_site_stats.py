"""Collect every number the website shows into one committed JSON file.

Reads runs/*/metrics.json (+ per-class / confusion files of the app models),
reports/field_baseline.json, reports/plantdoc_summary.json and the other reports/
outputs; writes reports/site_stats.json. Re-run after new training:

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
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if not len(a):
        return None
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def pretty(c: str) -> dict:
    crop, _, dis = c.partition("___")
    crop = crop.replace("_", " ").replace(",", "").replace("(including sour)", "").strip()
    dis = dis.replace("_", " ").strip()
    return {"class": c, "crop": crop, "disease": "Healthy" if dis.lower() == "healthy" else dis}


def family(run_id: str) -> str:
    if "random" in run_id:
        return "random"
    if "_field_nobg" in run_id:
        return "field_nobg"
    return "field" if "_field_" in run_id else "lab"


def domains_of(m: dict, baseline: dict) -> dict:
    return m.get("test_domains") or baseline.get(m["run_id"], {}).get("test_domains") or {}


def top_confusions(cm: np.ndarray, classes: list[str], k: int = 6) -> list[dict]:
    off = [(int(cm[i, j]), i, j) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    return [{"true": pretty(classes[i]), "pred": pretty(classes[j]), "count": n} for n, i, j in off[:k]]


def main() -> None:
    app_models = json.loads((ROOT / "app/models/models.json").read_text())
    efficiency = {r["model"]: r for r in json.loads((REPORTS / "efficiency.json").read_text())}
    baseline = json.loads((REPORTS / "field_baseline.json").read_text()) if (REPORTS / "field_baseline.json").exists() else {}
    runs = {}
    for f in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if not m["run_id"].startswith("smoke"):
            runs[m["run_id"]] = m

    def fam_stats(model: str, fam: str) -> dict | None:
        rs = [m for rid, m in runs.items() if m["model"] == model and family(rid) == fam]
        if not rs:
            return None
        dom = [domains_of(m, baseline) for m in rs]
        return {
            "seeds": len(rs),
            "lab_accuracy": mean_std([d.get("plantvillage", {}).get("accuracy", m["test"]["accuracy"]) for d, m in zip(dom, rs)]),
            "lab_macro_f1": mean_std([d.get("plantvillage", {}).get("macro_f1", m["test"]["macro_f1"]) for d, m in zip(dom, rs)]),
            "field_accuracy": mean_std([d.get("plantdoc", {}).get("accuracy") for d in dom]),
            "field_macro_f1": mean_std([d.get("plantdoc", {}).get("macro_f1") for d in dom]),
            "field_accuracy_masked": mean_std([d.get("plantdoc", {}).get("accuracy_masked") for d in dom]),
            "epochs_run": mean_std([m["epochs_run"] for m in rs]),
            "train_minutes": mean_std([m["train_minutes"] for m in rs]),
            "seconds_per_epoch": mean_std([m["mean_epoch_seconds"] for m in rs]),
            "curves": [
                {"seed": m["seed"], "points": [
                    {"epoch": h["global_epoch"] + 1, "phase": h["phase"],
                     "lab": h.get("val_f1_plantvillage", h["val_macro_f1"]), "field": h.get("val_f1_plantdoc")}
                    for h in m["history"]]}
                for m in sorted(rs, key=lambda m: m["seed"])
            ],
        }

    models = {}
    for model in MODELS:
        app_run = app_models[model]["run_id"]
        app_fam = family(app_run)
        d = RUNS / app_run
        classes = json.loads((d / "classes.json").read_text())
        eff = efficiency[DISPLAY_NAMES[model]]
        lab_pc = d / "test_per_class_plantvillage.json"
        lab_pc = lab_pc if lab_pc.exists() else d / "test_per_class.json"
        per_lab = json.loads(lab_pc.read_text())
        per_field = json.loads((d / "test_per_class_plantdoc.json").read_text()) if (d / "test_per_class_plantdoc.json").exists() else {}
        cm_all = np.load(d / "confusion.npy")
        cm_field = np.load(d / "confusion_plantdoc.npy") if (d / "confusion_plantdoc.npy").exists() else None
        app_m = runs[app_run]
        app_dom = domains_of(app_m, baseline)
        models[model] = {
            "display": DISPLAY_NAMES[model],
            "app_run": app_run,
            "app_family": app_fam,
            "unsure_threshold": app_models[model]["unsure_threshold"],
            "params_m": eff["params_M"],
            "gmacs": eff["gmacs"],
            "onnx_mb": eff.get("onnx_MB"),
            "latency_ms": {"onnx": eff.get("onnx_ms"), "torch": eff["torch_ms"]},
            "families": {fam: fam_stats(model, fam) for fam in ("lab", "field", "field_nobg")},
            "app_test": {
                "lab_errors": int(cm_all.sum() - np.trace(cm_all)) - (int(cm_field.sum() - np.trace(cm_field)) if cm_field is not None else 0),
                "lab_n": app_dom.get("plantvillage", {}).get("n"),
                "field_n": app_dom.get("plantdoc", {}).get("n"),
            },
            "per_class": [
                {**pretty(c), "lab": per_lab.get(c), "field": per_field.get(c) if per_field.get(c, {}).get("support") else None}
                for c in classes
            ],
            "top_confusions": {
                "lab": top_confusions(cm_all - (cm_field if cm_field is not None else 0), classes),
                "field": top_confusions(cm_field, classes) if cm_field is not None else [],
            },
        }
        ex = REPORTS / f"explain_{app_run}.json"
        if ex.exists():
            models[model]["explain"] = json.loads(ex.read_text())["summary"]

    rows = list(csv.DictReader(open(ROOT / "data/splits/pv_split.csv")))
    split_counts = Counter(r["split"] for r in rows)
    class_counts = Counter(r["label"] for r in rows)
    groups = Counter(r["group"] for r in rows)
    split_summary = json.loads((REPORTS / "split_summary.json").read_text())
    pd_summary = json.loads((REPORTS / "plantdoc_summary.json").read_text()) if (REPORTS / "plantdoc_summary.json").exists() else None
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
        "field_dataset": None if pd_summary is None else {
            "name": "PlantDoc", "crops": pd_summary["crops"], "classes": pd_summary["classes"],
            "split": pd_summary["split_counts"], "source_photos": pd_summary["source_photos"],
            "skipped": pd_summary["skipped"], "license": "CC BY 4.0",
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
