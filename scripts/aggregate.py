"""Aggregate runs/*/metrics.json into per-run CSV and mean +- std tables (lab and field test).

Families: "lab" = trained on PlantVillage only, "field" = PlantVillage + PlantDoc
(+ background swap), "field_nobg" = ablation without background swap, "random" =
leakage check. Lab-only runs get their field numbers from reports/field_baseline.json.

    python scripts/aggregate.py --runs runs --out reports
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cropdoc.models import DISPLAY_NAMES

FIELDS = ["run_id", "model", "family", "seed", "lab_accuracy", "lab_macro_f1", "field_accuracy", "field_macro_f1",
          "field_accuracy_masked", "field_macro_f1_masked", "epochs_run", "mean_epoch_seconds", "train_minutes",
          "params", "unsure_threshold", "device", "git_commit"]
FAMILY_LABEL = {"lab": "PlantVillage only", "field": "PlantVillage + PlantDoc", "field_nobg": "+ PlantDoc, no background swap"}


def family(run_id: str) -> str:
    if "random" in run_id:
        return "random"
    if "_field_nobg" in run_id:
        return "field_nobg"
    return "field" if "_field_" in run_id else "lab"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = Path(args.out)
    baseline_path = out / "field_baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    rows = []
    for f in sorted(Path(args.runs).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if m["run_id"].startswith("smoke"):
            continue
        dom = m.get("test_domains") or baseline.get(m["run_id"], {}).get("test_domains") or {}
        pv, pd = dom.get("plantvillage", {}), dom.get("plantdoc", {})
        rows.append({
            "run_id": m["run_id"], "model": m["model"], "family": family(m["run_id"]), "seed": m["seed"],
            "lab_accuracy": pv.get("accuracy", m["test"]["accuracy"]),
            "lab_macro_f1": pv.get("macro_f1", m["test"]["macro_f1"]),
            "field_accuracy": pd.get("accuracy"), "field_macro_f1": pd.get("macro_f1"),
            "field_accuracy_masked": pd.get("accuracy_masked"), "field_macro_f1_masked": pd.get("macro_f1_masked"),
            "epochs_run": m["epochs_run"], "mean_epoch_seconds": m["mean_epoch_seconds"],
            "train_minutes": m["train_minutes"], "params": m["params"],
            "unsure_threshold": m.get("unsure_threshold"), "device": m["device"], "git_commit": m["git_commit"],
        })
    if not rows:
        raise SystemExit(f"no metrics.json under {args.runs}/")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    def ms(vals, pct=True):
        a = np.array([v for v in vals if v is not None], dtype=float) * (100 if pct else 1)
        if not len(a):
            return "–"
        return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else f"{a.mean():.2f}"

    lines = [
        "# Results",
        "",
        "Lab test: 7,759 PlantVillage images (grouped split). Field test: 445 PlantDoc leaf crops",
        "(27 of the 38 classes). Mean ± sample std over seeds, in %.",
        "",
        "| Training data | Model | Seeds | Lab accuracy | Lab macro-F1 | Field accuracy | Field macro-F1 | Field accuracy (masked) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fam in ("lab", "field", "field_nobg"):
        for model in ("custom_cnn", "efficientnet_b0"):
            rs = [r for r in rows if r["model"] == model and r["family"] == fam]
            if not rs:
                continue
            lines.append(
                f"| {FAMILY_LABEL[fam]} | {DISPLAY_NAMES[model]} | {len(rs)} | {ms([r['lab_accuracy'] for r in rs])} | "
                f"{ms([r['lab_macro_f1'] for r in rs])} | {ms([r['field_accuracy'] for r in rs])} | "
                f"{ms([r['field_macro_f1'] for r in rs])} | {ms([r['field_accuracy_masked'] for r in rs])} |")
    lines += ["", "Masked: predictions restricted to the 27 classes PlantDoc contains.", "",
              "## Training cost", "",
              "| Training data | Model | Epochs run | s / epoch | Train min | Params (M) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for fam in ("lab", "field"):
        for model in ("custom_cnn", "efficientnet_b0"):
            rs = [r for r in rows if r["model"] == model and r["family"] == fam]
            if rs:
                lines.append(f"| {FAMILY_LABEL[fam]} | {DISPLAY_NAMES[model]} | {ms([r['epochs_run'] for r in rs], False)} | "
                             f"{ms([r['mean_epoch_seconds'] for r in rs], False)} | {ms([r['train_minutes'] for r in rs], False)} | "
                             f"{rs[0]['params'] / 1e6:.2f} |")
    rand = [r for r in rows if r["family"] == "random"]
    if rand:
        grouped = [r for r in rows if r["model"] == "efficientnet_b0" and r["family"] == "lab" and r["seed"] == 0]
        lines += ["", "## Leakage check (EfficientNet-B0, seed 0, PlantVillage only)", "",
                  "| Split | Lab accuracy | Lab macro-F1 |", "| --- | ---: | ---: |"]
        for label, rs in (("Grouped by near-duplicates", grouped), ("Random per image", rand)):
            if rs:
                lines.append(f"| {label} | {rs[0]['lab_accuracy'] * 100:.2f} | {rs[0]['lab_macro_f1'] * 100:.2f} |")
    lines += ["", "Per-run numbers: [results.csv](results.csv)."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
