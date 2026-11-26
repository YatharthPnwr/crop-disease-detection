"""Aggregate runs/*/metrics.json into per-run CSV and a mean +- std table.

    python scripts/aggregate.py --runs runs --out reports
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cropdoc.models import DISPLAY_NAMES

FIELDS = ["run_id", "model", "seed", "split", "test_accuracy", "test_macro_f1", "test_weighted_f1",
          "best_val_macro_f1", "best_phase", "best_epoch", "epochs_run", "mean_epoch_seconds",
          "train_minutes", "params", "unsure_threshold", "device", "git_commit"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.runs).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if m["run_id"].startswith("smoke"):
            continue
        rows.append({
            "run_id": m["run_id"], "model": m["model"], "seed": m["seed"],
            "split": "random" if "random" in m["run_id"] else "grouped",
            "test_accuracy": m["test"]["accuracy"], "test_macro_f1": m["test"]["macro_f1"],
            "test_weighted_f1": m["test"]["weighted_f1"], "best_val_macro_f1": m["best_epoch"]["val_macro_f1"],
            "best_phase": m["best_epoch"]["phase"], "best_epoch": m["best_epoch"]["epoch"],
            "epochs_run": m["epochs_run"], "mean_epoch_seconds": m["mean_epoch_seconds"],
            "train_minutes": m["train_minutes"], "params": m["params"],
            "unsure_threshold": m.get("unsure_threshold"), "device": m["device"], "git_commit": m["git_commit"],
        })
    if not rows:
        raise SystemExit(f"no metrics.json under {args.runs}/")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    def ms(vals, pct=True):
        a = np.array(vals, dtype=float) * (100 if pct else 1)
        return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else f"{a.mean():.2f}"

    lines = [
        "# Results (PlantVillage test set, grouped split)",
        "",
        "Mean ± sample std over seeds. Accuracy and F1 in %.",
        "",
        "| Model | Seeds | Test accuracy | Test macro-F1 | Test weighted F1 | Epochs run | s / epoch | Train min | Params (M) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = {}
    for model in ("custom_cnn", "efficientnet_b0"):
        rs = [r for r in rows if r["model"] == model and r["split"] == "grouped"]
        if not rs:
            continue
        summary[model] = {k: [r[k] for r in rs] for k in ("test_accuracy", "test_macro_f1", "test_weighted_f1")}
        lines.append(
            f"| {DISPLAY_NAMES[model]} | {len(rs)} | {ms(summary[model]['test_accuracy'])} | "
            f"{ms(summary[model]['test_macro_f1'])} | {ms(summary[model]['test_weighted_f1'])} | "
            f"{ms([r['epochs_run'] for r in rs], False)} | {ms([r['mean_epoch_seconds'] for r in rs], False)} | "
            f"{ms([r['train_minutes'] for r in rs], False)} | {rs[0]['params'] / 1e6:.2f} |"
        )
    rand = [r for r in rows if r["split"] == "random"]
    if rand:
        lines += ["", "## Leakage check (EfficientNet-B0, seed 0)", "",
                  "| Split | Test accuracy | Test macro-F1 |", "| --- | ---: | ---: |"]
        grouped = [r for r in rows if r["model"] == "efficientnet_b0" and r["split"] == "grouped" and r["seed"] == 0]
        for label, rs in (("Grouped by near-duplicates", grouped), ("Random per image", rand)):
            if rs:
                lines.append(f"| {label} | {rs[0]['test_accuracy'] * 100:.2f} | {rs[0]['test_macro_f1'] * 100:.2f} |")
    lines += ["", "Per-run numbers: [results.csv](results.csv)."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
