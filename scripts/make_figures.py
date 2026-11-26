"""Report figures: validation curves, confusion matrices, per-class F1 comparison.

    python scripts/make_figures.py --runs runs --out reports/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cropdoc.models import DISPLAY_NAMES  # noqa: E402

COLORS = {"custom_cnn": "#2a6fdb", "efficientnet_b0": "#e07b00"}


def short(c: str) -> str:
    crop, _, dis = c.partition("___")
    crop = crop.split("_(")[0].split(",")[0]
    return f"{crop}: {dis.replace('_', ' ')}"[:34]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for f in sorted(Path(args.runs).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if not m["run_id"].startswith("smoke") and "random" not in m["run_id"]:
            metrics[m["run_id"]] = m

    # 1. validation macro-F1 per epoch, every seed
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for rid, m in metrics.items():
        h = m["history"]
        ax.plot([e["global_epoch"] + 1 for e in h], [e["val_macro_f1"] * 100 for e in h],
                color=COLORS[m["model"]], alpha=0.8, marker="o", ms=3,
                label=f"{DISPLAY_NAMES[m['model']]} (seed {m['seed']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1 (%)")
    ax.set_title("Validation macro-F1 during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "val_macro_f1.png", dpi=150)
    plt.close(fig)

    # 2. confusion matrices (row-normalised) and 3. per-class F1, for runs with saved files
    per_class = {}
    for rid, m in metrics.items():
        d = Path(args.runs) / rid
        if (d / "test_per_class.json").exists() and m["model"] not in per_class:
            per_class[m["model"]] = json.loads((d / "test_per_class.json").read_text())
        if (d / "confusion.npy").exists() and (d / "classes.json").exists():
            cm = np.load(d / "confusion.npy").astype(float)
            classes = json.loads((d / "classes.json").read_text())
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            fig, ax = plt.subplots(figsize=(11, 10))
            im = ax.imshow(cmn * 100, cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(len(classes)), [short(c) for c in classes], rotation=90, fontsize=6)
            ax.set_yticks(range(len(classes)), [short(c) for c in classes], fontsize=6)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{DISPLAY_NAMES[m['model']]} ({rid}): test confusion matrix, % of true class")
            fig.colorbar(im, ax=ax, fraction=0.03)
            fig.tight_layout()
            fig.savefig(out / f"confusion_{rid}.png", dpi=150)
            plt.close(fig)

    if len(per_class) == 2:
        classes = sorted(per_class["efficientnet_b0"])
        y = np.arange(len(classes))
        fig, ax = plt.subplots(figsize=(7, 11))
        for k, (model, off) in enumerate((("custom_cnn", -0.2), ("efficientnet_b0", 0.2))):
            ax.barh(y + off, [per_class[model][c]["f1"] * 100 for c in classes], height=0.4,
                    color=COLORS[model], label=DISPLAY_NAMES[model])
        ax.set_yticks(y, [short(c) for c in classes], fontsize=6.5)
        ax.invert_yaxis()
        lo = min(min(v["f1"] for v in pc.values()) for pc in per_class.values()) * 100
        ax.set_xlim(max(0, lo - 5), 100.5)
        ax.set_xlabel("Test F1 (%)")
        ax.set_title("Per-class test F1")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "per_class_f1.png", dpi=150)
        plt.close(fig)
    print("figures in", out)


if __name__ == "__main__":
    main()
