"""Class counts per split and the imbalance ratio, from the split CSV.

    python scripts/dataset_stats.py --split data/splits/pv_split.csv --out reports/dataset_stats.md
"""

import argparse
from collections import Counter

from cropdoc.data import class_names, read_split_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="data/splits/pv_split.csv")
    ap.add_argument("--out", default="reports/dataset_stats.md")
    args = ap.parse_args()

    rows = read_split_csv(args.split)
    classes = class_names(rows)
    counts = {s: Counter(r["label"] for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    total = Counter(r["label"] for r in rows)
    groups = len({r["group"] for r in rows})
    big, small = max(total.values()), min(total.values())

    lines = [
        "# PlantVillage dataset statistics",
        "",
        f"- Images: {len(rows)}",
        f"- Classes: {len(classes)}",
        f"- Near-duplicate groups: {groups}",
        f"- Split sizes: " + ", ".join(f"{s} {sum(c.values())}" for s, c in counts.items()),
        f"- Largest class: {total.most_common(1)[0][0]} ({big})",
        f"- Smallest class: {min(total, key=total.get)} ({small})",
        f"- Imbalance ratio (largest / smallest): {big / small:.1f} : 1",
        "",
        "| Class | Train | Val | Test | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for c in classes:
        lines.append(f"| {c} | {counts['train'][c]} | {counts['val'][c]} | {counts['test'][c]} | {total[c]} |")
    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
