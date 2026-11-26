"""Build the leakage-safe grouped split for an image folder (root/<class>/*.jpg).

    python scripts/make_splits.py --root data/raw/color --out data/splits/pv_split.csv
"""

import argparse
import json

from cropdoc.splits import make_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="folder with one subfolder per class")
    ap.add_argument("--out", default="data/splits/pv_split.csv")
    ap.add_argument("--pairs", default=None, help="optional CSV of linked near-duplicate pairs")
    ap.add_argument("--max-dist", type=int, default=6, help="pHash Hamming threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    summary = make_split(args.root, args.out, args.max_dist, args.seed, args.workers, args.pairs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
