"""Download trained app models from the repo's GitHub release (needs `gh auth login`).

    python scripts/fetch_models.py            # latest models-* release
    python scripts/fetch_models.py --tag models-v1
"""

import argparse
import json
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dir", default="app/models")
    args = ap.parse_args()
    tag = args.tag
    if tag is None:
        releases = json.loads(subprocess.check_output(["gh", "release", "list", "--json", "tagName"], text=True))
        tags = [r["tagName"] for r in releases if r["tagName"].startswith("models-")]
        if not tags:
            raise SystemExit("no models-* release found")
        tag = tags[0]
    subprocess.run(["gh", "release", "download", tag, "--dir", args.dir, "--clobber", "--pattern", "*"], check=True)
    print(f"downloaded {tag} into {args.dir}")


if __name__ == "__main__":
    main()
