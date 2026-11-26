"""Modal app: dataset download, grouped split, training runs, result collection.

The laptop only orchestrates; every heavy step runs in a Modal container.

    modal run modal_app.py::download        # PlantVillage (color + segmented) -> volume cropdoc-data
    modal run modal_app.py::make_splits     # pHash-grouped split (+ random split for the leakage check)
    modal run modal_app.py::smoke           # 1 epoch on 5% of the data, both models: timing + sanity
    modal run --detach modal_app.py::grid   # custom CNN x3 seeds, EfficientNet-B0 x3 seeds (+ leakage run)
    modal run modal_app.py::collect         # metrics -> runs/, best weights -> app/models/
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("crop-disease-detection")
data_vol = modal.Volume.from_name("cropdoc-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("cropdoc-runs", create_if_missing=True)
DATA, RUNS, LOCAL_PV = "/data", "/runs", "/tmp/pv"
KAGGLE_SLUG = "abdallahalidev/plantvillage-dataset"

kaggle_image = modal.Image.debian_slim(python_version="3.11").pip_install("kaggle", "six")
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "numpy>=2.0",
        "pillow",
        "scikit-learn",
        "imagehash",
        "pyyaml",
        "tensorboard",
        "grad-cam",
        "pandas",
        "matplotlib",
    )
    .add_local_python_source("cropdoc")
)


# ----------------------------------------------------------------- remote side


def ensure_local(kind: str = "color") -> str:
    """Extract /data/pv_<kind>.tar to local SSD once per container (Volume reads of many small files are slow)."""
    import tarfile

    target = Path(LOCAL_PV) / kind
    if not target.exists():
        Path(LOCAL_PV).mkdir(parents=True, exist_ok=True)
        with tarfile.open(f"{DATA}/pv_{kind}.tar") as tar:
            tar.extractall(LOCAL_PV, filter="data")
    return str(target)


@app.function(
    image=kaggle_image,
    volumes={DATA: data_vol},
    secrets=[modal.Secret.from_name("kaggle")],
    cpu=4,
    memory=8192,
    timeout=3600,
)
def download_dataset(slug: str = KAGGLE_SLUG) -> dict:
    import tarfile
    import zipfile

    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    tok = kdir / "access_token"
    tok.write_text(os.environ["KAGGLE_API_TOKEN"].strip())
    tok.chmod(0o600)

    dl = Path("/tmp/dl")
    dl.mkdir(exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dl)]
    if subprocess.run(cmd).returncode != 0:  # newer CLIs take the slug positionally
        subprocess.run(["kaggle", "datasets", "download", slug, "-p", str(dl)], check=True)
    for z in dl.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dl / "x")
        z.unlink()

    found: dict[str, Path] = {}
    for d in sorted((dl / "x").rglob("*")):
        if d.is_dir() and d.name.lower() in ("color", "segmented"):
            if sum(1 for c in d.iterdir() if c.is_dir()) >= 30:
                found.setdefault(d.name.lower(), d)
    if "color" not in found:
        raise RuntimeError(f"no color/ folder with class subfolders under {dl / 'x'}")

    manifest = {"slug": slug}
    for kind, d in found.items():
        classes = sorted(c.name for c in d.iterdir() if c.is_dir())
        n = sum(1 for c in d.iterdir() if c.is_dir() for _ in c.iterdir())
        with tarfile.open(f"{DATA}/pv_{kind}.tar", "w") as tar:
            tar.add(d, arcname=kind)
        manifest[kind] = {"images": n, "classes": len(classes), "class_names": classes,
                          "source_dir": str(d.relative_to(dl / "x"))}
    Path(f"{DATA}/manifest.json").write_text(json.dumps(manifest, indent=2))
    data_vol.commit()
    return manifest


@app.function(image=train_image, volumes={DATA: data_vol}, cpu=8, memory=16384, timeout=3600)
def build_splits(max_dist: int = 6, seed: int = 42) -> dict:
    import csv

    import numpy as np
    from sklearn.model_selection import train_test_split

    from cropdoc.splits import make_split

    root = ensure_local("color")
    split_csv, pairs_csv = f"{DATA}/pv_split.csv", f"{DATA}/pv_pairs.csv"
    summary = make_split(root, split_csv, max_dist, seed, workers=8, pairs_csv=pairs_csv)

    # Random per-image split of the same images, only for the leakage check run.
    with open(split_csv) as f:
        rows = list(csv.DictReader(f))
    y = np.array([r["label"] for r in rows])
    idx = np.arange(len(rows))
    trval, test = train_test_split(idx, test_size=1 / 7, stratify=y, random_state=seed)
    train, val = train_test_split(trval, test_size=1 / 6, stratify=y[trval], random_state=seed)
    split = np.empty(len(rows), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    rand_csv = f"{DATA}/pv_split_random.csv"
    with open(rand_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "split"])
        w.writerows((r["path"], r["label"], r["group"], s) for r, s in zip(rows, split))

    data_vol.commit()
    return {
        "summary": summary,
        "pv_split.csv": Path(split_csv).read_bytes(),
        "pv_split_random.csv": Path(rand_csv).read_bytes(),
        "pv_pairs.csv": Path(pairs_csv).read_bytes(),
    }


@app.function(
    image=train_image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=4 * 3600,
    volumes={DATA: data_vol, RUNS: runs_vol},
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
)
def train(cfg: dict, seed: int, run_id: str, fraction: float = 1.0, max_epochs: int | None = None,
          commit: str = "unknown", split_name: str = "pv_split.csv") -> dict:
    os.environ["CROPDOC_GIT_COMMIT"] = commit
    from cropdoc.train import run

    root = ensure_local("color")
    try:
        runs_vol.reload()  # pick up last.pt from a pre-empted attempt
    except Exception as e:  # reload can fail in a reused container with open files
        print(f"volume reload skipped: {e}")
    m = run(cfg, root, f"{DATA}/{split_name}", Path(RUNS) / run_id, seed, fraction, max_epochs)
    runs_vol.commit()
    return {k: v for k, v in m.items() if k != "history"}


@app.function(image=train_image, volumes={RUNS: runs_vol}, timeout=12 * 3600)
def run_grid(jobs: list[dict]) -> list[dict]:
    """Remote orchestrator: survives the laptop disconnecting when launched with --detach."""
    args = [(j["cfg"], j["seed"], j["run_id"], 1.0, None, j["commit"], j["split"]) for j in jobs]
    out = []
    for job, res in zip(jobs, train.starmap(args, return_exceptions=True)):
        ok = not isinstance(res, BaseException)
        out.append({"run_id": job["run_id"], "ok": ok, "result": res if ok else repr(res)})
        print(json.dumps(out[-1], default=str), flush=True)
    runs_vol.reload()
    Path(f"{RUNS}/grid_summary.json").write_text(json.dumps(out, indent=2, default=str))
    runs_vol.commit()
    return out


@app.function(image=train_image, volumes={RUNS: runs_vol}, timeout=1800)
def collect_runs(weights_for: list[str]) -> dict:
    runs_vol.reload()
    out = {}
    for d in sorted(p for p in Path(RUNS).iterdir() if p.is_dir()):
        files = {}
        for name in ("metrics.json", "test_per_class.json", "classes.json", "confusion.npy"):
            if (d / name).exists():
                files[name] = (d / name).read_bytes()
        if d.name in weights_for and (d / "best.pt").exists():
            files["best.pt"] = (d / "best.pt").read_bytes()
        out[d.name] = files
    return out


# ----------------------------------------------------------------- laptop side

ROOT = Path(__file__).resolve().parent
MODELS = {"custom": "custom_cnn", "effb0": "efficientnet_b0"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _cfg(name: str) -> dict:
    from cropdoc.utils import load_config

    return load_config(name, ROOT / "configs")


@app.local_entrypoint()
def download():
    manifest = download_dataset.remote()
    for kind in ("color", "segmented"):
        if kind in manifest:
            m = manifest[kind]
            print(f"{kind}: {m['images']} images, {m['classes']} classes ({m['source_dir']})")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))


@app.local_entrypoint()
def make_splits(max_dist: int = 6, seed: int = 42):
    res = build_splits.remote(max_dist, seed)
    out = ROOT / "data" / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("pv_split.csv", "pv_split_random.csv", "pv_pairs.csv"):
        (out / name).write_bytes(res[name])
    (ROOT / "reports" / "split_summary.json").write_text(json.dumps(res["summary"], indent=2))
    print(json.dumps(res["summary"], indent=2))


@app.local_entrypoint()
def smoke(fraction: float = 0.05):
    commit = _git_commit()
    args = [(_cfg(c), 0, f"smoke_{c}", fraction, 1, commit, "pv_split.csv") for c in MODELS]
    for res in train.starmap(args):
        print(json.dumps({k: res[k] for k in ("run_id", "test", "mean_epoch_seconds", "train_minutes")}, indent=2))


@app.local_entrypoint()
def grid(seeds: str = "0,1,2", leakage: bool = True):
    commit = _git_commit()
    jobs = []
    for c in MODELS:
        cfg = _cfg(c)
        for s in (int(x) for x in seeds.split(",")):
            jobs.append({"cfg": cfg, "seed": s, "run_id": f"{c}_s{s}", "commit": commit, "split": "pv_split.csv"})
    if leakage:
        jobs.append({"cfg": _cfg("effb0"), "seed": 0, "run_id": "effb0_randomsplit_s0",
                     "commit": commit, "split": "pv_split_random.csv"})
    print(f"launching {len(jobs)} runs: {[j['run_id'] for j in jobs]}")
    for r in run_grid.remote(jobs):
        print(r["run_id"], "ok" if r["ok"] else "FAILED", r["result"] if not r["ok"] else r["result"]["test"])


@app.local_entrypoint()
def collect():
    """Metrics for every run -> runs/<id>/; best-val seed of each model -> app/models/<model>.pt."""
    meta = collect_runs.remote([])
    best: dict[str, tuple[float, str]] = {}
    for run_id, files in meta.items():
        if "metrics.json" not in files or run_id.startswith("smoke") or "random" in run_id:
            continue
        m = json.loads(files["metrics.json"])
        f1 = m["best_epoch"]["val_macro_f1"]
        if f1 > best.get(m["model"], (-1.0, ""))[0]:
            best[m["model"]] = (f1, run_id)
    weights = collect_runs.remote([rid for _, rid in best.values()])
    for run_id, files in weights.items():
        d = ROOT / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        for name, blob in files.items():
            (d / name).write_bytes(blob)
    (ROOT / "app" / "models").mkdir(parents=True, exist_ok=True)
    for model, (f1, run_id) in best.items():
        src = ROOT / "runs" / run_id / "best.pt"
        (ROOT / "app" / "models" / f"{model}.pt").write_bytes(src.read_bytes())
        print(f"{model}: {run_id} (val macro-F1 {f1:.4f}) -> app/models/{model}.pt")
