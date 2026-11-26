"""Modal app: dataset download, grouped split, training runs, result collection.

The laptop only orchestrates; every heavy step runs in a Modal container.

    modal run modal_app.py::download        # PlantVillage (color + segmented) -> volume cropdoc-data
    modal run modal_app.py::make_splits     # pHash-grouped split (+ random split for the leakage check)
    modal run modal_app.py::smoke           # 1 epoch on 5% of the data, both models: timing + sanity
    modal run --detach modal_app.py::grid   # custom CNN x3 seeds, EfficientNet-B0 x3 seeds (+ leakage run)
    modal run modal_app.py::collect         # metrics -> runs/, best weights -> app/models/
    modal run modal_app.py::explain         # Grad-CAM leaf-vs-background analysis + CAM grids
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
def build_splits(max_dist: int = 4, seed: int = 42) -> dict:
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


PLANTDOC_ZIP = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/archive/refs/heads/master.zip"


@app.function(image=train_image, volumes={DATA: data_vol}, cpu=8, memory=16384, timeout=3600)
def build_field_data(max_dist: int = 4, seed: int = 42, n_backgrounds: int = 600) -> dict:
    """PlantDoc leaf crops + field backgrounds into the volume, and the combined lab+field split."""
    import random
    import tarfile
    import urllib.request
    import zipfile

    from PIL import Image

    from cropdoc.data import read_split_csv
    from cropdoc.plantdoc import build_plantdoc, write_combined_split

    zpath, raw = Path("/tmp/plantdoc.zip"), Path("/tmp/pdraw")
    urllib.request.urlretrieve(PLANTDOC_ZIP, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(raw)
    det = next(p for p in raw.iterdir() if p.is_dir())
    color = ensure_local("color")
    pv_rows = read_split_csv(f"{DATA}/pv_split.csv")
    crops = Path(LOCAL_PV) / "plantdoc"
    rows, summary = build_plantdoc(det, crops, color, pv_rows, max_dist, seed, workers=8)

    # field backgrounds for background swap: PlantDoc TRAIN photos only (never test)
    bg = Path(LOCAL_PV) / "backgrounds"
    bg.mkdir(parents=True, exist_ok=True)
    train_imgs = sorted(p for p in (det / "TRAIN").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i, src in enumerate(random.Random(seed).sample(train_imgs, min(n_backgrounds, len(train_imgs)))):
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((480, 480))
                im.save(bg / f"bg{i:04d}.jpg", quality=88)
        except OSError:
            continue
    for kind in ("plantdoc", "backgrounds"):
        with tarfile.open(f"{DATA}/pv_{kind}.tar", "w") as tar:
            tar.add(Path(LOCAL_PV) / kind, arcname=kind)

    pd_csv, comb_csv = f"{DATA}/pd_split.csv", f"{DATA}/combined_split.csv"
    with open(pd_csv, "w", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=["path", "label", "group", "split", "source_split", "source_image"])
        w.writeheader()
        w.writerows(rows)
    write_combined_split(pv_rows, rows, comb_csv)
    data_vol.commit()
    summary["backgrounds"] = len(list(bg.glob("*.jpg")))
    return {"summary": summary, "pd_split.csv": Path(pd_csv).read_bytes()}


@app.function(image=train_image, gpu="A10G", cpu=8, memory=32768, timeout=3600,
              volumes={DATA: data_vol, RUNS: runs_vol})
def evaluate_runs(run_ids: list[str], split_name: str = "combined_split.csv") -> dict:
    """Lab and field test metrics for existing checkpoints (e.g. the lab-only baseline)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from cropdoc.data import LeafDataset, domain_of, eval_transform, read_split_csv, split_rows
    from cropdoc.evaluate import classification_metrics, domain_metrics, predict
    from cropdoc.models import build_model

    runs_vol.reload()
    ensure_local("color")
    ensure_local("plantdoc")
    rows = read_split_csv(f"{DATA}/{split_name}")
    test = split_rows(rows, "test")
    out = {}
    for rid in run_ids:
        ck = torch.load(f"{RUNS}/{rid}/best.pt", map_location="cpu", weights_only=False)
        classes = ck["classes"]
        model = build_model(ck["cfg"]["model"], len(classes), ck["cfg"]["dropout"], pretrained=False)
        model.load_state_dict(ck["model"])
        model = model.cuda().eval().to(memory_format=torch.channels_last)
        ds = LeafDataset(LOCAL_PV, test, classes, eval_transform())
        y, p, probs, _ = predict(model, DataLoader(ds, batch_size=256, num_workers=8), torch.device("cuda"), "bf16")
        field_idx = sorted({classes.index(r["label"]) for r in rows if domain_of(r) == "plantdoc"})
        dom = np.array(ds.domains)
        pd_sel = dom == "plantdoc"
        out[rid] = {
            "test_domains": domain_metrics(y, probs, ds.domains, field_idx),
            "per_class_plantdoc": classification_metrics(y[pd_sel], p[pd_sel], classes)["per_class"],
        }
        print(rid, json.dumps(out[rid]["test_domains"]), flush=True)
    return out


@app.function(image=train_image, volumes={DATA: data_vol}, cpu=4, memory=8192, timeout=1800)
def pair_sheet(pairs: list[tuple[str, str, int]], title: str) -> bytes:
    """Contact sheet of linked near-duplicate pairs (for checking the pHash threshold by eye)."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    root = Path(ensure_local("color"))
    cols = 6
    rows = -(-len(pairs) // (cols // 2))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1), squeeze=False)
    for k, (a, b, d) in enumerate(pairs):
        r, c = divmod(k, cols // 2)
        for j, p in enumerate((a, b)):
            ax = axes[r, 2 * c + j]
            ax.imshow(Image.open(root / p).convert("RGB"))
            ax.set_title(f"{p.split('/')[0][:18]}\nd={d}" if j == 0 else "", fontsize=6)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


@app.function(
    image=train_image,
    gpu="A10G",
    cpu=16,  # PIL augmentation is the bottleneck, not the GPU
    memory=32768,
    timeout=4 * 3600,
    volumes={DATA: data_vol, RUNS: runs_vol},
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
)
def train(cfg: dict, seed: int, run_id: str, fraction: float = 1.0, max_epochs: int | None = None,
          commit: str = "unknown", split_name: str = "pv_split.csv") -> dict:
    os.environ["CROPDOC_GIT_COMMIT"] = commit
    from cropdoc.train import run

    split_name = cfg.get("split", split_name)
    if split_name.startswith("combined"):  # lab + field: paths are color/... and plantdoc/...
        ensure_local("color")
        ensure_local("plantdoc")
        root = LOCAL_PV
        if cfg["data"].get("bg_swap", 0) > 0:
            cfg["data"]["seg_root"] = ensure_local("segmented")
            cfg["data"]["bg_dir"] = ensure_local("backgrounds")
    else:
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


@app.function(image=train_image, gpu="A10G", cpu=8, memory=32768, timeout=3600,
              volumes={DATA: data_vol, RUNS: runs_vol})
def explain_runs(run_ids: list[str], n: int = 500, seed: int = 0) -> dict:
    """Leaf-vs-background analysis + CAM grids (8 correct, 8 wrong test images) per run."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from cropdoc.data import LeafDataset, eval_transform, read_split_csv, split_rows
    from cropdoc.evaluate import predict
    from cropdoc.explain import analyze, cam_grid_png
    from cropdoc.models import DISPLAY_NAMES, build_model

    runs_vol.reload()
    color = ensure_local("color")
    seg = ensure_local("segmented") if Path(f"{DATA}/pv_segmented.tar").exists() else None
    test = split_rows(read_split_csv(f"{DATA}/pv_split.csv"), "test")
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    out = {}
    for rid in run_ids:
        ck = torch.load(f"{RUNS}/{rid}/best.pt", map_location="cpu", weights_only=False)
        classes = ck["classes"]
        model = build_model(ck["cfg"]["model"], len(classes), ck["cfg"]["dropout"], pretrained=False)
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        res = analyze(model, color, seg, test, classes, device, n=n, seed=seed)

        # full test pass to find enough correct and wrong examples for the grids
        loader = DataLoader(LeafDataset(color, test, classes, eval_transform()), batch_size=256, num_workers=8)
        y, p, _, _ = predict(model, loader, device, "fp32")
        cls_index = {c: i for i, c in enumerate(classes)}
        items = [{"path": r["path"], "label": cls_index[r["label"]], "pred": int(pi)} for r, pi in zip(test, p)]
        right = [items[i] for i in rng.permutation(np.flatnonzero(y == p))[:8]]
        wrong = [items[i] for i in rng.permutation(np.flatnonzero(y != p))[:8]]
        name = DISPLAY_NAMES[ck["cfg"]["model"]]
        out[rid] = {
            "summary": res["summary"] | {"test_errors": int((y != p).sum()), "test_images": int(len(y))},
            "per_image": res["per_image"],
            "correct.png": cam_grid_png(model, color, right, classes, device, f"{name}: correct predictions"),
            "wrong.png": cam_grid_png(model, color, wrong, classes, device, f"{name}: wrong predictions") if wrong else None,
        }
    return out


@app.function(image=train_image, volumes={RUNS: runs_vol}, timeout=1800)
def collect_runs(weights_for: list[str]) -> dict:
    runs_vol.reload()
    out = {}
    for d in sorted(p for p in Path(RUNS).iterdir() if p.is_dir()):
        files = {}
        for name in ("metrics.json", "test_per_class.json", "classes.json", "confusion.npy",
                     "test_per_class_plantdoc.json", "test_per_class_plantvillage.json", "confusion_plantdoc.npy"):
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
def make_splits(max_dist: int = 4, seed: int = 42):
    res = build_splits.remote(max_dist, seed)
    out = ROOT / "data" / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("pv_split.csv", "pv_split_random.csv", "pv_pairs.csv"):
        (out / name).write_bytes(res[name])
    (ROOT / "reports" / "split_summary.json").write_text(json.dumps(res["summary"], indent=2))
    print(json.dumps(res["summary"], indent=2))


@app.local_entrypoint()
def inspect_pairs(dist: int = 4, n: int = 15, seed: int = 0):
    """Random linked pairs at exactly `dist` -> reports/figures/pairs_d<dist>.png."""
    import csv
    import random

    with open(ROOT / "data" / "splits" / "pv_pairs.csv") as f:
        pairs = [(r["path_a"], r["path_b"], int(r["hamming"])) for r in csv.DictReader(f) if int(r["hamming"]) == dist]
    pick = random.Random(seed).sample(pairs, min(n, len(pairs)))
    png = pair_sheet.remote(pick, f"Random linked pairs at pHash Hamming distance {dist} ({len(pairs)} such pairs)")
    out = ROOT / "reports" / "figures" / f"pairs_d{dist}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    print(f"wrote {out}")


@app.function(image=train_image, volumes={DATA: data_vol}, cpu=2, timeout=900)
def read_images(paths: list[str]) -> dict[str, bytes]:
    root = Path(ensure_local("color"))
    return {p: (root / p).read_bytes() for p in paths}


@app.local_entrypoint()
def samples(seed: int = 0):
    """One test-split image per demo class -> app/samples/ (for the demo and an end-to-end check)."""
    import csv
    import random

    demo = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy", "Potato___Late_blight",
            "Apple___Apple_scab", "Corn_(maize)___Common_rust_", "Grape___Black_rot", "Pepper,_bell___Bacterial_spot"]
    with open(ROOT / "data" / "splits" / "pv_split.csv") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rng = random.Random(seed)
    picks = [rng.choice([r for r in test if r["label"] == c])["path"] for c in demo]
    blobs = read_images.remote(picks)
    out = ROOT / "app" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Sample leaf images", "",
             "One image per class from the PlantVillage **test** split (never seen in training).",
             "Source: PlantVillage via Kaggle abdallahalidev/plantvillage-dataset, CC BY-NC-SA 4.0.", "",
             "| File | True class |", "| --- | --- |"]
    for c, p in zip(demo, picks):
        name = c.split("___")[0].split("_(")[0].split(",")[0].lower() + "_" + c.split("___")[1].strip("_").lower() + ".jpg"
        (out / name).write_bytes(blobs[p])
        lines.append(f"| {name} | {c} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


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
def field_data():
    """PlantDoc crops + backgrounds -> volume; writes data/splits/pd_split.csv and reports/plantdoc_summary.json."""
    res = build_field_data.remote()
    (ROOT / "data" / "splits" / "pd_split.csv").write_bytes(res["pd_split.csv"])
    (ROOT / "reports" / "plantdoc_summary.json").write_text(json.dumps(res["summary"], indent=2))
    print(json.dumps({k: v for k, v in res["summary"].items() if k != "per_class"}, indent=2))


@app.local_entrypoint()
def field_baseline(runs: str = "custom_s0,custom_s1,custom_s2,effb0_s0,effb0_s1,effb0_s2"):
    """Lab-only models scored on the lab and field test sets -> reports/field_baseline.json."""
    res = evaluate_runs.remote([r for r in runs.split(",") if r])
    (ROOT / "reports" / "field_baseline.json").write_text(json.dumps(res, indent=2))


@app.local_entrypoint()
def grid_field(seeds: str = "0,1,2", ablation: bool = True):
    """Lab + field training: custom CNN and EfficientNet-B0 x seeds (+ EfficientNet-B0 without background swap)."""
    commit = _git_commit()
    jobs = []
    for c in ("custom_field", "effb0_field"):
        cfg = _cfg(c)
        for s in (int(x) for x in seeds.split(",")):
            jobs.append({"cfg": cfg, "seed": s, "run_id": f"{c}_s{s}", "commit": commit, "split": cfg["split"]})
    if ablation:
        cfg = _cfg("effb0_field_nobg")
        jobs.append({"cfg": cfg, "seed": 0, "run_id": "effb0_field_nobg_s0", "commit": commit, "split": cfg["split"]})
    print(f"launching {len(jobs)} runs: {[j['run_id'] for j in jobs]}")
    for r in run_grid.remote(jobs):
        res = r["result"]
        print(r["run_id"], "ok" if r["ok"] else "FAILED", json.dumps(res.get("test_domains")) if r["ok"] else res)


@app.local_entrypoint()
def explain(runs: str = "custom_s0,effb0_s0", n: int = 500):
    """Grad-CAM analysis -> reports/explain_<run>.json and reports/figures/cam_<run>_*.png."""
    res = explain_runs.remote([r for r in runs.split(",") if r], n)
    figs = ROOT / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for rid, r in res.items():
        (ROOT / "reports" / f"explain_{rid}.json").write_text(
            json.dumps({"summary": r["summary"], "per_image": r["per_image"]}, indent=2))
        for kind in ("correct", "wrong"):
            if r[f"{kind}.png"]:
                (figs / f"cam_{rid}_{kind}.png").write_bytes(r[f"{kind}.png"])
        print(rid, json.dumps(r["summary"], indent=2))


@app.local_entrypoint()
def collect(family: str = "field"):
    """Metrics for every run -> runs/<id>/; best-val seed of each model in `family`
    ("lab": PlantVillage-only runs, "field": lab + field runs) -> app/models/<model>.pt."""
    meta = collect_runs.remote([])
    best: dict[str, tuple[float, str]] = {}
    for run_id, files in meta.items():
        if "metrics.json" not in files or run_id.startswith("smoke") or "random" in run_id or "nobg" in run_id:
            continue
        if ("_field_" in run_id) != (family == "field"):
            continue
        m = json.loads(files["metrics.json"])
        f1 = m["best_epoch"].get("val_select", m["best_epoch"]["val_macro_f1"])
        if f1 > best.get(m["model"], (-1.0, ""))[0]:
            best[m["model"]] = (f1, run_id)
    weights = collect_runs.remote([rid for _, rid in best.values()])
    for run_id, files in weights.items():
        d = ROOT / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        for name, blob in files.items():
            (d / name).write_bytes(blob)
    models_dir = ROOT / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for model, (f1, run_id) in best.items():
        src = ROOT / "runs" / run_id
        (models_dir / f"{model}.pt").write_bytes((src / "best.pt").read_bytes())
        m = json.loads((src / "metrics.json").read_text())
        chosen[model] = {"run_id": run_id, "val_macro_f1": f1, "unsure_threshold": m.get("unsure_threshold", 0.5)}
        print(f"{model}: {run_id} (val macro-F1 {f1:.4f}) -> app/models/{model}.pt")
    (models_dir / "models.json").write_text(json.dumps(chosen, indent=2))
