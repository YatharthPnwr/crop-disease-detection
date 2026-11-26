"""Training loop: phases (head-only -> full fine-tune), bf16, checkpoints, early stopping.

Run on Modal through modal_app.py, or locally for a quick CPU sanity check:

    python -m cropdoc.train --config custom --data-root data/raw/color \
        --split data/splits/pv_split.csv --out runs/custom_s0 --fraction 0.01 --max-epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (BackgroundSwap, LeafDataset, class_names, domain_of, eval_transform, read_split_csv,
                   split_rows, train_transform)
from .evaluate import autocast, classification_metrics, confusion, domain_metrics, predict
from .models import build_model, count_params, freeze_backbone_bn, param_groups, set_backbone_trainable
from .utils import get_device, git_commit, load_config, load_json, save_json, seed_everything, seed_worker


def lr_lambda(total_steps: int, warmup_steps: int, schedule: str = "cosine"):
    """Linear warmup, then cosine decay to 0 (or constant)."""

    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f


def make_loaders(cfg: dict, data_root, rows, classes, seed: int, fraction: float, device):
    d = cfg["data"]
    size, resize = d["image_size"], d["resize"]
    pre = None
    if d.get("bg_swap", 0) > 0:
        pre = BackgroundSwap(d["seg_root"], d["bg_dir"], p=d["bg_swap"])
    sets = {
        "train": LeafDataset(data_root, split_rows(rows, "train", fraction, seed), classes, train_transform(size), pre),
        "val": LeafDataset(data_root, split_rows(rows, "val", fraction, seed), classes, eval_transform(size, resize)),
        "test": LeafDataset(data_root, split_rows(rows, "test", fraction, seed), classes, eval_transform(size, resize)),
    }
    workers = d["num_workers"]
    common = dict(
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    g = torch.Generator().manual_seed(seed)
    bs = d["batch_size"]
    sampler = None
    if d.get("domain_weights"):  # oversample the small field domain
        w = torch.tensor([float(d["domain_weights"].get(dom, 1.0)) for dom in sets["train"].domains], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=g)
    return {
        "train": DataLoader(sets["train"], batch_size=bs, shuffle=sampler is None, sampler=sampler,
                            drop_last=len(sets["train"]) > bs, generator=g if sampler is None else None, **common),
        "val": DataLoader(sets["val"], batch_size=bs * 2, shuffle=False, **common),
        "test": DataLoader(sets["test"], batch_size=bs * 2, shuffle=False, **common),
    }


def train_one_epoch(model, loader, criterion, opt, sched, scaler, device, precision, freeze_bn):
    model.train()
    if freeze_bn:
        freeze_backbone_bn(model)  # model.train() re-enables BN stats; undo it for the frozen backbone
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(x)
        loss = criterion(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def _cpu_state(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def run(
    cfg: dict,
    data_root: str | Path,
    split_csv: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    fraction: float = 1.0,
    max_epochs: int | None = None,
    resume: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if resume and (out / "metrics.json").exists():
        print(f"[{out.name}] already finished, skipping", flush=True)
        return load_json(out / "metrics.json")

    seed_everything(seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    precision = cfg["train"]["precision"] if device.type == "cuda" else "fp32"

    rows = read_split_csv(split_csv)
    classes = class_names(rows)
    field_idx = sorted({classes.index(r["label"]) for r in rows if domain_of(r) == "plantdoc"}) or None
    select_domains = cfg["train"].get("select_metric") == "domain_mean_f1"
    loaders = make_loaders(cfg, data_root, rows, classes, seed, fraction, device)
    print(f"[{out.name}] device={device} train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}", flush=True)

    model = build_model(cfg["model"], len(classes), cfg["dropout"], pretrained=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    use_scaler = device.type == "cuda" and precision != "fp32" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(out / "tb")
    except Exception:  # tensorboard missing: keep training
        writer = None

    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False) if resume and (out / "last.pt").exists() else None
    history = state["history"] if state else []
    best_f1 = state["best_f1"] if state else -1.0
    global_epoch = state["global_epoch"] if state else 0
    patience = cfg["train"]["patience"]
    phases = cfg["phases"]
    t_start = time.time()

    for pi, ph in enumerate(phases):
        if state and pi < state["phase"]:
            continue
        epochs = ph["epochs"] if max_epochs is None else min(ph["epochs"], max_epochs)
        freeze = ph.get("freeze_backbone", False)
        set_backbone_trainable(model, not freeze)
        opt = torch.optim.AdamW(param_groups(model, ph["head_lr"], ph["backbone_lr"], cfg["train"]["weight_decay"]))
        steps = epochs * len(loaders["train"])
        warm = ph.get("warmup_epochs", 0) * len(loaders["train"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, warm, ph.get("schedule", "cosine")))
        start_epoch, bad = 0, 0
        if state and pi == state["phase"]:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            start_epoch, bad = state["epoch"] + 1, state["bad_epochs"]
            print(f"[{out.name}] resumed at phase {ph['name']} epoch {start_epoch}", flush=True)
            state = None
        last_phase = pi == len(phases) - 1

        for ep in range(start_epoch, epochs):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, loaders["train"], criterion, opt, sched, scaler, device, precision, freeze
            )
            yv, pv, vprobs, val_loss = predict(model, loaders["val"], device, precision, criterion)
            vm = classification_metrics(yv, pv, classes)
            dm = domain_metrics(yv, vprobs, loaders["val"].dataset.domains, field_idx)
            score = vm["macro_f1"]
            if select_domains and len(dm) > 1:  # equal weight to lab and field validation
                score = float(np.mean([v["macro_f1"] for v in dm.values()]))
            rec = {
                "phase": ph["name"],
                "epoch": ep,
                "global_epoch": global_epoch,
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(vm["accuracy"], 5),
                "val_macro_f1": round(vm["macro_f1"], 5),
                **({f"val_f1_{k}": round(v["macro_f1"], 5) for k, v in dm.items()} if len(dm) > 1 else {}),
                "val_select": round(score, 5),
                "head_lr": opt.param_groups[-1]["lr"],
                "seconds": round(time.time() - t0, 1),
            }
            history.append(rec)
            print(f"[{out.name}] " + json.dumps(rec), flush=True)
            if writer:
                for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "head_lr"):
                    writer.add_scalar(k, rec[k], global_epoch)

            improved = score > best_f1
            if improved:
                best_f1 = score
                torch.save(
                    {"model": _cpu_state(model), "cfg": cfg, "classes": classes, "seed": seed, "val": rec},
                    out / "best.pt",
                )
            bad = 0 if improved else bad + 1
            global_epoch += 1
            torch.save(
                {
                    "model": _cpu_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "phase": pi, "epoch": ep, "bad_epochs": bad, "best_f1": best_f1,
                    "global_epoch": global_epoch, "history": history,
                },
                out / "last.pt",
            )
            if last_phase and bad >= patience:
                print(f"[{out.name}] early stop: no val macro-F1 gain for {patience} epochs", flush=True)
                break

    if writer:
        writer.close()

    # ---- test the best checkpoint
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    yt, pt, tprobs, test_loss = predict(model, loaders["test"], device, precision, criterion)
    tm = classification_metrics(yt, pt, classes)
    test_domains = domain_metrics(yt, tprobs, loaders["test"].dataset.domains, field_idx)
    tdom = np.array(loaders["test"].dataset.domains)
    for dname in set(tdom.tolist()) if len(set(tdom.tolist())) > 1 else []:
        sel = tdom == dname
        save_json(classification_metrics(yt[sel], pt[sel], classes)["per_class"], out / f"test_per_class_{dname}.json")
        np.save(out / f"confusion_{dname}.npy", confusion(yt[sel], pt[sel], len(classes)))
    np.save(out / "confusion.npy", confusion(yt, pt, len(classes)))

    # App "unsure" threshold from validation (never test): the top-1 confidence that
    # 95% of correct validation predictions exceed.
    yv, pv, val_probs, _ = predict(model, loaders["val"], device, precision, criterion)
    np.savez_compressed(out / "val_probs.npz", y=yv, probs=val_probs.astype(np.float16))
    conf_correct = val_probs.max(1)[yv == pv]
    unsure_threshold = float(np.percentile(conf_correct, 5)) if len(conf_correct) else 0.5
    save_json(classes, out / "classes.json")
    save_json(tm["per_class"], out / "test_per_class.json")

    epoch_secs = [h["seconds"] for h in history]
    metrics = {
        "run_id": out.name,
        "config": cfg.get("config_name", cfg["model"]),
        "model": cfg["model"],
        "seed": seed,
        "fraction": fraction,
        "params": count_params(model),
        "git_commit": git_commit(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "best_epoch": best["val"],
        "test": {k: v for k, v in tm.items() if k != "per_class"} | {"loss": test_loss},
        "test_domains": test_domains,
        "unsure_threshold": round(unsure_threshold, 4),
        "epochs_run": len(history),
        "mean_epoch_seconds": round(float(np.mean(epoch_secs)), 1) if epoch_secs else None,
        "train_minutes": round((time.time() - t_start) / 60, 2),
        "history": history,
    }
    save_json(metrics, out / "metrics.json")
    print(f"[{out.name}] test acc={tm['accuracy']:.4f} macro_f1={tm['macro_f1']:.4f}", flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config name in configs/ (custom, effb0) or a path")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    run(cfg, args.data_root, args.split, args.out, args.seed, args.fraction, args.max_epochs)


if __name__ == "__main__":
    main()
