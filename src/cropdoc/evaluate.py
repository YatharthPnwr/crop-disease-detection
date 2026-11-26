"""Prediction and classification metrics (accuracy, macro-F1, per-class, confusion matrix)."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def autocast(device: torch.device, precision: str = "bf16"):
    if device.type == "cuda" and precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and precision in ("fp16", "bf16"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, loader, device, precision: str = "bf16", criterion=None):
    """Returns (y_true, y_pred, probs, mean_loss)."""
    model.eval()
    ys, preds, probs, loss_sum, n = [], [], [], 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * len(y)
            n += len(y)
        p = logits.softmax(1)
        ys.append(y.cpu())
        preds.append(p.argmax(1).cpu())
        probs.append(p.cpu())
    return (
        torch.cat(ys).numpy(),
        torch.cat(preds).numpy(),
        torch.cat(probs).numpy(),
        loss_sum / max(n, 1),
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
