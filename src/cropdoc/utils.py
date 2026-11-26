"""Config loading, seeding and small I/O helpers."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(name_or_path: str, config_dir: str | Path | None = None) -> dict:
    """Load configs/base.yaml merged with configs/<name>.yaml (or an explicit path)."""
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    path = Path(name_or_path)
    if not path.suffix:
        path = config_dir / f"{name_or_path}.yaml"
    base = yaml.safe_load((config_dir / "base.yaml").read_text())
    cfg = deep_merge(base, yaml.safe_load(path.read_text()))
    cfg["config_name"] = path.stem
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit() -> str:
    if os.environ.get("CROPDOC_GIT_COMMIT"):
        return os.environ["CROPDOC_GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
