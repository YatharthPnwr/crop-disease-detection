"""Checkpoint loading, ONNX export and parity check.

    python -m cropdoc.export --checkpoint app/models/efficientnet_b0.pt --out app/models/efficientnet_b0.onnx
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from .models import build_model


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Returns (model in eval mode, classes, cfg) from a best.pt checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg, classes = ck["cfg"], ck["classes"]
    model = build_model(cfg["model"], len(classes), cfg["dropout"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.eval(), classes, cfg


def export_onnx(model: torch.nn.Module, out: str | Path, image_size: int = 224, opset: int = 17) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().to(memory_format=torch.contiguous_format)
    dummy = torch.randn(1, 3, image_size, image_size)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, (dummy,), str(out), dynamo=False, **kwargs)
        except Exception:  # legacy exporter unavailable: use the dynamo exporter
            torch.onnx.export(model, (dummy,), str(out), dynamo=True, input_names=["input"],
                              output_names=["logits"], dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
                              opset_version=max(opset, 18))
    return out


def onnx_parity(model: torch.nn.Module, onnx_path: str | Path, atol: float = 1e-3, batch: int = 4) -> float:
    """Max |torch - onnxruntime| on random input. Raises if above atol."""
    import onnxruntime as ort

    x = torch.randn(batch, 3, 224, 224)
    with torch.no_grad():
        ref = model.eval()(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff:.2e} (> {atol})")
    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, _, _ = load_checkpoint(args.checkpoint)
    path = export_onnx(model, args.out)
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB), max diff {onnx_parity(model, path):.2e}")


if __name__ == "__main__":
    main()
