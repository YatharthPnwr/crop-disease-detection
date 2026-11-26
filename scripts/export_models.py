"""Export every app/models/*.pt to ONNX next to it and check parity with PyTorch.

    python scripts/export_models.py
"""

from pathlib import Path

from cropdoc.export import export_onnx, load_checkpoint, onnx_parity


def main() -> None:
    for pt in sorted(Path("app/models").glob("*.pt")):
        model, _, _ = load_checkpoint(pt)
        onnx = export_onnx(model, pt.with_suffix(".onnx"))
        diff = onnx_parity(model, onnx)
        print(f"{pt.name} -> {onnx.name} ({onnx.stat().st_size / 1e6:.1f} MB), max |diff| {diff:.1e}")


if __name__ == "__main__":
    main()
