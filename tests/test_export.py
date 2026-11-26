import pytest
import torch

from cropdoc.export import export_onnx, load_checkpoint, onnx_parity
from cropdoc.models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_onnx_matches_torch(name, tmp_path):
    model = build_model(name, num_classes=38, pretrained=False).eval()
    path = export_onnx(model, tmp_path / f"{name}.onnx")
    assert onnx_parity(model, path) < 1e-3


def test_checkpoint_round_trip(tmp_path):
    model = build_model("custom_cnn", num_classes=3, dropout=0.3)
    cfg = {"model": "custom_cnn", "dropout": 0.3}
    torch.save({"model": model.state_dict(), "cfg": cfg, "classes": ["a", "b", "c"]}, tmp_path / "best.pt")
    loaded, classes, _ = load_checkpoint(tmp_path / "best.pt")
    assert classes == ["a", "b", "c"] and not loaded.training
