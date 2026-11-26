"""App inference helpers with an untrained checkpoint (no trained weights needed)."""

import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

import inference
from cropdoc.export import export_onnx
from cropdoc.models import build_model


@pytest.fixture()
def models_dir(tmp_path):
    classes = sorted(inference.load_disease_info().keys())
    model = build_model("custom_cnn", num_classes=len(classes), dropout=0.3)
    torch.save({"model": model.state_dict(), "cfg": {"model": "custom_cnn", "dropout": 0.3}, "classes": classes},
               tmp_path / "custom_cnn.pt")
    export_onnx(model.eval(), tmp_path / "custom_cnn.onnx")
    (tmp_path / "models.json").write_text(json.dumps({"custom_cnn": {"unsure_threshold": 0.42}}))
    return tmp_path


def _jpeg(w=300, h=200):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def test_disease_info_covers_38_classes():
    info = inference.load_disease_info()
    assert len(info) == 38
    for v in info.values():
        assert {"crop", "disease", "healthy", "management", "source_url", "verified"} <= set(v)


def test_predict_onnx_and_torch_agree(models_dir):
    lm = inference.load_model("custom_cnn", models_dir)
    assert lm.session is not None and lm.unsure_threshold == 0.42
    img = inference.read_upload(_jpeg())
    p_onnx, x = inference.predict(lm, img)
    lm.session = None
    p_torch, _ = inference.predict(lm, img)
    assert p_onnx.shape == (38,) and abs(p_onnx.sum() - 1) < 1e-4
    assert np.abs(p_onnx - p_torch).max() < 1e-4
    top = inference.top_k(p_torch, lm.classes, 3)
    assert len(top) == 3 and top[0][1] >= top[1][1] >= top[2][1]
    cam = inference.gradcam_overlay(lm, x, lm.classes.index(top[0][0]))
    assert cam.shape == (224, 224, 3) and cam.dtype == np.uint8


def test_invalid_uploads_rejected():
    with pytest.raises(inference.InvalidImage):
        inference.read_upload(b"not an image")
    with pytest.raises(inference.InvalidImage):
        inference.read_upload(_jpeg(40, 40))


def test_unsure_and_pretty():
    assert inference.unsure_reason(np.array([0.3, 0.2, 0.5]), tau=0.6) is not None
    assert inference.unsure_reason(np.array([0.05, 0.05, 0.9]), tau=0.6) is None
    assert inference.pretty("Tomato___Early_blight") == ("Tomato", "Early blight")
    assert inference.pretty("Apple___healthy") == ("Apple", "Healthy")
