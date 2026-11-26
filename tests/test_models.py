import pytest
import torch

from cropdoc.models import (
    MODEL_NAMES,
    build_model,
    count_params,
    freeze_backbone_bn,
    gradcam_target_layer,
    param_groups,
    set_backbone_trainable,
)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_shape(name):
    model = build_model(name, num_classes=38, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 38)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_target_layer_is_7x7(name):
    model = build_model(name, pretrained=False).eval()
    feats = {}
    h = gradcam_target_layer(model).register_forward_hook(lambda m, i, o: feats.update(out=o))
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    h.remove()
    assert feats["out"].shape[-2:] == (7, 7)


def test_custom_cnn_param_budget():
    n = count_params(build_model("custom_cnn"))
    assert 2.0e6 < n < 2.8e6, n


def test_freeze_keeps_backbone_bn_in_eval():
    model = build_model("efficientnet_b0", pretrained=False)
    set_backbone_trainable(model, False)
    model.train()
    freeze_backbone_bn(model)
    bns = [m for m in model.features.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)
    assert all(not p.requires_grad for p in model.features.parameters())
    groups = param_groups(model, head_lr=1e-3, backbone_lr=1e-4, weight_decay=0.01)
    # only head params remain trainable; biases get no weight decay
    assert {g["lr"] for g in groups} == {1e-3}
    assert any(g["weight_decay"] == 0.0 for g in groups)
