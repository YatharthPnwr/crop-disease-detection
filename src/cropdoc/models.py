"""The two models we compare: a custom CNN (from scratch) and EfficientNet-B0 (ImageNet).

Both expose `.features` (backbone) and `.classifier` (head), so the training loop
can freeze the backbone and give the two parts different learning rates.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

MODEL_NAMES = ("custom_cnn", "efficientnet_b0")
DISPLAY_NAMES = {"custom_cnn": "Custom CNN", "efficientnet_b0": "EfficientNet-B0"}


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """[Conv3x3 -> BN -> ReLU] x2 -> MaxPool2x2. Two 3x3 convs see a 5x5 area."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """5 conv blocks (32-64-128-256-256) -> GAP -> dropout -> linear.

    224x224 input -> 7x7x256 feature map. The GAP head keeps the model at ~2.4M
    params instead of the tens of millions a flatten -> FC head would need.
    """

    def __init__(self, num_classes: int = 38, dropout: float = 0.3, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        blocks, c_in = [], 3
        for w in widths:
            blocks.append(conv_block(c_in, w))
            c_in = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes: int = 38, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout=dropout)
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features  # 1280
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last spatial stage (7x7 for 224 input): most semantic layer that keeps layout."""
    return model.features[-1]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def freeze_backbone_bn(model: nn.Module) -> None:
    """Put backbone BatchNorm layers in eval mode so frozen running stats do not drift.

    Must be called after every model.train(), which would otherwise re-enable them.
    """
    for m in model.features.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float, weight_decay: float) -> list[dict]:
    """AdamW groups: backbone vs head LR, and no weight decay on BN weights or biases."""
    groups = []
    for part, lr in ((model.features, backbone_lr), (model.classifier, head_lr)):
        decay, no_decay = [], []
        for p in part.parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
