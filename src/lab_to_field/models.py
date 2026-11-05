"""Models: a small CNN built from scratch, and pretrained backbones from timm."""
import timm
import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "hardswish": nn.Hardswish}


def conv_block(c_in: int, c_out: int, act: str) -> nn.Sequential:
    """Conv -> BatchNorm -> activation -> MaxPool: the basic CNN building block.

    The 3x3 conv learns local patterns (edges, spots), BatchNorm keeps activations in a
    stable range, the activation adds non-linearity, and MaxPool halves height and width.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        ACTIVATIONS[act](),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Four conv blocks + global average pooling + linear classifier.

    For a 3x224x224 input the shapes are:
    3x224x224 -> 32x112x112 -> 64x56x56 -> 128x28x28 -> 256x14x14 -> pool -> 256 -> n_classes
    """

    def __init__(self, n_classes: int, act: str = "relu", dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, act),
            conv_block(32, 64, act),
            conv_block(64, 128, act),
            conv_block(128, 256, act),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def build_model(name: str, n_classes: int, pretrained: bool = True, freeze_backbone: bool = False,
                act: str = "relu", **timm_kwargs) -> nn.Module:
    """'simple_cnn' or any timm model name (e.g. 'mobilenetv3_large_100', 'efficientnet_b0').

    freeze_backbone=True trains only the final classifier (a "linear probe"): the pretrained
    features are kept exactly as they are.
    """
    if name == "simple_cnn":
        return SimpleCNN(n_classes, act=act)
    model = timm.create_model(name, pretrained=pretrained, num_classes=n_classes, **timm_kwargs)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
        model.frozen_backbone = True  # train loop keeps BatchNorm statistics fixed too
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
