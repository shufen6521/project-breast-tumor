from __future__ import annotations

from torch import nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported Grad-CAM model: {model_name}")

