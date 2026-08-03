"""Shared model definition for training and inference."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
