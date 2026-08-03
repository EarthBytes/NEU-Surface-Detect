"""Tests for the model definition."""

import torch

from training.model import build_model


def test_build_model_output_shape() -> None:
    model = build_model(num_classes=6)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 6)
