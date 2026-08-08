"""Tests for model comparison and retraining helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from training.compare_models import compare_models


def test_compare_models_picks_challenger_when_better(tmp_path: Path) -> None:
    champion = tmp_path / "champion.pt"
    challenger = tmp_path / "challenger.pt"
    champion.write_bytes(b"x")
    challenger.write_bytes(b"y")

    champion_metrics = {
        "test_f1_macro": 0.80,
        "test_accuracy": 0.80,
        "test_precision_macro": 0.80,
        "test_recall_macro": 0.80,
        "checkpoint": str(champion),
    }
    challenger_metrics = {
        "test_f1_macro": 0.90,
        "test_accuracy": 0.90,
        "test_precision_macro": 0.90,
        "test_recall_macro": 0.90,
        "checkpoint": str(challenger),
    }

    with patch("training.compare_models.evaluate_checkpoint") as mock_eval:
        mock_eval.side_effect = [champion_metrics, challenger_metrics]
        result = compare_models(champion, challenger)

    assert result["winner"] == "challenger"
    assert result["margin_f1_macro"] == pytest.approx(0.10)


def test_log_run_attaches_run_id(tmp_path: Path) -> None:
    from training.log_run import attach_run_id_to_checkpoint

    checkpoint = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": {}, "metadata": {"image_size": 224}}, checkpoint)
    attach_run_id_to_checkpoint(checkpoint, "abc123")

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert data["run_id"] == "abc123"
