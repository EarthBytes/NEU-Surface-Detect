"""Tests for MLflow evaluation logging helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from training.mlflow_tracking import log_evaluation_run


def test_log_evaluation_run_creates_new_run_when_no_run_id(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    matrix_path = tmp_path / "matrix.png"
    metrics_path.write_text("{}")
    matrix_path.write_bytes(b"png")

    mlflow_cfg = {
        "tracking_uri": "sqlite:///tmp/test.db",
        "experiment_name": "test-exp",
    }

    with (
        patch("training.mlflow_tracking.is_mlflow_enabled", return_value=True),
        patch("training.mlflow_tracking.configure_mlflow"),
        patch("training.mlflow_tracking.mlflow.start_run") as mock_start_run,
        patch("training.mlflow_tracking.log_summary_metrics") as mock_log_metrics,
        patch("training.mlflow_tracking.mlflow.log_params"),
        patch("training.mlflow_tracking.mlflow.log_artifact"),
    ):
        mock_run = MagicMock()
        mock_run.info.run_id = "new-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        run_id = log_evaluation_run(
            mlflow_cfg,
            {"test_accuracy": 0.99},
            {str(metrics_path): "evaluation", str(matrix_path): "evaluation"},
            run_name="evaluation-best_model",
            extra_params={"checkpoint": "models/checkpoints/best_model.pt"},
        )

    assert run_id == "new-run-id"
    mock_log_metrics.assert_called_once()
