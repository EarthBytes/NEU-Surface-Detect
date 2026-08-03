"""MLflow experiment tracking, model logging, and registry helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow.tracking import MlflowClient

from training.utils import resolve_path

logger = logging.getLogger(__name__)


def resolve_tracking_uri(tracking_uri: str) -> str:
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "", 1)
        if not Path(db_path).is_absolute():
            db_path = str(resolve_path(db_path))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    return tracking_uri


def configure_mlflow(tracking_uri: str, experiment_name: str) -> str:
    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)
    return resolved_uri


def flatten_params(config: dict[str, Any]) -> dict[str, Any]:
    """Turn nested config into flat MLflow params."""
    flat: dict[str, Any] = {}
    for section, values in config.items():
        if section in ("paths", "mlflow", "inference"):
            continue
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
        else:
            flat[section] = values
    return flat


def log_training_params(config: dict[str, Any]) -> None:
    mlflow.log_params(flatten_params(config))


def log_epoch_metrics(epoch: int, metrics: dict[str, float]) -> None:
    prefixed = {f"epoch_{epoch}_{name}": value for name, value in metrics.items()}
    mlflow.log_metrics(prefixed, step=epoch)


def log_summary_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_model_artifact(
    model: Any,
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> str:
    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    mlflow.log_dict(metadata, "processed_metadata.json")

    image_size = metadata.get("image_size", 224)
    input_example = torch.randn(1, 3, image_size, image_size)

    model_info = mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=input_example,
        serialization_format="pickle",
    )
    return model_info.model_uri


def get_latest_run_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment named '{experiment_name}'")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    return runs.iloc[0]["run_id"]


def register_model_from_run(run_id: str, registered_model_name: str) -> Any:
    model_uri = f"runs:/{run_id}/model"
    return mlflow.register_model(model_uri, registered_model_name)


def promote_model_version(
    registered_model_name: str,
    version: int,
    stage: str,
    tracking_uri: str,
) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    alias = stage.lower()
    client.set_registered_model_alias(registered_model_name, alias, str(version))
    logger.info(
        "Promoted %s v%d to alias '%s'",
        registered_model_name,
        version,
        alias,
    )
