"""MLflow experiment tracking, model logging, and registry helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow.tracking import MlflowClient

from training.utils import resolve_path

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower()


def is_sagemaker_job() -> bool:
    """True when running inside a SageMaker training container."""
    return bool(os.environ.get("SM_TRAINING_ENV") or os.environ.get("TRAINING_JOB_NAME"))


def is_remote_tracking_uri(tracking_uri: str) -> bool:
    lowered = tracking_uri.strip().lower()
    return lowered.startswith(("http://", "https://", "databricks"))


def resolve_tracking_uri(tracking_uri: str) -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", tracking_uri)

    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "", 1)
        if not Path(db_path).is_absolute():
            db_path = str(resolve_path(db_path))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    return tracking_uri


def is_mlflow_enabled(tracking_uri: str) -> bool:
    explicit = _env_flag("MLFLOW_ENABLED")
    if explicit is not None:
        return explicit in {"1", "true", "yes", "on"}

    resolved = os.environ.get("MLFLOW_TRACKING_URI", tracking_uri)
    if is_sagemaker_job() and not is_remote_tracking_uri(resolved):
        logger.info(
            "Skipping MLflow on SageMaker: tracking URI %r is local/ephemeral. "
            "Set MLFLOW_TRACKING_URI to an http(s) server to enable remote tracking.",
            resolved,
        )
        return False
    return True


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
    input_example = torch.randn(1, 3, image_size, image_size).numpy()

    model_info = mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=input_example,
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


def log_evaluation_run(
    mlflow_cfg: dict,
    metrics: dict[str, float],
    artifact_paths: dict[str, Path],
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    extra_params: dict[str, str] | None = None,
) -> str | None:
    """Log evaluation metrics and artifacts to MLflow. Returns the run ID."""
    if not is_mlflow_enabled(mlflow_cfg["tracking_uri"]):
        return None

    configure_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"])

    if run_id:
        with mlflow.start_run(run_id=run_id):
            if extra_params:
                mlflow.log_params(extra_params)
            log_summary_metrics(metrics)
            for artifact_path, subdir in artifact_paths.items():
                mlflow.log_artifact(artifact_path, artifact_path=subdir)
        return run_id

    resolved_name = run_name or "evaluation"
    with mlflow.start_run(run_name=resolved_name) as run:
        if extra_params:
            mlflow.log_params(extra_params)
        log_summary_metrics(metrics)
        for artifact_path, subdir in artifact_paths.items():
            mlflow.log_artifact(artifact_path, artifact_path=subdir)
        return run.info.run_id


def log_training_run_from_checkpoint(
    checkpoint_path: Path,
    config: dict,
    *,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Create an MLflow run for an existing checkpoint (e.g. after SageMaker training)."""
    mlflow_cfg = config["mlflow"]
    if not is_mlflow_enabled(mlflow_cfg["tracking_uri"]):
        raise RuntimeError("MLflow is disabled for this environment.")

    import torch

    from training.model import build_model

    configure_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"])
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint_data.get("metadata")
    if metadata is None:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing metadata")

    resolved_name = run_name or f"imported-{checkpoint_path.stem}"
    with mlflow.start_run(run_name=resolved_name) as run:
        log_training_params(config)
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("processed_version", config["data"]["processed_version"])
        mlflow.set_tag("source", tags.get("source", "import") if tags else "import")

        summary = {}
        if checkpoint_data.get("val_accuracy") is not None:
            summary["best_val_accuracy"] = float(checkpoint_data["val_accuracy"])
        if checkpoint_data.get("epoch") is not None:
            summary["best_epoch"] = float(checkpoint_data["epoch"])
        if summary:
            log_summary_metrics(summary)

        model = build_model(config["model"]["num_classes"])
        model.load_state_dict(checkpoint_data["model_state_dict"])
        log_model_artifact(model, checkpoint_path, metadata)
        return run.info.run_id
