#!/usr/bin/env python3
"""Import an existing checkpoint into MLflow (e.g. after SageMaker training)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.mlflow_tracking import is_mlflow_enabled, log_training_run_from_checkpoint
from training.utils import load_config, resolve_path, setup_logging

logger = setup_logging(__name__)


def attach_run_id_to_checkpoint(checkpoint_path: Path, run_id: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["run_id"] = run_id
    torch.save(checkpoint, checkpoint_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log an existing checkpoint to MLflow and optionally update its run_id."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to import (defaults to models/checkpoints/best_model.pt)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name")
    parser.add_argument(
        "--update-checkpoint",
        action="store_true",
        help="Write the new MLflow run_id back into the checkpoint file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    mlflow_cfg = config["mlflow"]
    checkpoint = args.checkpoint or (
        resolve_path(config["paths"]["checkpoints"]) / config["training"]["checkpoint_name"]
    )

    if not is_mlflow_enabled(mlflow_cfg["tracking_uri"]):
        logger.error(
            "MLflow is disabled. Use a local sqlite tracking URI or set MLFLOW_ENABLED=true."
        )
        return 1

    try:
        run_id = log_training_run_from_checkpoint(
            checkpoint,
            config,
            run_name=args.run_name,
            tags={"source": "import"},
        )
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Logged checkpoint to MLflow run %s", run_id)

    if args.update_checkpoint:
        attach_run_id_to_checkpoint(checkpoint, run_id)
        logger.info("Updated checkpoint %s with run_id", checkpoint)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
