#!/usr/bin/env python3
"""Register a trained model in MLflow and promote it to a target stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.mlflow_tracking import (
    configure_mlflow,
    get_latest_run_id,
    promote_model_version,
    register_model_from_run,
)
from training.utils import load_config, setup_logging

logger = setup_logging(__name__)

VALID_STAGES = ("Staging", "Production")


def register_and_promote(
    config_path: Path | None,
    run_id: str | None,
    stage: str,
) -> None:
    config = load_config(config_path)
    mlflow_cfg = config["mlflow"]
    resolved_uri = configure_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"])

    resolved_run_id = run_id or get_latest_run_id(mlflow_cfg["experiment_name"])
    logger.info("Using MLflow run %s", resolved_run_id)

    model_version = register_model_from_run(resolved_run_id, mlflow_cfg["registered_model_name"])
    promote_model_version(
        mlflow_cfg["registered_model_name"],
        model_version.version,
        stage,
        resolved_uri,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Register and promote a model in MLflow.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run ID")
    parser.add_argument(
        "--stage",
        type=str,
        default="Staging",
        choices=VALID_STAGES,
        help="Registry stage to promote to",
    )
    args = parser.parse_args()

    try:
        register_and_promote(args.config, args.run_id, args.stage)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
