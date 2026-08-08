#!/usr/bin/env python3
"""Step 13 retraining loop: train, compare, promote."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlflow

from training.aws_training import AwsTrainingError, launch_sagemaker_training
from training.compare_models import compare_models
from training.mlflow_tracking import (
    configure_mlflow,
    is_mlflow_enabled,
    log_summary_metrics,
    log_training_run_from_checkpoint,
    promote_model_version,
    register_model_from_run,
    resolve_tracking_uri,
)
from training.utils import load_config, resolve_path, setup_logging

logger = setup_logging(__name__)


def _copy_champion_backup(champion_path: Path) -> Path:
    backup_path = champion_path.with_name(f"{champion_path.stem}_backup{champion_path.suffix}")
    shutil.copy2(champion_path, backup_path)
    return backup_path


def run_retraining_loop(
    config_path: Path | None = None,
    *,
    launch_aws: bool = False,
    epochs: int | None = None,
    job_name: str | None = None,
    challenger_path: Path | None = None,
    champion_path: Path | None = None,
    promote: bool = False,
    compare_only: bool = False,
) -> dict:
    config = load_config(config_path)
    retrain_cfg = config.get("retraining", {})
    mlflow_cfg = config["mlflow"]

    champion = champion_path or resolve_path(retrain_cfg.get("champion_checkpoint", "models/checkpoints/best_model.pt"))
    challenger = challenger_path or resolve_path(
        retrain_cfg.get("challenger_checkpoint", "models/checkpoints/challenger_model.pt")
    )
    min_improvement = float(retrain_cfg.get("min_improvement", 0.0))
    comparison_output = resolve_path(retrain_cfg.get("comparison_output", "models/evaluation/comparison.json"))

    if launch_aws and not compare_only:
        logger.info("Launching SageMaker retraining job...")
        _, downloaded = launch_sagemaker_training(
            config_path=config_path,
            epochs_override=epochs,
            job_name=job_name,
            wait=True,
        )
        if downloaded is None:
            raise AwsTrainingError("SageMaker job finished but no checkpoint was downloaded.")
        shutil.copy2(downloaded, challenger)
        logger.info("Challenger checkpoint saved to %s", challenger)

    if not challenger.exists():
        raise FileNotFoundError(
            f"Challenger checkpoint not found at {challenger}. "
            "Run with --launch-aws or provide --challenger."
        )
    if not champion.exists():
        raise FileNotFoundError(f"Champion checkpoint not found at {champion}.")

    comparison = compare_models(champion, challenger, config_path, min_improvement=min_improvement)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output.write_text(json.dumps(comparison, indent=2))

    if is_mlflow_enabled(mlflow_cfg["tracking_uri"]):
        retrain_experiment = retrain_cfg.get("experiment_name", "neu-surface-defect-retraining")
        configure_mlflow(mlflow_cfg["tracking_uri"], retrain_experiment)
        with mlflow.start_run(run_name=job_name or "retraining-loop") as run:
            mlflow.log_param("champion_checkpoint", str(champion))
            mlflow.log_param("challenger_checkpoint", str(challenger))
            mlflow.log_param("winner", comparison["winner"])
            log_summary_metrics(
                {
                    "champion_test_f1_macro": comparison["champion"]["test_f1_macro"],
                    "challenger_test_f1_macro": comparison["challenger"]["test_f1_macro"],
                    "margin_f1_macro": comparison["margin_f1_macro"],
                }
            )
            mlflow.log_artifact(str(comparison_output), artifact_path="retraining")
            logger.info("Logged retraining comparison to MLflow run %s", run.info.run_id)

    if comparison["winner"] == "challenger" and promote:
        backup = _copy_champion_backup(champion)
        shutil.copy2(challenger, champion)
        logger.info("Promoted challenger to champion at %s (backup: %s)", champion, backup)

        if is_mlflow_enabled(mlflow_cfg["tracking_uri"]):
            run_id = log_training_run_from_checkpoint(
                champion,
                config,
                run_name="production-promotion",
                tags={"source": "retraining", "winner": "challenger"},
            )
            model_version = register_model_from_run(run_id, mlflow_cfg["registered_model_name"])
            promote_model_version(
                mlflow_cfg["registered_model_name"],
                model_version.version,
                "Production",
                resolve_tracking_uri(mlflow_cfg["tracking_uri"]),
            )
            comparison["promoted_run_id"] = run_id
            comparison["promoted_version"] = model_version.version
    elif comparison["winner"] != "challenger":
        logger.info("Challenger did not beat champion; keeping current production model.")

    comparison["comparison_output"] = str(comparison_output)
    comparison_output.write_text(json.dumps(comparison, indent=2))
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Step 13 retraining loop.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--launch-aws", action="store_true", help="Train challenger on SageMaker first")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for AWS training")
    parser.add_argument("--job-name", type=str, default=None, help="Optional SageMaker job name")
    parser.add_argument("--champion", type=Path, default=None, help="Champion checkpoint path")
    parser.add_argument("--challenger", type=Path, default=None, help="Challenger checkpoint path")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Replace champion and register in MLflow if challenger wins",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip AWS launch and only compare existing checkpoints",
    )
    args = parser.parse_args()

    try:
        result = run_retraining_loop(
            args.config,
            launch_aws=args.launch_aws,
            epochs=args.epochs,
            job_name=args.job_name,
            challenger_path=args.challenger,
            champion_path=args.champion,
            promote=args.promote,
            compare_only=args.compare_only,
        )
    except (FileNotFoundError, AwsTrainingError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Retraining loop finished. Winner: %s", result["winner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
