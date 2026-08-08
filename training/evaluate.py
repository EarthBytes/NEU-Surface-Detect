#!/usr/bin/env python3
"""Evaluate a trained classifier and report classification metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_ingestion.config import CLASS_NAMES
from training.data_source import DatasetSourceError, resolve_processed_root
from training.dataset import create_dataloaders
from training.metrics import collect_predictions, summarise_misclassifications
from training.mlflow_tracking import is_mlflow_enabled, log_evaluation_run
from training.model import build_model
from training.utils import get_device, load_config, resolve_path, setup_logging

logger = setup_logging(__name__)


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                format(matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    run_id: str | None = None,
) -> Path:
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]
    device = get_device()

    processed_root = resolve_processed_root(config)
    checkpoint = checkpoint_path or (
        resolve_path(config["paths"]["checkpoints"]) / config["training"]["checkpoint_name"]
    )
    evaluation_dir = resolve_path(config["paths"]["evaluation"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, _ = create_dataloaders(
        processed_root=processed_root,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        aug_cfg=config["augmentation"],
        splits=("test",),
    )
    if test_loader is None:
        raise RuntimeError("Test dataloader was not created.")

    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(model_cfg["num_classes"])
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "checkpoint": str(checkpoint),
        "processed_version": data_cfg["processed_version"],
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "misclassifications": summarise_misclassifications(y_true, y_pred, CLASS_NAMES),
    }

    matrix = confusion_matrix(y_true, y_pred)
    metrics_path = evaluation_dir / "metrics.json"
    matrix_path = evaluation_dir / "confusion_matrix.png"
    matrix_csv_path = evaluation_dir / "confusion_matrix.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(matrix, CLASS_NAMES, matrix_path)
    np.savetxt(matrix_csv_path, matrix, fmt="%d", delimiter=",", header=",".join(CLASS_NAMES))

    resolved_run_id = run_id or checkpoint_data.get("run_id")
    mlflow_metrics = {
        "test_accuracy": metrics["test_accuracy"],
        "test_precision_macro": metrics["test_precision_macro"],
        "test_recall_macro": metrics["test_recall_macro"],
        "test_f1_macro": metrics["test_f1_macro"],
    }
    logged_run_id = log_evaluation_run(
        mlflow_cfg,
        mlflow_metrics,
        {
            str(metrics_path): "evaluation",
            str(matrix_path): "evaluation",
        },
        run_id=resolved_run_id,
        run_name=f"evaluation-{Path(checkpoint).stem}",
        extra_params={
            "checkpoint": str(checkpoint),
            "processed_version": data_cfg["processed_version"],
        },
    )
    if logged_run_id:
        logger.info("Logged test metrics to MLflow run %s", logged_run_id)
    elif is_mlflow_enabled(mlflow_cfg["tracking_uri"]):
        logger.warning("MLflow logging was skipped unexpectedly")
    else:
        logger.info("MLflow disabled; test metrics saved locally only")

    logger.info("Test accuracy:  %.4f", metrics["test_accuracy"])
    logger.info("Precision (macro): %.4f", metrics["test_precision_macro"])
    logger.info("Recall (macro):    %.4f", metrics["test_recall_macro"])
    logger.info("F1 (macro):        %.4f", metrics["test_f1_macro"])
    logger.info("Results saved to %s", evaluation_dir)

    if metrics["misclassifications"]:
        logger.info("Most common misclassifications:")
        for entry in metrics["misclassifications"][:3]:
            logger.info(
                "  %s → %s (%d)",
                entry["true_class"],
                entry["predicted_class"],
                entry["count"],
            )

    return metrics_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained defect classifier.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path")
    parser.add_argument("--run-id", type=str, default=None, help="Optional MLflow run ID")
    args = parser.parse_args()

    try:
        evaluate(args.config, args.checkpoint, args.run_id)
    except (FileNotFoundError, DatasetSourceError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
