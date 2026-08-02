#!/usr/bin/env python3
"""Evaluate a trained classifier and report classification metrics."""

from __future__ import annotations

import argparse
import json
import logging
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
from torchvision import models

from data_ingestion.config import CLASS_NAMES
from training.dataset import create_dataloaders
from training.utils import get_device, load_config, resolve_path, setup_logging

logger = setup_logging(__name__)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


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


def summarise_misclassifications(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> list[dict[str, str | int]]:
    pairs: dict[tuple[int, int], int] = {}
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            key = (true_label, pred_label)
            pairs[key] = pairs.get(key, 0) + 1

    summary = [
        {
            "true_class": class_names[true_label],
            "predicted_class": class_names[pred_label],
            "count": count,
        }
        for (true_label, pred_label), count in sorted(
            pairs.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return summary


def evaluate(config_path: Path | None = None, checkpoint_path: Path | None = None) -> Path:
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    device = get_device()

    processed_root = resolve_path(config["paths"]["processed_data"]) / data_cfg["processed_version"]
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
    )

    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(model_cfg["num_classes"])
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
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

    logger.info("Test accuracy:  %.4f", metrics["accuracy"])
    logger.info("Precision (macro): %.4f", metrics["precision_macro"])
    logger.info("Recall (macro):    %.4f", metrics["recall_macro"])
    logger.info("F1 (macro):        %.4f", metrics["f1_macro"])
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
    args = parser.parse_args()

    try:
        evaluate(args.config, args.checkpoint)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
