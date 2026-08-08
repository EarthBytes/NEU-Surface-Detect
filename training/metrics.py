"""Shared evaluation metrics helpers."""

from __future__ import annotations

from pathlib import Path

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
from torch.utils.data import DataLoader

from data_ingestion.config import CLASS_NAMES
from training.data_source import resolve_processed_root
from training.dataset import create_dataloaders
from training.model import build_model
from training.utils import get_device, load_config


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
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

    return [
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


def evaluate_checkpoint(
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> dict:
    """Run the test split and return metrics for a single checkpoint."""
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    device = get_device()

    processed_root = resolve_processed_root(config)
    _, _, test_loader, _ = create_dataloaders(
        processed_root=processed_root,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        aug_cfg=config["augmentation"],
        splits=("test",),
    )
    if test_loader is None:
        raise RuntimeError("Test dataloader was not created.")

    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_cfg["num_classes"])
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    return {
        "checkpoint": str(checkpoint_path),
        "processed_version": data_cfg["processed_version"],
        "val_accuracy": float(checkpoint_data.get("val_accuracy") or 0.0),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "misclassifications": summarise_misclassifications(y_true, y_pred, CLASS_NAMES),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "run_id": checkpoint_data.get("run_id"),
    }
