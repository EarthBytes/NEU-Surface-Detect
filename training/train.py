#!/usr/bin/env python3
"""Train a ResNet18 defect classifier on processed NEU-DET data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models

from training.dataset import create_dataloaders
from training.utils import get_device, load_config, resolve_path, set_seed, setup_logging

logger = setup_logging(__name__)


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_training = optimiser is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimiser.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_training:
            loss.backward()
            optimiser.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: dict,
    epoch: int,
    val_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "metadata": metadata,
        },
        path,
    )


def train(config_path: Path | None = None, epochs_override: int | None = None) -> Path:
    config = load_config(config_path)
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    epochs = epochs_override or train_cfg["epochs"]

    set_seed(data_cfg["seed"])
    device = get_device()

    processed_root = resolve_path(config["paths"]["processed_data"]) / data_cfg["processed_version"]
    checkpoint_dir = resolve_path(config["paths"]["checkpoints"])
    checkpoint_path = checkpoint_dir / train_cfg["checkpoint_name"]

    train_loader, val_loader, _, metadata = create_dataloaders(
        processed_root=processed_root,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        aug_cfg=config["augmentation"],
    )

    model = build_model(model_cfg["num_classes"], model_cfg["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    best_val_accuracy = 0.0
    logger.info("Training on %s with %d epochs", device, epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(checkpoint_path, model, metadata, epoch, val_accuracy)
            logger.info(
                "Epoch %02d: train_loss=%.4f, val_loss=%.4f, val_acc=%.4f (new best)",
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
            )
        else:
            logger.info(
                "Epoch %02d: train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, val_acc=%.4f",
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )

    logger.info("Training complete. Best validation accuracy: %.4f", best_val_accuracy)
    logger.info("Checkpoint saved to %s", checkpoint_path)
    return checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the NEU defect classifier.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    try:
        train(args.config, epochs_override=args.epochs)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
