"""Dataset loading and transforms for processed NEU-DET images."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_metadata(processed_root: Path) -> dict:
    metadata_path = processed_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata at {metadata_path}. Run data_ingestion.preprocess first."
        )
    return json.loads(metadata_path.read_text())


def build_transforms(metadata: dict, augment: bool, aug_cfg: dict) -> transforms.Compose:
    mean = metadata["normalisation"]["mean"]
    std = metadata["normalisation"]["std"]
    # ResNet expects three channels, so repeat the greyscale stats
    channel_stats = ([mean] * 3, [std] * 3)

    steps: list = []

    if augment:
        if aug_cfg.get("horizontal_flip"):
            steps.append(transforms.RandomHorizontalFlip(p=aug_cfg["horizontal_flip"]))
        if aug_cfg.get("rotation_degrees"):
            steps.append(transforms.RandomRotation(degrees=aug_cfg["rotation_degrees"]))

    steps.extend([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(*channel_stats),
    ])
    return transforms.Compose(steps)


def create_dataloaders(
    processed_root: Path,
    batch_size: int,
    num_workers: int,
    aug_cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    metadata = load_metadata(processed_root)

    train_dataset = datasets.ImageFolder(
        processed_root / "train",
        transform=build_transforms(metadata, augment=True, aug_cfg=aug_cfg),
    )
    val_dataset = datasets.ImageFolder(
        processed_root / "validation",
        transform=build_transforms(metadata, augment=False, aug_cfg=aug_cfg),
    )
    test_dataset = datasets.ImageFolder(
        processed_root / "test",
        transform=build_transforms(metadata, augment=False, aug_cfg=aug_cfg),
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, metadata
