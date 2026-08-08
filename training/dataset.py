"""Dataset loading and transforms for processed NEU-DET images."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

VALID_SPLITS = frozenset({"train", "validation", "test"})


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
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None, dict]:
    unknown = set(splits) - VALID_SPLITS
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}. Expected subset of {sorted(VALID_SPLITS)}")
    if not splits:
        raise ValueError("At least one split must be requested.")

    metadata = load_metadata(processed_root)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    test_loader: DataLoader | None = None

    if "train" in splits:
        train_dataset = datasets.ImageFolder(
            processed_root / "train",
            transform=build_transforms(metadata, augment=True, aug_cfg=aug_cfg),
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    if "validation" in splits:
        val_dataset = datasets.ImageFolder(
            processed_root / "validation",
            transform=build_transforms(metadata, augment=False, aug_cfg=aug_cfg),
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    if "test" in splits:
        test_dataset = datasets.ImageFolder(
            processed_root / "test",
            transform=build_transforms(metadata, augment=False, aug_cfg=aug_cfg),
        )
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, metadata
