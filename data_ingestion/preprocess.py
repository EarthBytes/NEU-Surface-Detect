#!/usr/bin/env python3
"""Resize, normalise, and split the NEU-DET dataset into a versioned layout."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from data_ingestion.config import CLASS_NAMES, NEU_DET_ROOT, PROCESSED_DATA_ROOT
from training.utils import load_config, resolve_path, set_seed, setup_logging

logger = setup_logging(__name__)


def collect_images(source_root: Path, split: str) -> dict[str, list[Path]]:
    images_by_class: dict[str, list[Path]] = {}
    images_dir = source_root / split / "images"
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        images_by_class[class_name] = sorted(class_dir.glob("*.jpg")) if class_dir.exists() else []
    return images_by_class


def split_train_for_test(
    train_images: dict[str, list[Path]],
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Hold out a stratified test set from the training images."""
    rng = random.Random(seed)
    new_train: dict[str, list[Path]] = {}
    test_images: dict[str, list[Path]] = {}

    for class_name, paths in train_images.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * test_ratio))
        test_images[class_name] = shuffled[:test_count]
        new_train[class_name] = shuffled[test_count:]

    return new_train, test_images


def save_greyscale_image(source: Path, destination: Path, size: int) -> np.ndarray:
    """Resize to greyscale and return pixel values in [0, 1] for stats."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        grey = img.convert("L").resize((size, size), Image.BILINEAR)
        grey.save(destination)
        return np.asarray(grey, dtype=np.float32) / 255.0


def compute_normalisation_stats(train_arrays: list[np.ndarray]) -> dict[str, float]:
    stacked = np.concatenate([arr.flatten() for arr in train_arrays])
    return {"mean": float(stacked.mean()), "std": float(max(stacked.std(), 1e-6))}


def write_split(
    images_by_class: dict[str, list[Path]],
    output_root: Path,
    split_name: str,
    size: int,
    collect_stats: bool = False,
) -> list[np.ndarray]:
    saved_arrays: list[np.ndarray] = []
    for class_name, paths in images_by_class.items():
        for source in paths:
            destination = output_root / split_name / class_name / source.name
            array = save_greyscale_image(source, destination, size)
            if collect_stats:
                saved_arrays.append(array)
    return saved_arrays


def build_metadata(
    version: str,
    size: int,
    normalisation: dict[str, float],
    split_counts: dict[str, dict[str, int]],
) -> dict:
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_size": size,
        "colour_mode": "greyscale",
        "class_names": CLASS_NAMES,
        "normalisation": normalisation,
        "splits": split_counts,
    }


def count_split(output_root: Path, split_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    split_dir = output_root / split_name
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        counts[class_name] = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
    return counts


def preprocess(
    source_root: Path,
    output_root: Path,
    version: str,
    image_size: int,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> Path:
    version_dir = output_root / version
    if version_dir.exists():
        if not overwrite:
            logger.info("Processed data already exists at %s", version_dir)
            return version_dir
        shutil.rmtree(version_dir)

    set_seed(seed)
    logger.info("Preparing version %s at %s", version, version_dir)

    train_images = collect_images(source_root, "train")
    validation_images = collect_images(source_root, "validation")
    train_images, test_images = split_train_for_test(train_images, test_ratio, seed)

    train_arrays = write_split(train_images, version_dir, "train", image_size, collect_stats=True)
    write_split(validation_images, version_dir, "validation", image_size)
    write_split(test_images, version_dir, "test", image_size)

    normalisation = compute_normalisation_stats(train_arrays)
    split_counts = {
        split: count_split(version_dir, split)
        for split in ("train", "validation", "test")
    }

    metadata = build_metadata(version, image_size, normalisation, split_counts)
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    totals = {split: sum(counts.values()) for split, counts in split_counts.items()}
    logger.info(
        "Finished preprocessing: train=%d, validation=%d, test=%d",
        totals["train"],
        totals["validation"],
        totals["test"],
    )
    logger.info("Normalisation stats: mean=%.4f, std=%.4f", normalisation["mean"], normalisation["std"])
    return version_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess the NEU-DET dataset.")
    parser.add_argument("--config", type=Path, default=None, help="Training config YAML")
    parser.add_argument("--source", type=Path, default=NEU_DET_ROOT, help="Raw NEU-DET root")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing version")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    version = data_cfg["processed_version"]
    output_root = resolve_path(config["paths"]["processed_data"])

    if not args.source.exists():
        logger.error("Source dataset not found at %s", args.source)
        return 1

    preprocess(
        source_root=args.source,
        output_root=output_root,
        version=version,
        image_size=data_cfg["image_size"],
        test_ratio=data_cfg["test_ratio"],
        seed=data_cfg["seed"],
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
