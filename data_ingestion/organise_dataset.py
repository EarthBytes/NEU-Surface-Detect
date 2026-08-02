#!/usr/bin/env python3
"""Organise the NEU-DET dataset into a standardised folder layout."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from data_ingestion.config import CLASS_NAMES, NEU_DET_ROOT, ORGANISED_DATA_DIR, SPLITS


def validate_layout(root: Path) -> list[str]:
    """Return a list of validation errors, empty if layout is valid."""
    errors: list[str] = []
    if not root.exists():
        return [f"Dataset root does not exist: {root}"]

    for split in SPLITS:
        images_dir = root / split / "images"
        if not images_dir.exists():
            errors.append(f"Missing: {images_dir}")
            continue
        for class_name in CLASS_NAMES:
            class_dir = images_dir / class_name
            if not class_dir.exists():
                errors.append(f"Missing class folder: {class_dir}")
            elif not any(class_dir.glob("*.jpg")):
                errors.append(f"No images in: {class_dir}")

        ann_dir = root / split / "annotations"
        if not ann_dir.exists():
            errors.append(f"Missing annotations: {ann_dir}")

    return errors


def write_manifest(root: Path) -> Path:
    """Write a JSON manifest describing the dataset layout."""
    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "NEU-DET",
        "root": str(root),
        "splits": {},
        "classes": CLASS_NAMES,
    }

    for split in SPLITS:
        split_info: dict[str, int | str] = {"images": 0}
        images_dir = root / split / "images"
        class_counts: dict[str, int] = {}
        if images_dir.exists():
            for class_name in CLASS_NAMES:
                class_dir = images_dir / class_name
                count = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
                class_counts[class_name] = count
            split_info["images"] = sum(class_counts.values())
        split_info["per_class"] = class_counts
        ann_dir = root / split / "annotations"
        split_info["annotations"] = len(list(ann_dir.glob("*.xml"))) if ann_dir.exists() else 0
        manifest["splits"][split] = split_info

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def organise(source: Path, dest: Path, copy: bool) -> Path:
    """
    Ensure dataset exists at dest in the standard layout.

    If source != dest, copy or symlink the tree. Otherwise validate in place.
    """
    if source.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        if copy:
            shutil.copytree(source, dest)
        else:
            shutil.copytree(source, dest, symlinks=True)
        print(f"{'Copied' if copy else 'Linked'} {source} -> {dest}")
    else:
        print(f"Organising in place at {dest}")

    errors = validate_layout(dest)
    if errors:
        print("\nLayout validation failed:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    manifest_path = write_manifest(dest)
    print(f"Manifest written to {manifest_path}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Organise NEU-DET into a standardised folder layout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=NEU_DET_ROOT,
        help="Source NEU-DET folder",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ORGANISED_DATA_DIR,
        help="Destination for the organised dataset",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking when source != dest",
    )
    args = parser.parse_args()

    organise(args.source, args.dest, copy=args.copy)
    print("\nStandard layout:")
    print("  {root}/{split}/images/{class}/*.jpg")
    print("  {root}/{split}/annotations/*.xml")
    print("  {root}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
