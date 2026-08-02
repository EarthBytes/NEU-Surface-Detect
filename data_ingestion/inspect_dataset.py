#!/usr/bin/env python3
"""Inspect the NEU-DET dataset: structure, labels, image sizes, and quality."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from data_ingestion.config import CLASS_NAMES, NEU_DET_ROOT, SPLITS


def count_images_by_split(root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_counts: dict[str, int] = {}
        images_dir = root / split / "images"
        if not images_dir.exists():
            counts[split] = split_counts
            continue
        for class_name in CLASS_NAMES:
            class_dir = images_dir / class_name
            split_counts[class_name] = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
        counts[split] = split_counts
    return counts


def analyse_images(root: Path) -> dict:
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    corrupt: list[str] = []
    total = 0

    for split in SPLITS:
        images_dir = root / split / "images"
        if not images_dir.exists():
            continue
        for image_path in images_dir.rglob("*.jpg"):
            total += 1
            try:
                with Image.open(image_path) as img:
                    sizes[img.size] += 1
                    modes[img.mode] += 1
            except Exception as exc:  # noqa: BLE001
                corrupt.append(f"{image_path}: {exc}")

    return {
        "total_images": total,
        "sizes": {f"{w}x{h}": count for (w, h), count in sizes.items()},
        "modes": dict(modes),
        "corrupt_files": corrupt,
    }


def count_annotations(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        ann_dir = root / split / "annotations"
        counts[split] = len(list(ann_dir.glob("*.xml"))) if ann_dir.exists() else 0
    return counts


def build_report(root: Path) -> dict:
    if not root.exists():
        return {
            "dataset_path": str(root),
            "exists": False,
            "message": "Dataset not found. Run data_ingestion.download_neu first.",
        }

    split_counts = count_images_by_split(root)
    image_stats = analyse_images(root)
    annotation_counts = count_annotations(root)

    train_total = sum(split_counts.get("train", {}).values())
    val_total = sum(split_counts.get("validation", {}).values())

    return {
        "dataset_path": str(root),
        "exists": True,
        "splits": list(SPLITS),
        "classes": CLASS_NAMES,
        "images_per_class": split_counts,
        "totals": {"train": train_total, "validation": val_total, "all": train_total + val_total},
        "image_stats": image_stats,
        "annotation_counts": annotation_counts,
        "recommendation": (
            "Start with classification only. All images are 200x200 greyscale with "
            "balanced class counts. Pascal VOC XML annotations are available for "
            "future object-detection work."
        ),
    }


def print_report(report: dict) -> None:
    print("=" * 60)
    print("NEU-DET Dataset Inspection Report")
    print("=" * 60)
    print(f"Path: {report['dataset_path']}")

    if not report.get("exists"):
        print(f"\n{report['message']}")
        return

    print(f"\nClasses ({len(report['classes'])}): {', '.join(report['classes'])}")
    print(f"Splits: {', '.join(report['splits'])}")

    print("\nImages per class:")
    for split, counts in report["images_per_class"].items():
        print(f"  {split}:")
        for cls, count in counts.items():
            print(f"    {cls}: {count}")

    totals = report["totals"]
    print(f"\nTotals: train={totals['train']}, validation={totals['validation']}, all={totals['all']}")

    stats = report["image_stats"]
    print("\nImage statistics:")
    print(f"  Sizes: {stats['sizes']}")
    print(f"  Modes: {stats['modes']}")
    if stats["corrupt_files"]:
        print(f"  Corrupt files ({len(stats['corrupt_files'])}):")
        for entry in stats["corrupt_files"][:5]:
            print(f"    - {entry}")
    else:
        print("  Corrupt files: none")

    ann = report["annotation_counts"]
    print(f"\nAnnotations: train={ann.get('train', 0)}, validation={ann.get('validation', 0)}")

    print(f"\nRecommendation: {report['recommendation']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the NEU-DET dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=NEU_DET_ROOT,
        help="Path to the NEU-DET dataset root",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON",
    )
    args = parser.parse_args()

    report = build_report(args.root)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {args.json}")

    return 0 if report.get("exists") else 1


if __name__ == "__main__":
    raise SystemExit(main())
