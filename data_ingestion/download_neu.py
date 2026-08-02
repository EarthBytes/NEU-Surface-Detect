#!/usr/bin/env python3
"""Download or locate the NEU Surface Defect (NEU-DET) dataset."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from data_ingestion.config import KAGGLE_DATASET_URL, NEU_DET_ROOT, RAW_DATA_DIR


def dataset_is_ready(root: Path) -> bool:
    """Return True if train and validation image folders exist with content."""
    for split in ("train", "validation"):
        images_dir = root / split / "images"
        if not images_dir.exists():
            return False
        if not any(images_dir.rglob("*.jpg")):
            return False
    return True


def try_kaggle_download(raw_dir: Path) -> Path | None:
    """Attempt download via the Kaggle CLI. Returns extracted root or None."""
    if shutil.which("kaggle") is None:
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "neu-det.zip"

    print("Downloading via Kaggle CLI ...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "kaustubhdikshit/neu-surface-defect-database",
            "-p",
            str(raw_dir),
            "-o",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Kaggle download failed: {result.stderr.strip()}")
        return None

    # Kaggle may write neu-det.zip or a similarly named archive
    archives = list(raw_dir.glob("*.zip"))
    if not archives:
        print("No zip archive found after Kaggle download.")
        return None

    archive = archives[0]
    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive.name} ...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_dir)

    # Find NEU-DET root inside extracted tree
    candidates = list(extract_dir.rglob("NEU-DET"))
    if candidates:
        return candidates[0]
    if (extract_dir / "train").exists():
        return extract_dir
    return None


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def download_or_locate(source: Path | None, dest: Path) -> Path:
    """Ensure NEU-DET exists at dest, copying from source or downloading."""
    if dataset_is_ready(dest):
        print(f"Dataset already present at {dest}")
        return dest

    if source and source.exists():
        print(f"Copying dataset from {source} to {dest} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy_tree(source, dest)
        return dest

    extracted = try_kaggle_download(RAW_DATA_DIR)
    if extracted and extracted.exists():
        print(f"Installing downloaded dataset to {dest} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy_tree(extracted, dest)
        return dest

    print("\nCould not download the dataset automatically.")
    print("Manual steps:")
    print(f"  1. Download from: {KAGGLE_DATASET_URL}")
    print("  2. Extract the archive")
    print(f"  3. Place the NEU-DET folder at: {dest}")
    print("\nAlternatively, install the Kaggle CLI and configure credentials:")
    print("  pip install kaggle")
    print("  kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database")
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or locate the NEU-DET dataset.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=NEU_DET_ROOT,
        help="Target path for the organised NEU-DET dataset",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional local path to copy an existing NEU-DET folder from",
    )
    args = parser.parse_args()

    download_or_locate(args.source, args.dest)

    if dataset_is_ready(args.dest):
        print(f"\nDataset ready at {args.dest}")
        return 0

    print(f"\nDataset at {args.dest} is incomplete.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
