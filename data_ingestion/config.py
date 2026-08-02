"""Shared path and dataset configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
NEU_DET_ROOT = DATASET_ROOT / "NEU-DET"
RAW_DATA_DIR = DATASET_ROOT / "raw"
ORGANISED_DATA_DIR = NEU_DET_ROOT
PROCESSED_DATA_ROOT = DATASET_ROOT / "processed"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

SPLITS = ("train", "validation")

# Kaggle dataset page (manual download fallback)
KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database"
)
