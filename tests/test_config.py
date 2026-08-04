"""Tests for project configuration."""

from data_ingestion.config import CLASS_NAMES
from training.utils import load_config, resolve_path


def test_class_names_count() -> None:
    assert len(CLASS_NAMES) == 6


def test_config_loads_required_sections() -> None:
    config = load_config()
    for section in (
        "data",
        "dataset",
        "aws",
        "model",
        "training",
        "mlflow",
        "inference",
        "monitoring",
        "paths",
    ):
        assert section in config


def test_inference_checkpoint_path_is_under_project() -> None:
    config = load_config()
    checkpoint = resolve_path(config["inference"]["checkpoint"])
    assert checkpoint.name == "best_model.pt"
