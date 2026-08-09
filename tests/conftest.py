"""Shared pytest fixtures."""

from __future__ import annotations

import importlib.util
from importlib import import_module
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference.predictor import DefectPredictor

# Load from this directory to avoid collision with SageMaker's top-level `tests` package.
_fixture_module_path = Path(__file__).with_name("create_fixture_checkpoint.py")
_spec = importlib.util.spec_from_file_location("create_fixture_checkpoint", _fixture_module_path)
assert _spec and _spec.loader
_fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture_module)
DEFAULT_METADATA = _fixture_module.DEFAULT_METADATA
create_checkpoint = _fixture_module.create_checkpoint

inference_app = import_module("inference.app")


@pytest.fixture
def fixture_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkpoint = create_checkpoint(tmp_path / "best_model.pt")
    monkeypatch.setenv("MODEL_CHECKPOINT", str(checkpoint))
    return checkpoint


@pytest.fixture
def sample_metadata() -> dict:
    return DEFAULT_METADATA.copy()


@pytest.fixture
def predictor(fixture_checkpoint: Path) -> DefectPredictor:
    return DefectPredictor(fixture_checkpoint)


@pytest.fixture
def api_client(fixture_checkpoint: Path) -> TestClient:
    with TestClient(inference_app.app) as client:
        yield client


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = Image.new("L", (200, 200), color=128)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
