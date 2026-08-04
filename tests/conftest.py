"""Shared pytest fixtures."""

from __future__ import annotations

from importlib import import_module
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference.predictor import DefectPredictor
from tests.create_fixture_checkpoint import DEFAULT_METADATA, create_checkpoint

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
