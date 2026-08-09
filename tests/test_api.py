"""Tests for the FastAPI inference service."""

from importlib import import_module

from fastapi.testclient import TestClient

inference_app = import_module("inference.app")


def test_health_ok_when_model_loaded(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert len(body["classes"]) == 6


def test_health_degraded_without_model() -> None:
    original = inference_app.predictor
    try:
        with TestClient(inference_app.app) as client:
            inference_app.predictor = None
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "degraded"
    finally:
        inference_app.predictor = original


def test_predict_returns_class_and_confidence(api_client, sample_image_bytes) -> None:
    response = api_client.post(
        "/predict",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in body["probabilities"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 6


def test_predict_rejects_non_image(api_client) -> None:
    response = api_client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
