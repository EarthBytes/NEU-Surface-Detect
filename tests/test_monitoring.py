"""Tests for inference monitoring and drift detection."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from monitoring.alerting import AlertManager
from monitoring.drift_check import DriftTracker
from monitoring.inference_monitor import InferenceMonitor
from monitoring.metrics import build_prediction_record, compute_image_stats
from training.utils import load_config


def _image_bytes(brightness: int = 128) -> bytes:
    image = Image.new("L", (64, 64), color=brightness)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_compute_image_stats_from_greyscale_image() -> None:
    stats = compute_image_stats(_image_bytes(brightness=200))

    assert stats.width == 64
    assert stats.height == 64
    assert stats.mean_brightness == pytest.approx(200 / 255, rel=1e-3)
    assert stats.contrast == pytest.approx(0.0, abs=1e-6)


def test_drift_tracker_flags_low_confidence() -> None:
    tracker = DriftTracker(low_confidence_threshold=0.8)
    record = build_prediction_record(
        image_bytes=_image_bytes(),
        predicted_class="crazing",
        confidence=0.2,
        probabilities={"crazing": 0.2},
    )

    alerts = tracker.add(record)

    assert any(alert.code == "low_confidence" for alert in alerts)


def test_drift_tracker_flags_brightness_drift() -> None:
    tracker = DriftTracker(brightness_baseline=0.5, brightness_tolerance=0.05)
    record = build_prediction_record(
        image_bytes=_image_bytes(brightness=20),
        predicted_class="scratches",
        confidence=0.95,
        probabilities={"scratches": 0.95},
    )

    alerts = tracker.add(record)

    assert any(alert.code == "brightness_drift" for alert in alerts)


def test_alert_manager_invokes_registered_hooks() -> None:
    from monitoring.drift_check import DriftAlert

    manager = AlertManager()
    seen: list[str] = []

    manager.register_hook(lambda alert, _context: seen.append(alert.code))
    manager.notify(
        DriftAlert(code="low_confidence", message="test"),
        {"predicted_class": "patches"},
    )

    assert seen == ["low_confidence"]


def test_inference_monitor_writes_jsonl_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "predictions.jsonl"
    config = load_config()
    config["monitoring"] = {
        "enabled": True,
        "log_path": str(log_path),
        "drift_window_size": 20,
        "low_confidence_threshold": 0.5,
    }

    monitor = InferenceMonitor(config)
    image_bytes = _image_bytes()
    result = {
        "predicted_class": "inclusion",
        "confidence": 0.91,
        "probabilities": {"inclusion": 0.91},
    }

    monitor.record_prediction(image_bytes, result)

    assert log_path.exists()
    logged = log_path.read_text(encoding="utf-8")
    assert "inclusion" in logged
    assert "mean_brightness" in logged


def test_monitoring_summary_endpoint(api_client, sample_image_bytes, tmp_path: Path) -> None:
    summary_before = api_client.get("/monitoring/summary")
    assert summary_before.status_code == 200
    assert summary_before.json()["enabled"] is True

    response = api_client.post(
        "/predict",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200

    summary_after = api_client.get("/monitoring/summary")
    body = summary_after.json()
    assert summary_after.status_code == 200
    assert body["prediction_count"] >= 1
    assert body["average_confidence"] is not None


def test_config_includes_monitoring_section() -> None:
    config = load_config()
    assert "monitoring" in config
    assert config["monitoring"]["enabled"] is True
