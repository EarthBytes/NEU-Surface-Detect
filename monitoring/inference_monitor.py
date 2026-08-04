"""Log inference events and evaluate drift during prediction."""

from __future__ import annotations

import json
import logging
from typing import Any

from monitoring.alerting import AlertManager
from monitoring.drift_check import DriftAlert, DriftTracker
from monitoring.metrics import PredictionRecord, build_prediction_record
from training.utils import resolve_path

logger = logging.getLogger(__name__)


class InferenceMonitor:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        brightness_baseline: float | None = None,
    ) -> None:
        monitoring_cfg = config.get("monitoring", {})
        self.enabled = monitoring_cfg.get("enabled", True)
        self.log_path = resolve_path(monitoring_cfg.get("log_path", "models/monitoring/predictions.jsonl"))

        baseline = brightness_baseline
        if baseline is None:
            baseline = monitoring_cfg.get("brightness_baseline", 0.5045)

        self.tracker = DriftTracker(
            window_size=monitoring_cfg.get("drift_window_size", 100),
            low_confidence_threshold=monitoring_cfg.get("low_confidence_threshold", 0.5),
            class_share_threshold=monitoring_cfg.get("class_share_threshold", 0.6),
            brightness_baseline=baseline,
            brightness_tolerance=monitoring_cfg.get("brightness_tolerance", 0.15),
        )
        self.alerts = AlertManager()
        self._recent_alerts: list[DriftAlert] = []

        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_prediction(
        self,
        image_bytes: bytes,
        result: dict[str, float | str | dict[str, float]],
    ) -> list[DriftAlert]:
        if not self.enabled:
            return []

        record = build_prediction_record(
            image_bytes=image_bytes,
            predicted_class=str(result["predicted_class"]),
            confidence=float(result["confidence"]),
            probabilities={
                class_name: float(probability)
                for class_name, probability in result["probabilities"].items()
            },
        )
        self._append_record(record)

        triggered = self.tracker.add(record)
        for alert in triggered:
            self.alerts.notify(alert, record.to_dict())
        self._recent_alerts.extend(triggered)
        if len(self._recent_alerts) > 50:
            self._recent_alerts = self._recent_alerts[-50:]

        return triggered

    def summary(self) -> dict[str, Any]:
        tracker_summary = self.tracker.summary()
        return {
            "enabled": self.enabled,
            "log_path": str(self.log_path),
            "recent_alerts": [
                {"code": alert.code, "message": alert.message, "severity": alert.severity}
                for alert in self._recent_alerts[-10:]
            ],
            **tracker_summary,
        }

    def _append_record(self, record: PredictionRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")
        logger.info(
            "Logged prediction class=%s confidence=%.3f brightness=%.3f",
            record.predicted_class,
            record.confidence,
            record.image_stats.mean_brightness,
        )
