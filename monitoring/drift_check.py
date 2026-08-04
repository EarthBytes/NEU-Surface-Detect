"""Track prediction patterns and flag potential drift."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from monitoring.metrics import PredictionRecord


@dataclass(frozen=True)
class DriftAlert:
    code: str
    message: str
    severity: str = "warning"


class DriftTracker:
    def __init__(
        self,
        *,
        window_size: int = 100,
        low_confidence_threshold: float = 0.5,
        class_share_threshold: float = 0.6,
        brightness_baseline: float = 0.5045,
        brightness_tolerance: float = 0.15,
    ) -> None:
        self.window_size = window_size
        self.low_confidence_threshold = low_confidence_threshold
        self.class_share_threshold = class_share_threshold
        self.brightness_baseline = brightness_baseline
        self.brightness_tolerance = brightness_tolerance
        self._records: deque[PredictionRecord] = deque(maxlen=window_size)

    def add(self, record: PredictionRecord) -> list[DriftAlert]:
        self._records.append(record)
        return self.check(record)

    def check(self, record: PredictionRecord) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []

        if record.confidence < self.low_confidence_threshold:
            alerts.append(
                DriftAlert(
                    code="low_confidence",
                    message=(
                        f"Prediction confidence {record.confidence:.3f} is below "
                        f"{self.low_confidence_threshold:.3f}"
                    ),
                )
            )

        brightness_delta = abs(record.image_stats.mean_brightness - self.brightness_baseline)
        if brightness_delta > self.brightness_tolerance:
            alerts.append(
                DriftAlert(
                    code="brightness_drift",
                    message=(
                        f"Image brightness {record.image_stats.mean_brightness:.3f} deviates "
                        f"from baseline {self.brightness_baseline:.3f}"
                    ),
                )
            )

        if len(self._records) >= max(10, self.window_size // 5):
            class_counts = Counter(entry.predicted_class for entry in self._records)
            total = sum(class_counts.values())
            dominant_class, dominant_count = class_counts.most_common(1)[0]
            dominant_share = dominant_count / total
            if dominant_share >= self.class_share_threshold:
                alerts.append(
                    DriftAlert(
                        code="class_distribution_skew",
                        message=(
                            f"Class '{dominant_class}' accounts for "
                            f"{dominant_share:.0%} of recent predictions"
                        ),
                    )
                )

        return alerts

    def summary(self) -> dict:
        if not self._records:
            return {
                "prediction_count": 0,
                "average_confidence": None,
                "class_distribution": {},
                "average_brightness": None,
                "low_confidence_rate": None,
            }

        confidences = [record.confidence for record in self._records]
        brightness_values = [record.image_stats.mean_brightness for record in self._records]
        class_counts = Counter(record.predicted_class for record in self._records)
        total = len(self._records)
        low_confidence = sum(
            1 for value in confidences if value < self.low_confidence_threshold
        )

        return {
            "prediction_count": total,
            "average_confidence": sum(confidences) / total,
            "class_distribution": dict(class_counts),
            "average_brightness": sum(brightness_values) / total,
            "low_confidence_rate": low_confidence / total,
        }
