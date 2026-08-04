"""Compute image statistics and build prediction log records."""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageStats:
    width: int
    height: int
    mean_brightness: float
    contrast: float


@dataclass(frozen=True)
class PredictionRecord:
    timestamp: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    image_stats: ImageStats

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["image_stats"] = asdict(self.image_stats)
        return payload


def compute_image_stats(image_bytes: bytes) -> ImageStats:
    with Image.open(io.BytesIO(image_bytes)) as img:
        greyscale = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    return ImageStats(
        width=int(greyscale.shape[1]),
        height=int(greyscale.shape[0]),
        mean_brightness=float(greyscale.mean()),
        contrast=float(greyscale.std()),
    )


def build_prediction_record(
    image_bytes: bytes,
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    *,
    timestamp: datetime | None = None,
) -> PredictionRecord:
    moment = timestamp or datetime.now(timezone.utc)
    return PredictionRecord(
        timestamp=moment.isoformat(),
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        image_stats=compute_image_stats(image_bytes),
    )
