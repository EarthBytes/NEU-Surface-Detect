"""Load the trained model and run predictions on uploaded images."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from data_ingestion.config import CLASS_NAMES
from training.dataset import build_transforms
from training.model import build_model
from training.utils import get_device, load_config, resolve_path

logger = logging.getLogger(__name__)


class DefectPredictor:
    def __init__(self, checkpoint_path: Path, config_path: Path | None = None) -> None:
        self.device = get_device()
        self.config = load_config(config_path)
        self.class_names = CLASS_NAMES

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        metadata = checkpoint.get("metadata")
        if metadata is None:
            raise ValueError("Checkpoint is missing preprocessing metadata")

        self.metadata = metadata
        self.transform = build_transforms(metadata, augment=False, aug_cfg={})
        self.model = self._load_model_from_checkpoint(checkpoint)

        val_accuracy = checkpoint.get("val_accuracy")
        logger.info("Loaded model from %s", checkpoint_path)
        if val_accuracy is not None:
            logger.info("Checkpoint validation accuracy: %.4f", val_accuracy)

    def _load_model_from_checkpoint(self, checkpoint: dict) -> torch.nn.Module:
        model = build_model(self.config["model"]["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_bytes: bytes) -> dict[str, float | str | dict[str, float]]:
        with Image.open(io.BytesIO(image_bytes)) as img:
            tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()

        predicted_index = int(probabilities.argmax())
        confidence = float(probabilities[predicted_index])
        probability_map = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.class_names)
        }

        return {
            "predicted_class": self.class_names[predicted_index],
            "confidence": confidence,
            "probabilities": probability_map,
        }


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    config_path: Path | None,
) -> Path:
    config = load_config(config_path)

    if checkpoint_path:
        ckpt = Path(checkpoint_path)
    else:
        ckpt = resolve_path(config["inference"]["checkpoint"])

    if ckpt.exists():
        return ckpt

    search_dirs: list[Path] = []
    tmpdir = os.getenv("TMPDIR") or "/tmp"
    search_dirs.append(Path(tmpdir))
    search_dirs.append(Path.cwd())

    candidates: list[Path] = []
    for root in search_dirs:
        if root.exists():
            candidates.extend(root.rglob("best_model.pt"))

    if candidates:
        resolved = candidates[0]
        logger.info("Found checkpoint at %s (fallback search)", resolved)
        return resolved

    raise FileNotFoundError(f"Checkpoint not found at {ckpt}")


def create_predictor(
    checkpoint_path: str | Path | None = None,
    config_path: Path | None = None,
) -> DefectPredictor:
    resolved = _resolve_checkpoint_path(checkpoint_path, config_path)
    return DefectPredictor(resolved, config_path)
