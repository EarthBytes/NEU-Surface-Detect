"""Create a minimal model checkpoint for CI and local API tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.model import build_model

DEFAULT_METADATA = {
    "version": "v1",
    "image_size": 224,
    "colour_mode": "greyscale",
    "normalisation": {"mean": 0.5045, "std": 0.2075},
    "class_names": [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ],
}


def create_checkpoint(output_path: Path, num_classes: int = 6) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(num_classes)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "val_accuracy": 0.0,
            "metadata": DEFAULT_METADATA,
            "run_id": None,
        },
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixture checkpoint for tests and CI.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/checkpoints/best_model.pt"),
        help="Path for the generated checkpoint",
    )
    args = parser.parse_args()
    path = create_checkpoint(args.output)
    print(f"Wrote fixture checkpoint to {path}")


if __name__ == "__main__":
    main()
