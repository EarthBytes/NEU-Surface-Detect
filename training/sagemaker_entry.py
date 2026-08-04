#!/usr/bin/env python3
"""SageMaker training entry point — runs on the AWS GPU instance, not locally."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# SageMaker sets SM_MODEL_DIR; training reads dataset from S3 via IAM role.
os.environ.setdefault("DATA_SOURCE", "s3")

from training.train import train  # noqa: E402


def export_checkpoint_for_sagemaker(checkpoint_path: Path) -> Path | None:
    """Copy the trained checkpoint into SM_MODEL_DIR so SageMaker uploads it to S3."""
    sm_model_dir = os.environ.get("SM_MODEL_DIR")
    if not sm_model_dir:
        return None

    destination = Path(sm_model_dir) / checkpoint_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination)
    print(f"Exported checkpoint to SageMaker model directory: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args, _unknown = parser.parse_known_args()

    checkpoint = train(epochs_override=args.epochs)
    export_checkpoint_for_sagemaker(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
