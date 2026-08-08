#!/usr/bin/env python3
"""Launch training on AWS SageMaker and save the model locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.aws_training import (
    AwsTrainingError,
    download_completed_job_model,
    launch_sagemaker_training,
)
from training.utils import load_dotenv_file, setup_logging

load_dotenv_file()
logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train on AWS SageMaker GPU and download best_model.pt to models/checkpoints/. "
            "Requires AWS credentials and SAGEMAKER_ROLE_ARN."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--job-name", type=str, default=None, help="Optional SageMaker job name")
    parser.add_argument(
        "--download-only",
        type=str,
        default=None,
        metavar="S3_URI",
        help="Skip training; download model.tar.gz from a completed job (s3://.../model.tar.gz)",
    )
    args = parser.parse_args()

    try:
        if args.download_only:
            local_path = download_completed_job_model(args.download_only, args.config)
            logger.info("Model saved locally to %s", local_path)
            return 0

        model_uri, local_path = launch_sagemaker_training(
            config_path=args.config,
            epochs_override=args.epochs,
            job_name=args.job_name,
            wait=True,
        )
        logger.info("SageMaker model artifact: %s", model_uri)
        if local_path:
            logger.info("Local checkpoint: %s", local_path)
    except AwsTrainingError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
