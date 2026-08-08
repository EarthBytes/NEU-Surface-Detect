#!/usr/bin/env python3
"""Upload a processed dataset version to S3 for cloud training."""

from __future__ import annotations

import argparse
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from training.utils import load_config, resolve_path, setup_logging

logger = setup_logging(__name__)


def upload_processed_dataset(
    local_root: Path,
    bucket: str,
    prefix: str,
    version: str,
) -> int:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required. Install with: pip install boto3") from exc

    processed_dir = local_root / version
    metadata = processed_dir / "metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing processed dataset at {processed_dir}")

    prefix = prefix.strip("/")
    s3_prefix = f"{prefix}/processed/{version}"
    s3 = boto3.client("s3")

    uploaded = 0
    for path in processed_dir.rglob("*"):
        if not path.is_file():
            continue
        key = f"{s3_prefix}/{path.relative_to(processed_dir).as_posix()}"
        logger.debug("Uploading %s -> s3://%s/%s", path, bucket, key)
        s3.upload_file(str(path), bucket, key)
        uploaded += 1

    logger.info(
        "Uploaded %d file(s) to s3://%s/%s/",
        uploaded,
        bucket,
        s3_prefix,
    )
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload processed dataset to S3.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--version", type=str, default=None, help="Processed version directory name")
    parser.add_argument("--bucket", type=str, default=None, help="Override S3 bucket")
    parser.add_argument("--prefix", type=str, default=None, help="Override S3 dataset prefix")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    data_cfg = config["data"]
    version = args.version or data_cfg["processed_version"]
    bucket = args.bucket or dataset_cfg.get("bucket", "neu-cnn-surface-detect")
    prefix = args.prefix or dataset_cfg.get("prefix", "dataset/")
    local_root = resolve_path(config["paths"]["processed_data"])

    try:
        upload_processed_dataset(local_root, bucket, prefix, version)
    except (FileNotFoundError, RuntimeError, ClientError, BotoCoreError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
