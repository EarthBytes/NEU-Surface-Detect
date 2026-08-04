"""Resolve training dataset paths from local disk or AWS S3."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from training.utils import resolve_path

logger = logging.getLogger(__name__)

VALID_SOURCES = frozenset({"local", "s3"})


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    bucket: str
    prefix: str
    cache_dir: Path
    processed_version: str
    local_processed_root: Path


class DatasetSourceError(RuntimeError):
    """Raised when the configured dataset source cannot be accessed."""


def _normalise_prefix(prefix: str) -> str:
    cleaned = prefix.strip("/")
    return f"{cleaned}/" if cleaned else ""


def parse_dataset_config(config: dict) -> DatasetConfig:
    """Build dataset settings from YAML, with optional ``DATA_SOURCE`` env override."""
    data_cfg = config["data"]
    dataset_cfg = config.get("dataset", {})
    paths_cfg = config["paths"]

    source = os.environ.get("DATA_SOURCE", dataset_cfg.get("source", "local")).lower()
    if source not in VALID_SOURCES:
        raise DatasetSourceError(
            f"Invalid dataset source {source!r}. Expected one of: {sorted(VALID_SOURCES)}"
        )

    cache_relative = dataset_cfg.get("cache_dir", ".cache/dataset")
    cache_dir = resolve_path(cache_relative)

    return DatasetConfig(
        source=source,
        bucket=dataset_cfg.get("bucket", "neu-cnn-surface-detect"),
        prefix=_normalise_prefix(dataset_cfg.get("prefix", "dataset/")),
        cache_dir=cache_dir,
        processed_version=data_cfg["processed_version"],
        local_processed_root=resolve_path(paths_cfg["processed_data"]),
    )


def _s3_processed_marker(dataset_cfg: DatasetConfig, cache_root: Path) -> Path:
    """Marker file after syncing the full S3 dataset prefix (contains processed/)."""
    return cache_root / "processed" / dataset_cfg.processed_version / "metadata.json"


def _local_processed_marker(dataset_cfg: DatasetConfig) -> Path:
    """Marker file for a locally preprocessed dataset."""
    return (
        dataset_cfg.local_processed_root
        / dataset_cfg.processed_version
        / "metadata.json"
    )


def _sync_s3_prefix(bucket: str, prefix: str, destination: Path) -> None:
    """Download all objects under an S3 prefix into *destination*."""
    try:
        import boto3
    except ImportError as exc:
        raise DatasetSourceError(
            "boto3 is required for S3 dataset loading. Install it with: pip install boto3"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    prefix = _normalise_prefix(prefix)

    try:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    except NoCredentialsError as exc:
        raise DatasetSourceError(
            "AWS credentials not found. Configure credentials via environment variables, "
            "~/.aws/credentials, or an IAM role attached to the instance."
        ) from exc
    except (ClientError, BotoCoreError) as exc:
        raise DatasetSourceError(
            f"Failed to connect to S3 bucket {bucket!r}: {exc}"
        ) from exc

    downloaded = 0
    try:
        for page in pages:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith("/"):
                    continue

                relative_key = key[len(prefix) :] if key.startswith(prefix) else key
                if not relative_key:
                    continue

                local_path = destination / relative_key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug("Downloading s3://%s/%s -> %s", bucket, key, local_path)
                s3.download_file(bucket, key, str(local_path))
                downloaded += 1
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        if error_code in {"NoSuchBucket", "404"}:
            raise DatasetSourceError(
                f"S3 bucket {bucket!r} does not exist or is not accessible."
            ) from exc
        if error_code in {"AccessDenied", "403"}:
            raise DatasetSourceError(
                f"Access denied reading s3://{bucket}/{prefix}. Check IAM permissions."
            ) from exc
        raise DatasetSourceError(
            f"Failed to download s3://{bucket}/{prefix}: {exc}"
        ) from exc
    except BotoCoreError as exc:
        raise DatasetSourceError(
            f"Failed to download s3://{bucket}/{prefix}: {exc}"
        ) from exc

    if downloaded == 0:
        raise DatasetSourceError(
            f"No objects found at s3://{bucket}/{prefix}. "
            "Verify the bucket, prefix, and that the dataset has been uploaded."
        )

    logger.info(
        "Synced %d object(s) from s3://%s/%s to %s",
        downloaded,
        bucket,
        prefix,
        destination,
    )


def ensure_s3_dataset_cached(dataset_cfg: DatasetConfig) -> Path:
    """Download the S3 dataset into the cache if not already present."""
    marker = _s3_processed_marker(dataset_cfg, dataset_cfg.cache_dir)
    if marker.is_file():
        logger.info("Using cached S3 dataset at %s", dataset_cfg.cache_dir)
        return dataset_cfg.cache_dir

    logger.info(
        "Downloading dataset from s3://%s/%s ...",
        dataset_cfg.bucket,
        dataset_cfg.prefix,
    )
    _sync_s3_prefix(dataset_cfg.bucket, dataset_cfg.prefix, dataset_cfg.cache_dir)

    if not marker.is_file():
        raise DatasetSourceError(
            f"S3 sync completed but expected processed data at {marker}. "
            f"Ensure s3://{dataset_cfg.bucket}/{dataset_cfg.prefix} contains "
            f"processed/{dataset_cfg.processed_version}/metadata.json."
        )

    return dataset_cfg.cache_dir


def resolve_processed_root(config: dict) -> Path:
    """Return the local path to the versioned processed dataset directory."""
    dataset_cfg = parse_dataset_config(config)

    if dataset_cfg.source == "local":
        processed_root = (
            dataset_cfg.local_processed_root / dataset_cfg.processed_version
        )
        if not _local_processed_marker(dataset_cfg).is_file():
            raise FileNotFoundError(
                f"Missing processed dataset at {processed_root}. "
                "Run data_ingestion.preprocess or switch dataset.source to s3."
            )
        logger.info("Using local processed dataset at %s", processed_root)
        return processed_root

    dataset_root = ensure_s3_dataset_cached(dataset_cfg)
    processed_root = dataset_root / "processed" / dataset_cfg.processed_version
    logger.info("Using S3-backed processed dataset at %s", processed_root)
    return processed_root
