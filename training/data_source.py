"""Resolve training dataset paths from local disk or AWS S3."""

from __future__ import annotations

import json
import logging
import os
import shutil
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


def _processed_root_from_cache(cache_root: Path, version: str) -> Path:
    return cache_root / "processed" / version


def _count_class_images(split_dir: Path, class_name: str) -> int:
    class_dir = split_dir / class_name
    if not class_dir.is_dir():
        return 0
    return len(list(class_dir.glob("*.jpg")))


def _count_split_on_disk(processed_root: Path, split_name: str, class_names: list[str]) -> dict[str, int]:
    split_dir = processed_root / split_name
    return {class_name: _count_class_images(split_dir, class_name) for class_name in class_names}


def processed_dataset_is_complete(processed_root: Path) -> bool:
    """Return True when on-disk image counts match metadata.json split counts."""
    metadata_path = processed_root / "metadata.json"
    if not metadata_path.is_file():
        return False

    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False

    expected_splits: dict[str, dict[str, int]] = metadata.get("splits", {})
    class_names: list[str] = metadata.get("class_names", [])
    if not expected_splits or not class_names:
        return False

    for split_name, expected_counts in expected_splits.items():
        split_dir = processed_root / split_name
        if not split_dir.is_dir():
            logger.debug("Missing split directory: %s", split_dir)
            return False

        actual_counts = _count_split_on_disk(processed_root, split_name, class_names)
        for class_name, expected in expected_counts.items():
            actual = actual_counts.get(class_name, 0)
            if actual != expected:
                logger.debug(
                    "Cache mismatch at %s/%s: expected %d images, found %d",
                    split_name,
                    class_name,
                    expected,
                    actual,
                )
                return False

    return True


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
    processed_root = _processed_root_from_cache(
        dataset_cfg.cache_dir,
        dataset_cfg.processed_version,
    )

    if processed_dataset_is_complete(processed_root):
        logger.info("Using cached S3 dataset at %s", dataset_cfg.cache_dir)
        return dataset_cfg.cache_dir

    if processed_root.exists():
        logger.warning(
            "Incomplete cached dataset at %s; re-downloading from s3://%s/%s",
            processed_root,
            dataset_cfg.bucket,
            dataset_cfg.prefix,
        )
        shutil.rmtree(processed_root.parent)

    logger.info(
        "Downloading dataset from s3://%s/%s ...",
        dataset_cfg.bucket,
        dataset_cfg.prefix,
    )
    _sync_s3_prefix(dataset_cfg.bucket, dataset_cfg.prefix, dataset_cfg.cache_dir)

    if not processed_dataset_is_complete(processed_root):
        marker = _s3_processed_marker(dataset_cfg, dataset_cfg.cache_dir)
        raise DatasetSourceError(
            f"S3 sync completed but processed dataset at {processed_root} is incomplete. "
            f"Expected a complete layout with metadata at {marker}."
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
        if not processed_dataset_is_complete(processed_root):
            raise DatasetSourceError(
                f"Local processed dataset at {processed_root} is incomplete. "
                "Re-run: python -m data_ingestion.preprocess --overwrite"
            )
        logger.info("Using local processed dataset at %s", processed_root)
        return processed_root

    dataset_root = ensure_s3_dataset_cached(dataset_cfg)
    processed_root = dataset_root / "processed" / dataset_cfg.processed_version
    logger.info("Using S3-backed processed dataset at %s", processed_root)
    return processed_root
