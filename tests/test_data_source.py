"""Tests for S3/local dataset path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.data_source import (
    DatasetSourceError,
    ensure_s3_dataset_cached,
    parse_dataset_config,
    resolve_processed_root,
)
from training.utils import load_config


@pytest.fixture
def base_config() -> dict:
    config = load_config()
    config["dataset"] = {
        "source": "local",
        "bucket": "test-bucket",
        "prefix": "dataset/",
        "cache_dir": ".cache/dataset",
    }
    return config


def test_parse_dataset_config_defaults(base_config: dict) -> None:
    cfg = parse_dataset_config(base_config)
    assert cfg.source == "local"
    assert cfg.bucket == "test-bucket"
    assert cfg.prefix == "dataset/"
    assert cfg.processed_version == "v1"


def test_parse_dataset_config_env_override(
    base_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_SOURCE", "s3")
    cfg = parse_dataset_config(base_config)
    assert cfg.source == "s3"


def test_parse_dataset_config_invalid_source(base_config: dict) -> None:
    base_config["dataset"]["source"] = "ftp"
    with pytest.raises(DatasetSourceError, match="Invalid dataset source"):
        parse_dataset_config(base_config)


def test_resolve_processed_root_local_missing(base_config: dict) -> None:
    with pytest.raises(FileNotFoundError, match="Missing processed dataset"):
        resolve_processed_root(base_config)


def test_resolve_processed_root_local_found(
    base_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_root = tmp_path / "processed" / "v1"
    processed_root.mkdir(parents=True)
    (processed_root / "metadata.json").write_text("{}")

    base_config["paths"]["processed_data"] = str(tmp_path / "processed")

    resolved = resolve_processed_root(base_config)
    assert resolved == processed_root


def test_ensure_s3_dataset_cached_skips_when_marker_exists(
    base_config: dict, tmp_path: Path
) -> None:
    from training.data_source import DatasetConfig

    cache_dir = tmp_path / "cache"
    marker = cache_dir / "processed" / "v1" / "metadata.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")

    dataset_cfg = DatasetConfig(
        source="s3",
        bucket="test-bucket",
        prefix="dataset/",
        cache_dir=cache_dir,
        processed_version="v1",
        local_processed_root=tmp_path / "processed",
    )

    with patch("training.data_source._sync_s3_prefix") as mock_sync:
        result = ensure_s3_dataset_cached(dataset_cfg)

    assert result == cache_dir
    mock_sync.assert_not_called()


def test_sync_s3_prefix_downloads_objects(tmp_path: Path) -> None:
    from training.data_source import _sync_s3_prefix

    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "dataset/processed/v1/metadata.json"},
                {"Key": "dataset/processed/v1/train/crazing/img.jpg"},
            ]
        }
    ]

    with patch("boto3.client", return_value=mock_client):
        _sync_s3_prefix("test-bucket", "dataset/", tmp_path)

    assert (tmp_path / "processed" / "v1" / "metadata.json").exists()
    assert mock_client.download_file.call_count == 2


def test_sync_s3_prefix_empty_prefix_raises(tmp_path: Path) -> None:
    from training.data_source import _sync_s3_prefix

    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"Contents": []}]

    with (
        patch("boto3.client", return_value=mock_client),
        pytest.raises(DatasetSourceError, match="No objects found"),
    ):
        _sync_s3_prefix("test-bucket", "dataset/", tmp_path)


def test_config_loads_dataset_section() -> None:
    config = load_config()
    assert "dataset" in config
    assert config["dataset"]["source"] == "s3"
    assert config["dataset"]["bucket"] == "neu-cnn-surface-detect"
