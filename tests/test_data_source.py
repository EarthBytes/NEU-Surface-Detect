"""Tests for S3/local dataset path resolution."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.data_source import (
    DatasetSourceError,
    ensure_s3_dataset_cached,
    parse_dataset_config,
    processed_dataset_is_complete,
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


def test_resolve_processed_root_local_missing(base_config: dict, tmp_path: Path) -> None:
    base_config["paths"]["processed_data"] = str(tmp_path / "missing-processed")
    with pytest.raises(FileNotFoundError, match="Missing processed dataset"):
        resolve_processed_root(base_config)


def test_resolve_processed_root_local_found(
    base_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_root = tmp_path / "processed" / "v1"
    _write_complete_processed_cache(processed_root)

    base_config["paths"]["processed_data"] = str(tmp_path / "processed")

    resolved = resolve_processed_root(base_config)
    assert resolved == processed_root


def _write_complete_processed_cache(processed_root: Path) -> None:
    metadata = {
        "class_names": ["crazing", "inclusion"],
        "splits": {
            "train": {"crazing": 1, "inclusion": 1},
            "validation": {"crazing": 1, "inclusion": 0},
            "test": {"crazing": 0, "inclusion": 1},
        },
    }
    processed_root.mkdir(parents=True, exist_ok=True)
    (processed_root / "metadata.json").write_text(json.dumps(metadata))

    for split_name, class_counts in metadata["splits"].items():
        for class_name, count in class_counts.items():
            if count == 0:
                continue
            class_dir = processed_root / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                (class_dir / f"{class_name}_{index}.jpg").write_bytes(b"jpg")


def test_processed_dataset_is_complete_detects_missing_split(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed" / "v1"
    _write_complete_processed_cache(processed_root)
    assert processed_dataset_is_complete(processed_root)

    shutil.rmtree(processed_root / "validation")
    assert not processed_dataset_is_complete(processed_root)


def test_processed_dataset_is_complete_detects_partial_class(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed" / "v1"
    _write_complete_processed_cache(processed_root)

    partial = processed_root / "train" / "patches"
    partial.mkdir(parents=True)
    (partial / "patches_0.jpg").write_bytes(b"jpg")

    metadata = json.loads((processed_root / "metadata.json").read_text())
    metadata["class_names"].append("patches")
    metadata["splits"]["train"]["patches"] = 2
    (processed_root / "metadata.json").write_text(json.dumps(metadata))

    assert not processed_dataset_is_complete(processed_root)


def test_ensure_s3_dataset_cached_skips_when_cache_complete(
    base_config: dict, tmp_path: Path
) -> None:
    from training.data_source import DatasetConfig

    cache_dir = tmp_path / "cache"
    processed_root = cache_dir / "processed" / "v1"
    _write_complete_processed_cache(processed_root)

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


def test_ensure_s3_dataset_cached_resyncs_incomplete_cache(
    base_config: dict, tmp_path: Path
) -> None:
    from training.data_source import DatasetConfig

    cache_dir = tmp_path / "cache"
    processed_root = cache_dir / "processed" / "v1"
    processed_root.mkdir(parents=True)
    (processed_root / "metadata.json").write_text("{}")

    dataset_cfg = DatasetConfig(
        source="s3",
        bucket="test-bucket",
        prefix="dataset/",
        cache_dir=cache_dir,
        processed_version="v1",
        local_processed_root=tmp_path / "processed",
    )

    with patch("training.data_source._sync_s3_prefix") as mock_sync:
        with pytest.raises(DatasetSourceError, match="incomplete"):
            ensure_s3_dataset_cached(dataset_cfg)

    mock_sync.assert_called_once()


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

    def fake_download(bucket: str, key: str, filename: str) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"jpg")

    mock_client.download_file.side_effect = fake_download

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
