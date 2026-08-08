"""Tests for AWS training artifact download and config parsing."""

from __future__ import annotations

import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.aws_training import (
    AwsTrainingError,
    download_model_artifact,
    parse_aws_training_config,
    parse_s3_uri,
)
from training.utils import load_config


def test_parse_s3_uri() -> None:
    bucket, key = parse_s3_uri("s3://my-bucket/path/to/model.tar.gz")
    assert bucket == "my-bucket"
    assert key == "path/to/model.tar.gz"


def test_parse_s3_uri_invalid() -> None:
    with pytest.raises(AwsTrainingError, match="Invalid S3 URI"):
        parse_s3_uri("https://example.com/model.tar.gz")


def test_parse_aws_training_config() -> None:
    config = load_config()
    aws_cfg = parse_aws_training_config(config)
    assert aws_cfg.instance_type == "ml.g4dn.xlarge"
    assert aws_cfg.bucket == "neu-cnn-surface-detect"
    assert aws_cfg.checkpoint_name == "best_model.pt"


def test_parse_aws_training_config_role_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/TestRole")
    aws_cfg = parse_aws_training_config(config)
    assert aws_cfg.sagemaker_role_arn == "arn:aws:iam::123456789012:role/TestRole"


def test_download_model_artifact_extracts_checkpoint(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        checkpoint = tmp_path / "best_model.pt"
        checkpoint.write_bytes(b"checkpoint-bytes")
        archive.add(checkpoint, arcname="best_model.pt")

    local_checkpoint = tmp_path / "output" / "best_model.pt"
    mock_client = MagicMock()

    def fake_download(bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(archive_path.read_bytes())

    mock_client.download_file.side_effect = fake_download

    with patch("training.aws_training._get_s3_client", return_value=mock_client):
        result = download_model_artifact(
            "s3://test-bucket/output/model.tar.gz",
            local_checkpoint,
            region="eu-west-1",
        )

    assert result == local_checkpoint
    assert local_checkpoint.read_bytes() == b"checkpoint-bytes"
