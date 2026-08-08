"""Tests for MLflow enable/skip logic used by local and SageMaker training."""

from __future__ import annotations

import pytest

from training.mlflow_tracking import (
    is_mlflow_enabled,
    is_remote_tracking_uri,
    is_sagemaker_job,
)


def test_is_remote_tracking_uri() -> None:
    assert is_remote_tracking_uri("https://mlflow.example.com")
    assert is_remote_tracking_uri("http://127.0.0.1:5000")
    assert is_remote_tracking_uri("databricks")
    assert not is_remote_tracking_uri("sqlite:///models/mlflow.db")
    assert not is_remote_tracking_uri("file:./mlruns")


def test_is_sagemaker_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SM_TRAINING_ENV", raising=False)
    monkeypatch.delenv("TRAINING_JOB_NAME", raising=False)
    assert not is_sagemaker_job()

    monkeypatch.setenv("TRAINING_JOB_NAME", "neu-cnn-job-1")
    assert is_sagemaker_job()


def test_mlflow_disabled_on_sagemaker_with_local_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("SM_TRAINING_ENV", "{}")

    assert not is_mlflow_enabled("sqlite:///models/mlflow.db")


def test_mlflow_enabled_on_sagemaker_with_remote_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    monkeypatch.setenv("SM_TRAINING_ENV", "{}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")

    assert is_mlflow_enabled("sqlite:///models/mlflow.db")


def test_mlflow_enabled_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SM_TRAINING_ENV", "{}")
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    assert is_mlflow_enabled("sqlite:///models/mlflow.db")

    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    assert not is_mlflow_enabled("https://mlflow.example.com")


def test_sagemaker_environment_disables_local_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from training.aws_training import _sagemaker_environment

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)

    env = _sagemaker_environment()
    assert env["DATA_SOURCE"] == "s3"
    assert env["MLFLOW_ENABLED"] == "false"


def test_sagemaker_environment_passes_remote_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    from training.aws_training import _sagemaker_environment

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)

    env = _sagemaker_environment()
    assert env["MLFLOW_TRACKING_URI"] == "https://mlflow.example.com"
    assert env["MLFLOW_ENABLED"] == "true"
