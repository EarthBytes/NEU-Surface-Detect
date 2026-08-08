"""Launch SageMaker GPU training and download the model locally."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from training.utils import PROJECT_ROOT, load_config, resolve_path

logger = logging.getLogger(__name__)

S3_URI_PATTERN = re.compile(r"^s3://([^/]+)/(.+)$")
SOURCE_IGNORE_PATTERNS = [
    ".cache",
    ".env",
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dataset",
    "docs",
    "models",
]


class AwsTrainingError(RuntimeError):
    """Raised when AWS training orchestration fails."""


@dataclass(frozen=True)
class AwsTrainingConfig:
    region: str
    bucket: str
    output_prefix: str
    sagemaker_role_arn: str | None
    instance_type: str
    instance_count: int
    max_runtime_seconds: int
    pytorch_version: str
    python_version: str
    checkpoint_name: str
    local_checkpoint_dir: Path


def parse_aws_training_config(config: dict) -> AwsTrainingConfig:
    """Build AWS training settings from YAML with optional environment overrides."""
    aws_cfg = config.get("aws", {})
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN") or aws_cfg.get("sagemaker_role_arn")
    region = os.environ.get("AWS_DEFAULT_REGION") or aws_cfg.get("region", "eu-west-1")

    return AwsTrainingConfig(
        region=region,
        bucket=aws_cfg.get("bucket", config.get("dataset", {}).get("bucket", "neu-cnn-surface-detect")),
        output_prefix=_normalise_prefix(aws_cfg.get("output_prefix", "training-output/")),
        sagemaker_role_arn=role_arn,
        instance_type=aws_cfg.get("instance_type", "ml.g4dn.xlarge"),
        instance_count=int(aws_cfg.get("instance_count", 1)),
        max_runtime_seconds=int(aws_cfg.get("max_runtime_seconds", 3600)),
        pytorch_version=str(aws_cfg.get("pytorch_version", "2.0.0")),
        python_version=str(aws_cfg.get("python_version", "py310")),
        checkpoint_name=train_cfg["checkpoint_name"],
        local_checkpoint_dir=resolve_path(paths_cfg["checkpoints"]),
    )


def _normalise_prefix(prefix: str) -> str:
    cleaned = prefix.strip("/")
    return f"{cleaned}/" if cleaned else ""


def _require_role(aws_cfg: AwsTrainingConfig) -> str:
    if not aws_cfg.sagemaker_role_arn:
        raise AwsTrainingError(
            "SageMaker execution role is required. Set SAGEMAKER_ROLE_ARN or "
            "aws.sagemaker_role_arn in training/config.yaml."
        )
    return aws_cfg.sagemaker_role_arn


def _get_s3_client(region: str) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise AwsTrainingError("boto3 is required. Install with: pip install boto3") from exc

    try:
        return boto3.client("s3", region_name=region)
    except NoCredentialsError as exc:
        raise AwsTrainingError(
            "AWS credentials not found. Configure credentials before launching training."
        ) from exc


def parse_s3_uri(uri: str) -> tuple[str, str]:
    match = S3_URI_PATTERN.match(uri)
    if not match:
        raise AwsTrainingError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)


def download_model_artifact(model_s3_uri: str, local_checkpoint: Path, region: str) -> Path:
    """Download a SageMaker ``model.tar.gz`` and extract ``best_model.pt`` locally."""
    bucket, key = parse_s3_uri(model_s3_uri)
    s3 = _get_s3_client(region)
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="neu-sagemaker-model-") as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        try:
            logger.info("Downloading model artifact from %s", model_s3_uri)
            s3.download_file(bucket, key, str(tar_path))
        except ClientError as exc:
            raise AwsTrainingError(
                f"Failed to download model artifact from {model_s3_uri}: {exc}"
            ) from exc

        with tarfile.open(tar_path, mode="r:gz") as archive:
            archive.extractall(path=tmp)

        candidates = list(Path(tmp).rglob("best_model.pt"))
        if not candidates and local_checkpoint.name != "best_model.pt":
            candidates = list(Path(tmp).rglob(local_checkpoint.name))
        if not candidates:
            raise AwsTrainingError(
                f"No checkpoint named {local_checkpoint.name!r} found inside {model_s3_uri}"
            )

        shutil.copy2(candidates[0], local_checkpoint)

    logger.info("Saved trained model locally to %s", local_checkpoint)
    return local_checkpoint


def _get_boto3_session(region: str):
    import boto3

    return boto3.Session(region_name=region)


def _build_hyperparameters(epochs_override: int | None) -> dict[str, str]:
    if epochs_override is None:
        return {}
    return {"epochs": str(epochs_override)}


def _sagemaker_environment() -> dict[str, str]:
    """Env vars injected into the training container.

    Local sqlite MLflow tracking is ephemeral on SageMaker and requires
    sqlalchemy (not shipped with mlflow-skinny). Disable by default; pass through
    a remote MLFLOW_TRACKING_URI if the launcher environment has one.
    """
    env = {"DATA_SOURCE": "s3"}
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri and tracking_uri.strip().lower().startswith(("http://", "https://", "databricks")):
        env["MLFLOW_TRACKING_URI"] = tracking_uri.strip()
        env["MLFLOW_ENABLED"] = os.environ.get("MLFLOW_ENABLED", "true")
    else:
        env["MLFLOW_ENABLED"] = os.environ.get("MLFLOW_ENABLED", "false")
    return env


def _resolve_model_uri_v3(trainer: Any, wait: bool) -> str:
    training_job = trainer._latest_training_job
    if training_job is None:
        raise AwsTrainingError("SageMaker did not return a training job handle.")

    if wait:
        training_job.refresh()

    model_artifacts = training_job.model_artifacts
    if model_artifacts is None or not model_artifacts.s3_model_artifacts:
        raise AwsTrainingError("SageMaker job finished but no model artifact URI was returned.")

    return model_artifacts.s3_model_artifacts


def _launch_with_sagemaker_v3(
    aws_cfg: AwsTrainingConfig,
    role: str,
    hyperparameters: dict[str, str],
    job_name: str | None,
    wait: bool,
) -> str:
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition

    output_path = f"s3://{aws_cfg.bucket}/{aws_cfg.output_prefix}"
    training_image = image_uris.retrieve(
        framework="pytorch",
        region=aws_cfg.region,
        version=aws_cfg.pytorch_version,
        py_version=aws_cfg.python_version,
        instance_type=aws_cfg.instance_type,
        image_scope="training",
    )

    logger.info("Submitting SageMaker v3 job on %s (region=%s)", aws_cfg.instance_type, aws_cfg.region)
    logger.info("Training output will be written to %s", output_path)

    session = Session(boto_session=_get_boto3_session(aws_cfg.region))
    trainer = ModelTrainer(
        training_image=training_image,
        source_code=SourceCode(
            source_dir=str(PROJECT_ROOT),
            entry_script="training/sagemaker_entry.py",
            requirements="requirements.txt",
            ignore_patterns=SOURCE_IGNORE_PATTERNS,
        ),
        compute=Compute(
            instance_type=aws_cfg.instance_type,
            instance_count=aws_cfg.instance_count,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=aws_cfg.max_runtime_seconds),
        output_data_config=OutputDataConfig(s3_output_path=output_path),
        role=role,
        base_job_name=job_name,
        hyperparameters=hyperparameters or None,
        environment=_sagemaker_environment(),
        sagemaker_session=session,
    )

    try:
        trainer.train(wait=wait)
    except (ClientError, BotoCoreError) as exc:
        raise AwsTrainingError(f"SageMaker training job failed: {exc}") from exc

    return _resolve_model_uri_v3(trainer, wait=wait)


def _launch_with_sagemaker_v2(
    aws_cfg: AwsTrainingConfig,
    role: str,
    hyperparameters: dict[str, str],
    job_name: str | None,
    wait: bool,
) -> str:
    import sagemaker
    from sagemaker.pytorch import PyTorch

    output_path = f"s3://{aws_cfg.bucket}/{aws_cfg.output_prefix}"

    logger.info("Submitting SageMaker v2 job on %s (region=%s)", aws_cfg.instance_type, aws_cfg.region)
    logger.info("Training output will be written to %s", output_path)

    session = sagemaker.Session(boto_session=_get_boto3_session(aws_cfg.region))
    estimator = PyTorch(
        entry_point="training/sagemaker_entry.py",
        source_dir=str(PROJECT_ROOT),
        dependencies=[str(PROJECT_ROOT / "requirements.txt")],
        role=role,
        instance_type=aws_cfg.instance_type,
        instance_count=aws_cfg.instance_count,
        framework_version=aws_cfg.pytorch_version,
        py_version=aws_cfg.python_version,
        hyperparameters=hyperparameters or None,
        environment=_sagemaker_environment(),
        max_run=aws_cfg.max_runtime_seconds,
        output_path=output_path,
        sagemaker_session=session,
        disable_profiler=True,
    )

    try:
        estimator.fit(wait=wait, job_name=job_name)
    except (ClientError, BotoCoreError) as exc:
        raise AwsTrainingError(f"SageMaker training job failed: {exc}") from exc

    model_uri = estimator.model_data
    if not model_uri or model_uri == "None":
        raise AwsTrainingError("SageMaker job finished but no model artifact URI was returned.")
    return model_uri


def launch_sagemaker_training(
    config_path: Path | None = None,
    epochs_override: int | None = None,
    job_name: str | None = None,
    wait: bool = True,
) -> tuple[str, Path | None]:
    """Submit a SageMaker training job and optionally download the model locally."""
    config = load_config(config_path)
    aws_cfg = parse_aws_training_config(config)
    role = _require_role(aws_cfg)
    hyperparameters = _build_hyperparameters(epochs_override)

    try:
        model_uri = _launch_with_sagemaker_v3(
            aws_cfg,
            role,
            hyperparameters,
            job_name,
            wait,
        )
    except ImportError as v3_error:
        try:
            model_uri = _launch_with_sagemaker_v2(
                aws_cfg,
                role,
                hyperparameters,
                job_name,
                wait,
            )
        except ImportError as exc:
            import sys

            raise AwsTrainingError(
                "Could not import a compatible SageMaker training API. "
                f"SageMaker v3 error: {v3_error}. "
                f"SageMaker v2 error: {exc}. "
                f"Install/update with: {sys.executable} -m pip install -r requirements-aws.txt"
            ) from exc

    local_checkpoint = aws_cfg.local_checkpoint_dir / aws_cfg.checkpoint_name
    if wait:
        download_model_artifact(model_uri, local_checkpoint, aws_cfg.region)
        return model_uri, local_checkpoint

    logger.info("Job submitted. Download the model artifact from %s when complete.", model_uri)
    return model_uri, None


def download_completed_job_model(
    model_s3_uri: str,
    config_path: Path | None = None,
) -> Path:
    """Download a checkpoint from a completed SageMaker job artifact URI."""
    config = load_config(config_path)
    aws_cfg = parse_aws_training_config(config)
    local_checkpoint = aws_cfg.local_checkpoint_dir / aws_cfg.checkpoint_name
    return download_model_artifact(model_s3_uri, local_checkpoint, aws_cfg.region)
