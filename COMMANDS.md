# Commands Reference

Step-by-step commands to run the NEU-Surface-Detect pipeline from scratch. Run each section in order the first time; later you can jump to the step you need.

All commands assume you are in the project root and using the virtual environment:

```bash
cd /path/to/NEU-Surface-Detect
source .venv/bin/activate
```

---

## 1. First-time setup

```bash
# Create venv and install runtime + dev dependencies
bash scripts/setup_env.sh
source .venv/bin/activate

# Verify Python, packages, and project layout
python scripts/verify_env.py

# Optional: AWS SageMaker launcher extras
pip install -r requirements-aws.txt
```

Create `.env` in the project root (never commit this file):

```env
SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/neu-cnn-sagemaker-role
AWS_DEFAULT_REGION=eu-west-1
```

Configure AWS credentials separately (`aws configure` or `~/.aws/credentials`).

---

## 2. Download and prepare data

```bash
# Download NEU-DET from Kaggle (requires kaggle.json in ~/.kaggle/)
python -m data_ingestion.download_neu

# Organise raw files into train/validation layout (if needed)
python -m data_ingestion.organise_dataset

# Inspect raw dataset statistics
python -m data_ingestion.inspect_dataset

# Preprocess: resize, normalise, split → dataset/processed/v1/
python -m data_ingestion.preprocess

# Upload processed data to S3 (for cloud training)
python -m data_ingestion.upload_to_s3 --version v1
```

Expected output after preprocess:

```text
dataset/processed/v1/
  metadata.json
  train/          # 6 class folders
  validation/     # 6 class folders
  test/           # 6 class folders
```

---

## 3. Train the model

### Option A — Local training (uses laptop CPU/GPU)

```bash
DATA_SOURCE=local python -m training.train
```

### Option B — AWS SageMaker GPU training

```bash
python -m training.launch_aws_training

# Cheaper smoke test (2 epochs)
python -m training.launch_aws_training --epochs 2 --job-name neu-smoke-test
```

On success, the checkpoint is saved to `models/checkpoints/best_model.pt`.

Download a completed job manually:

```bash
python -m training.launch_aws_training \
  --download-only s3://neu-cnn-surface-detect/training-output/<job-name>/output/model.tar.gz
```

---

## 4. Evaluate the model

```bash
# Default: sync dataset from S3 cache, evaluate test split
python -m training.evaluate

# Use local processed data instead
DATA_SOURCE=local python -m training.evaluate

# Specific checkpoint
python -m training.evaluate --checkpoint models/checkpoints/best_model.pt
```

Results: `models/evaluation/metrics.json`, `confusion_matrix.png`, `confusion_matrix.csv`

---

## 5. MLflow experiment tracking

```bash
# Import a SageMaker checkpoint into MLflow (first time after cloud training)
python -m training.log_run --run-name sagemaker-best-model --update-checkpoint

# Register and promote model
python -m training.register_model --stage Staging
python -m training.register_model --stage Production

# Open local MLflow UI
bash scripts/start_mlflow_ui.sh
# → http://127.0.0.1:5000
```

If MLflow DB migration fails after upgrading from mlflow-skinny:

```bash
mv models/mlflow.db models/mlflow.db.bak
python -m training.log_run --update-checkpoint
```

---

## 6. Serve predictions (API)

```bash
python -m inference.serve
```

Test endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Predict on an image
curl -F "file=@path/to/defect.jpg" http://localhost:8000/predict

# Monitoring summary
curl http://localhost:8000/monitoring/summary
```

Override checkpoint:

```bash
MODEL_CHECKPOINT=models/checkpoints/best_model.pt python -m inference.serve
```

---

## 7. Docker

```bash
# Build
docker build -f inference/Dockerfile -t neu-defect-api .

# Run
docker run -p 8000:8000 neu-defect-api

# Health check
curl http://localhost:8000/health
```

---

## 8. Retraining loop

```bash
# Full loop: SageMaker train → compare → promote if better
python -m training.retrain --launch-aws --promote --job-name neu-retrain-v1

# Compare two existing checkpoints only
python -m training.retrain --compare-only \
  --champion models/checkpoints/best_model.pt \
  --challenger models/checkpoints/challenger_model.pt

# Standalone comparison
python -m training.compare_models \
  --challenger models/checkpoints/challenger_model.pt \
  --champion models/checkpoints/best_model.pt
```

Low-confidence production images are archived to `dataset/feedback/inbox/` when monitoring is enabled.

---

## 9. Development and CI

```bash
# Lint
ruff check .

# Tests
pytest tests/ -v

# Create minimal checkpoint for CI/Docker tests
python tests/create_fixture_checkpoint.py
```

---

## 10. Environment variables reference

| Variable | Purpose |
|----------|---------|
| `DATA_SOURCE=local` | Read dataset from `dataset/processed/` |
| `DATA_SOURCE=s3` | Sync dataset from S3 into `.cache/dataset/` (default) |
| `SAGEMAKER_ROLE_ARN` | IAM role for SageMaker jobs |
| `AWS_DEFAULT_REGION` | AWS region (default: `eu-west-1`) |
| `MLFLOW_ENABLED=false` | Skip MLflow (default on SageMaker without remote URI) |
| `MLFLOW_TRACKING_URI` | Override MLflow tracking server |
| `MODEL_CHECKPOINT` | Checkpoint path for inference API |

---

## Quick reference — typical workflows

### Full pipeline (first run, local)

```bash
bash scripts/setup_env.sh && source .venv/bin/activate
python -m data_ingestion.download_neu
python -m data_ingestion.preprocess
DATA_SOURCE=local python -m training.train
DATA_SOURCE=local python -m training.evaluate
python -m training.log_run --update-checkpoint
python -m inference.serve
```

### Full pipeline (cloud training)

```bash
bash scripts/setup_env.sh && source .venv/bin/activate
pip install -r requirements-aws.txt
python -m data_ingestion.preprocess
python -m data_ingestion.upload_to_s3 --version v1
python -m training.launch_aws_training
python -m training.evaluate
python -m training.log_run --update-checkpoint
bash scripts/start_mlflow_ui.sh
python -m inference.serve
```

### Re-run evaluation only

```bash
source .venv/bin/activate
python -m training.evaluate
```

---

## Internal commands (do not run manually)

These are invoked by SageMaker or CI, not by users:

```bash
python -m training.sagemaker_entry --epochs 15   # SageMaker container entry
```

---

See also: [README.md](../README.md)