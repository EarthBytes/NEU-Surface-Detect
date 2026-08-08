#!/usr/bin/env bash
# Start the local MLflow tracking UI for this project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

STORE_URI="${MLFLOW_TRACKING_URI:-sqlite:///${ROOT}/models/mlflow.db}"
echo "Starting MLflow UI with backend store: ${STORE_URI}"
exec mlflow ui --backend-store-uri "${STORE_URI}" --host 127.0.0.1 --port 5000
