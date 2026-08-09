#!/usr/bin/env bash
# Create a virtual environment and install project dependencies
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python3
fi

echo "Using: $($PYTHON --version)"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo ""
echo "Environment ready. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Verify with:"
echo "  python scripts/verify_env.py"
