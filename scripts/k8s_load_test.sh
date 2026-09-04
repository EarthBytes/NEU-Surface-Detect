#!/usr/bin/env bash
# Generate load against /predict to observe HPA scale-up.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAMESPACE="${NAMESPACE:-neu-surface-detect}"
SERVICE="${SERVICE:-neu-surface-detect-service}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
REQUESTS="${REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-20}"
IMAGE_PATH="${IMAGE_PATH:-}"

log() {
  printf '==> %s\n' "$*"
}

create_test_image() {
  local output="$1"
  python - <<'PY' "$output"
from pathlib import Path
import sys
from io import BytesIO
from PIL import Image

path = Path(sys.argv[1])
image = Image.new("RGB", (224, 224), color=(128, 128, 128))
buffer = BytesIO()
image.save(buffer, format="JPEG")
path.write_bytes(buffer.getvalue())
PY
}

main() {
  if [[ -z "$IMAGE_PATH" ]]; then
    IMAGE_PATH="$(mktemp /tmp/neu-defect-XXXXXX.jpg)"
    create_test_image "$IMAGE_PATH"
    trap 'rm -f "$IMAGE_PATH"' EXIT
  fi

  log "Starting port-forward on localhost:${LOCAL_PORT}"
  kubectl port-forward "service/${SERVICE}" "${LOCAL_PORT}:80" -n "$NAMESPACE" >/tmp/neu-k8s-load-test-pf.log 2>&1 &
  PF_PID=$!
  trap 'kill "$PF_PID" 2>/dev/null || true; rm -f "$IMAGE_PATH"' EXIT

  for _ in $(seq 1 30); do
    if curl -fs "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null; then
      break
    fi
    sleep 1
  done

  log "Sending ${REQUESTS} /predict requests (${CONCURRENCY} concurrent)"
  seq 1 "$REQUESTS" | xargs -P "$CONCURRENCY" -I{} \
    curl -fs -o /dev/null \
      -F "file=@${IMAGE_PATH}" \
      "http://127.0.0.1:${LOCAL_PORT}/predict" || true

  log "Current HPA status"
  kubectl get hpa -n "$NAMESPACE"
  kubectl get pods -n "$NAMESPACE" -l app=neu-surface-detect-api
}

main "$@"
