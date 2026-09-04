#!/usr/bin/env bash
# Build the inference image, load it into kind, and apply Kubernetes manifests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLUSTER_NAME="${CLUSTER_NAME:-neu-surface-detect}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
IMAGE_NAME="${IMAGE_NAME:-neu-surface-detect-api}"
MODEL_SOURCE="${MODEL_SOURCE:-image}"  # image | s3

log() {
  printf '==> %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

ensure_checkpoint() {
  if [[ ! -f models/checkpoints/best_model.pt ]]; then
    log "Checkpoint missing; creating CI fixture checkpoint"
    python tests/create_fixture_checkpoint.py
  fi
}

build_and_load_image() {
  local image_ref="${IMAGE_NAME}:${IMAGE_TAG}"

  log "Building Docker image ${image_ref}"
  docker build -f inference/Dockerfile -t "$image_ref" .

  log "Loading image into kind cluster '${CLUSTER_NAME}'"
  kind load docker-image "$image_ref" --name "$CLUSTER_NAME"
}

apply_manifests() {
  local image_ref="${IMAGE_NAME}:${IMAGE_TAG}"

  if [[ "$MODEL_SOURCE" == "s3" ]]; then
    if ! kubectl get secret neu-aws-credentials -n neu-surface-detect >/dev/null 2>&1; then
      echo "S3 mode requires neu-aws-credentials secret. Run:" >&2
      echo "  CREATE_AWS_SECRET=true bash scripts/k8s_setup.sh" >&2
      exit 1
    fi

    log "Applying S3 manifests (image: ${image_ref})"
    sed "s|neu-surface-detect-api:v1|${image_ref}|g" k8s/deployment-s3.yaml \
      | kubectl apply -f k8s/namespace.yaml \
                      -f k8s/configmap-s3.yaml \
                      -f - \
                      -f k8s/service.yaml \
                      -f k8s/hpa.yaml \
                      -f k8s/ingress.yaml
  else
    log "Applying default manifests (image: ${image_ref})"
    kubectl kustomize k8s \
      | sed "s|neu-surface-detect-api:v1|${image_ref}|g" \
      | kubectl apply -f -
  fi

  kubectl rollout status deployment/neu-surface-detect-api -n neu-surface-detect --timeout=300s
}

print_access_help() {
  cat <<EOF

Deployment complete.

Service (port-forward):
  kubectl port-forward service/neu-surface-detect-service 8000:80 -n neu-surface-detect

Ingress (after adding '127.0.0.1 neu-surface-detect.local' to /etc/hosts):
  kubectl port-forward -n ingress-nginx service/ingress-nginx-controller 8080:80
  open http://neu-surface-detect.local:8080/health

Status:
  kubectl get pods,svc,hpa,ingress -n neu-surface-detect

EOF
}

main() {
  require_cmd docker
  require_cmd kind
  require_cmd kubectl

  if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    echo "kind cluster '${CLUSTER_NAME}' not found. Run: bash scripts/k8s_setup.sh" >&2
    exit 1
  fi

  ensure_checkpoint
  build_and_load_image
  apply_manifests
  print_access_help
}

main "$@"
