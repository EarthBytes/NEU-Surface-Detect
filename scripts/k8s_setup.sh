#!/usr/bin/env bash
# One-time local cluster bootstrap for kind: metrics-server, ingress, optional AWS secret.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-neu-surface-detect}"
INSTALL_INGRESS="${INSTALL_INGRESS:-true}"
CREATE_AWS_SECRET="${CREATE_AWS_SECRET:-false}"

log() {
  printf '==> %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

install_metrics_server() {
  if ! kubectl get deployment metrics-server -n kube-system >/dev/null 2>&1; then
    log "Installing metrics-server"
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  else
    log "metrics-server already installed"
  fi

  if ! kubectl get deployment metrics-server -n kube-system \
    -o jsonpath='{.spec.template.spec.containers[0].args}' \
    | grep -q 'kubelet-insecure-tls'; then
    log "Patching metrics-server for kind (insecure kubelet TLS)"
    kubectl patch deployment metrics-server -n kube-system --type=json \
      -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
  fi

  kubectl rollout status deployment/metrics-server -n kube-system --timeout=120s
}

install_ingress_controller() {
  if [[ "$INSTALL_INGRESS" != "true" ]]; then
    log "Skipping ingress controller install (INSTALL_INGRESS=$INSTALL_INGRESS)"
    return
  fi

  if kubectl get deployment ingress-nginx-controller -n ingress-nginx >/dev/null 2>&1; then
    log "ingress-nginx already installed"
    return
  fi

  log "Installing ingress-nginx controller for kind"
  kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
  kubectl rollout status deployment/ingress-nginx-controller -n ingress-nginx --timeout=180s
}

create_kind_cluster() {
  if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    log "kind cluster '$CLUSTER_NAME' already exists"
    return
  fi

  log "Creating kind cluster '$CLUSTER_NAME'"
  kind create cluster --name "$CLUSTER_NAME"
}

maybe_create_aws_secret() {
  if [[ "$CREATE_AWS_SECRET" != "true" ]]; then
    return
  fi

  require_cmd aws

  log "Creating neu-aws-credentials secret from local AWS CLI config"
  kubectl create namespace neu-surface-detect --dry-run=client -o yaml | kubectl apply -f -
  kubectl create secret generic neu-aws-credentials \
    --namespace neu-surface-detect \
    --from-literal=AWS_ACCESS_KEY_ID="$(aws configure get aws_access_key_id)" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$(aws configure get aws_secret_access_key)" \
    --from-literal=AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-eu-west-1}" \
    --dry-run=client -o yaml | kubectl apply -f -
}

main() {
  require_cmd kind
  require_cmd kubectl
  require_cmd docker

  create_kind_cluster
  install_metrics_server
  install_ingress_controller
  maybe_create_aws_secret

  log "Cluster bootstrap complete"
  kubectl cluster-info
  kubectl get nodes
}

main "$@"
