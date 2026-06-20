#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${TOPIC_KUBE_E2E_IMAGE:-topic:e2e-kubernetes}"
HOSTPATH_IMAGE="${TOPIC_KUBE_E2E_HOSTPATH_IMAGE:-alpine:3.20}"
USE_HOSTPATH="${TOPIC_KUBE_E2E_USE_HOSTPATH:-0}"
NAMESPACE="${TOPIC_KUBE_E2E_NAMESPACE:-topic-e2e-$(date +%s)}"
POD_NAME="topic-e2e"
POD_IMAGE="$IMAGE"
VOLUME_MOUNTS=""
VOLUMES=""

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required for Kubernetes E2E integration validation" >&2
  exit 1
fi

if ! kubectl cluster-info >/dev/null 2>&1; then
  echo "kubectl is not connected to a cluster" >&2
  exit 1
fi

cleanup() {
  kubectl delete namespace "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

load_local_image() {
  local context
  context="$(kubectl config current-context 2>/dev/null || true)"

  if command -v kind >/dev/null 2>&1 && [[ "$context" == kind-* ]]; then
    kind load docker-image "$IMAGE" >/dev/null
    return 0
  fi

  if command -v k3d >/dev/null 2>&1 && [[ "$context" == k3d-* ]]; then
    k3d image import "$IMAGE" >/dev/null
    return 0
  fi

  if command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
    minikube image load "$IMAGE" >/dev/null
    return 0
  fi

  if command -v k3s >/dev/null 2>&1; then
    if [[ "$(id -u)" == "0" ]]; then
      docker save "$IMAGE" | k3s ctr images import - >/dev/null
      return 0
    fi
    if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      docker save "$IMAGE" | sudo -n k3s ctr images import - >/dev/null
      return 0
    fi
  fi

  return 1
}

if [[ "$USE_HOSTPATH" == "1" ]]; then
  make -C "$ROOT_DIR" build >/dev/null
  POD_IMAGE="$HOSTPATH_IMAGE"
  VOLUME_MOUNTS="
      volumeMounts:
        - name: topic-binary
          mountPath: /usr/local/bin/topic
          readOnly: true"
  VOLUMES="
  volumes:
    - name: topic-binary
      hostPath:
        path: ${ROOT_DIR}/dist/topic
        type: File"
elif [[ "${TOPIC_KUBE_E2E_BUILD_IMAGE:-1}" == "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to build the default Kubernetes E2E image; set TOPIC_KUBE_E2E_USE_HOSTPATH=1, or set TOPIC_KUBE_E2E_BUILD_IMAGE=0 and TOPIC_KUBE_E2E_IMAGE to use an existing image" >&2
    exit 1
  fi
  make -C "$ROOT_DIR" build >/dev/null
  docker build -t "$IMAGE" -f - "$ROOT_DIR" >/dev/null <<'DOCKERFILE'
FROM alpine:3.20
COPY dist/topic /usr/local/bin/topic
ENTRYPOINT ["/usr/local/bin/topic"]
DOCKERFILE
  if ! load_local_image; then
    cat >&2 <<EOF
No automatic local image loader was available for the current Kubernetes context.
If the pod cannot pull "$IMAGE", either:
  - load the image into your cluster manually, or
  - publish an image and run:
    TOPIC_KUBE_E2E_BUILD_IMAGE=0 TOPIC_KUBE_E2E_IMAGE=<registry>/<image>:<tag> make e2e-kubernetes
EOF
  fi
fi

kubectl create namespace "$NAMESPACE" >/dev/null

cat <<YAML | kubectl apply -n "$NAMESPACE" -f - >/dev/null
apiVersion: v1
kind: ServiceAccount
metadata:
  name: topic-e2e
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: topic-e2e
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: topic-e2e
subjects:
  - kind: ServiceAccount
    name: topic-e2e
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: topic-e2e
---
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  labels:
    app: topic-e2e
spec:
  restartPolicy: Never
  serviceAccountName: topic-e2e
  containers:
    - name: topic
      image: ${POD_IMAGE}
      imagePullPolicy: IfNotPresent
      command: ["/usr/local/bin/topic"]
      env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
      args: ["--once", "--json"]
${VOLUME_MOUNTS}
${VOLUMES}
YAML

deadline=$((SECONDS + 120))
phase=""
while (( SECONDS < deadline )); do
  phase="$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  waiting_reason="$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)"
  if [[ "$waiting_reason" == "ErrImagePull" || "$waiting_reason" == "ImagePullBackOff" ]]; then
    kubectl describe pod "$POD_NAME" -n "$NAMESPACE" >&2 || true
    cat >&2 <<EOF
Kubernetes E2E pod could not pull "$POD_IMAGE".
Use a cluster-visible image:
  TOPIC_KUBE_E2E_BUILD_IMAGE=0 TOPIC_KUBE_E2E_IMAGE=<registry>/<image>:<tag> make e2e-kubernetes
or load the local image into the cluster runtime and rerun this target.
For single-node local clusters where the node can see this checkout, try:
  TOPIC_KUBE_E2E_USE_HOSTPATH=1 make e2e-kubernetes
EOF
    exit 1
  fi
  case "$phase" in
    Succeeded) break ;;
    Failed)
      kubectl describe pod "$POD_NAME" -n "$NAMESPACE" >&2 || true
      kubectl logs "$POD_NAME" -n "$NAMESPACE" >&2 || true
      exit 1
      ;;
  esac
  sleep 2
done

if [[ "$phase" != "Succeeded" ]]; then
  kubectl describe pod "$POD_NAME" -n "$NAMESPACE" >&2 || true
  kubectl logs "$POD_NAME" -n "$NAMESPACE" >&2 || true
  echo "timed out waiting for Kubernetes E2E pod to succeed; last phase=$phase" >&2
  exit 1
fi

snapshot="$(kubectl logs "$POD_NAME" -n "$NAMESPACE")"

SNAPSHOT="$snapshot" NAMESPACE="$NAMESPACE" POD_NAME="$POD_NAME" python3 - <<'PY'
import json
import os
import sys

snapshot = json.loads(os.environ["SNAPSHOT"])
integrations = {
    item.get("name"): item
    for item in snapshot.get("static", {}).get("integrations", [])
}
kubernetes = integrations.get("kubernetes")
if not kubernetes:
    sys.exit("missing kubernetes integration status")
if kubernetes.get("available") is not True:
    sys.exit(f"kubernetes integration was not available: {kubernetes}")

metadata = snapshot.get("metadata", {})
expected_namespace = os.environ["NAMESPACE"]
expected_pod = os.environ["POD_NAME"]
if metadata.get("runtime") != "kubernetes":
    sys.exit(f"expected kubernetes runtime metadata, got: {metadata}")
if metadata.get("namespace") != expected_namespace:
    sys.exit(f"expected namespace {expected_namespace}, got: {metadata}")
if metadata.get("pod") != expected_pod:
    sys.exit(f"expected pod {expected_pod}, got: {metadata}")
labels = metadata.get("labels") or {}
if labels.get("app") != "topic-e2e":
    sys.exit(f"expected pod label metadata, got: {metadata}")

print(f"kubernetes=true namespace={metadata.get('namespace')} pod={metadata.get('pod')} node={metadata.get('node')}")
PY
