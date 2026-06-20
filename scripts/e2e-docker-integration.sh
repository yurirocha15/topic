#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${TOPIC_DOCKER_E2E_IMAGE:-topic:e2e-docker}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for Docker E2E integration validation" >&2
  exit 1
fi

if [[ ! -S /var/run/docker.sock ]]; then
  echo "/var/run/docker.sock is required for Docker E2E integration validation" >&2
  exit 1
fi

make -C "$ROOT_DIR" build >/dev/null

docker build -t "$IMAGE" -f - "$ROOT_DIR" >/dev/null <<'DOCKERFILE'
FROM alpine:3.20
COPY dist/topic /usr/local/bin/topic
ENTRYPOINT ["/usr/local/bin/topic"]
DOCKERFILE

snapshot="$(
  docker run --rm \
    --label topic.e2e=docker \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    "$IMAGE" --once --json
)"

SNAPSHOT="$snapshot" python3 - <<'PY'
import json
import os
import sys

snapshot = json.loads(os.environ["SNAPSHOT"])
integrations = {
    item.get("name"): item
    for item in snapshot.get("static", {}).get("integrations", [])
}
docker = integrations.get("docker")
if not docker:
    sys.exit("missing docker integration status")
if docker.get("available") is not True:
    sys.exit(f"docker integration was not available: {docker}")

metadata = snapshot.get("metadata", {})
if metadata.get("runtime") != "docker":
    sys.exit(f"expected docker runtime metadata, got: {metadata}")
if not metadata.get("id"):
    sys.exit(f"expected docker container id metadata, got: {metadata}")
labels = metadata.get("labels") or {}
if labels.get("topic.e2e") != "docker":
    sys.exit(f"expected docker label metadata, got: {metadata}")

print(f"docker=true id={metadata.get('id')[:12]} image={metadata.get('image')} name={metadata.get('name')}")
PY
