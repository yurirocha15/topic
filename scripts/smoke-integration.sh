#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${TOPIC_BINARY:-$ROOT_DIR/dist/topic}"

if [[ ! -x "$BIN" ]]; then
  make -C "$ROOT_DIR" build >/dev/null
fi

snapshot="$("$BIN" --once --json)"

if command -v jq >/dev/null 2>&1; then
  printf '%s\n' "$snapshot" | jq -e '
    (.static.integrations | type == "array") and
    ([.static.integrations[].name] | index("docker") and index("kubernetes") and index("nvml"))
  ' >/dev/null
  printf '%s\n' "$snapshot" | jq -r '.static.integrations[] | "\(.name)=\(.available) \(.detail // "")"'
else
  SNAPSHOT="$snapshot" python3 - <<'PY'
import json
import os
import sys

snapshot = json.loads(os.environ["SNAPSHOT"])
integrations = snapshot.get("static", {}).get("integrations", [])
if not isinstance(integrations, list):
    sys.exit("static.integrations is not a list")
names = {item.get("name") for item in integrations}
missing = {"docker", "kubernetes", "nvml"} - names
if missing:
    sys.exit(f"missing integration statuses: {', '.join(sorted(missing))}")
for item in integrations:
    print(f"{item.get('name')}={item.get('available', False)} {item.get('detail', '')}")
PY
fi
