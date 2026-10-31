#!/usr/bin/env bash
# Prints the CHANGELOG.md section for a release tag, and fails if there is none.
# Usage: scripts/release-notes.sh vX.Y.Z [changelog-path]

set -euo pipefail

tag="${1:-}"
changelog="${2:-CHANGELOG.md}"

if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Version tag must look like vX.Y.Z, got: '$tag'" >&2
    exit 1
fi

if [[ ! -f "$changelog" ]]; then
    echo "Changelog not found: $changelog" >&2
    exit 1
fi

# Keep the lines under the matching heading, without link references or outer blank lines.
notes="$(awk -v heading="## [${tag#v}]" '
    /^## / {
        if (found) exit
        found = (index($0, heading) == 1)
        next
    }
    found && !/^\[[^]]+\]: / { print }
' "$changelog" | sed -e '/./,$!d')"

if ! grep -Eq '^[-*] ' <<<"$notes"; then
    echo "$changelog has no entries for ${tag#v}. Rename [Unreleased] to [${tag#v}] before releasing." >&2
    exit 1
fi

printf '%s\n' "$notes"
