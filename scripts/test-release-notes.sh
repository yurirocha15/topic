#!/usr/bin/env bash
# Tests scripts/release-notes.sh against a fixture changelog and the real CHANGELOG.md.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
notes_script="$script_dir/release-notes.sh"
real_changelog="$script_dir/../CHANGELOG.md"
fixture="$(mktemp)"
trap 'rm -f "$fixture"' EXIT

cat >"$fixture" <<'CHANGELOG'
# Changelog

## [Unreleased]

### Fixed

- Unreleased entry.

## [1.10.0] - 2026-02-01

### Added

- Ten entry.

## [1.1.0] - 2026-01-01

### Added

- One entry.

### Fixed

- Another entry.

## [1.0.1] - 2025-12-01

### Fixed

## [1.0.0] - 2025-11-01

- Oldest entry.

[Unreleased]: https://example.com/compare/v1.10.0...HEAD
[1.10.0]: https://example.com/compare/v1.1.0...v1.10.0
CHANGELOG

failures=0

fail() {
    echo "FAIL: $1" >&2
    failures=$((failures + 1))
}

expect_notes() {
    local name="$1" tag="$2" expected="$3" actual
    if ! actual="$("$notes_script" "$tag" "$fixture" 2>&1)"; then
        fail "$name: exited with an error: $actual"
    elif [[ "$actual" != "$expected" ]]; then
        fail "$name: got '$actual'"
    fi
}

expect_error() {
    local name="$1" tag="$2" message="$3" changelog="${4:-$fixture}" actual
    if actual="$("$notes_script" "$tag" "$changelog" 2>&1)"; then
        fail "$name: expected an error, got '$actual'"
    elif [[ "$actual" != *"$message"* ]]; then
        fail "$name: unexpected error '$actual'"
    fi
}

expect_notes "section between versions" v1.1.0 "### Added

- One entry.

### Fixed

- Another entry."
expect_notes "version prefix is not a match" v1.10.0 "### Added

- Ten entry."
expect_notes "last section drops link references" v1.0.0 "- Oldest entry."

expect_error "missing version" v9.9.9 "no entries for 9.9.9"
expect_error "section without entries" v1.0.1 "no entries for 1.0.1"
expect_error "tag without v" 1.1.0 "must look like vX.Y.Z"
expect_error "pre-release tag" v1.1.0-rc1 "must look like vX.Y.Z"
expect_error "newline in tag" $'v1.1.0\nx' "must look like vX.Y.Z"
expect_error "empty tag" "" "must look like vX.Y.Z"
expect_error "missing changelog" v1.1.0 "Changelog not found" /nonexistent/CHANGELOG.md

for tag in v0.0.1 v0.0.2 v0.0.3; do
    "$notes_script" "$tag" "$real_changelog" >/dev/null || fail "CHANGELOG.md has no section for $tag"
done

if ((failures > 0)); then
    echo "$failures release notes test(s) failed." >&2
    exit 1
fi
echo "Release notes tests passed."
