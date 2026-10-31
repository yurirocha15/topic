# Repository Guide

## Project Shape

- `topic` is a Go terminal UI that reports container-aware CPU, memory, GPU, storage, and process usage.
- The Go module lives in `pkg/`; run direct Go commands from that directory.
- Static assets for README/release material live under `static/`.
- Build output belongs in `dist/`. Do not commit generated binaries such as `pkg/topic`.

## Common Commands

- `make test` runs `go test -coverprofile=coverage.out ./...` from `pkg/`.
- `make lint` runs `golangci-lint run` from `pkg/`.
- `make build` builds `dist/topic`.
- `make test-integrations` runs deterministic fake integration tests plus a `--once --json` integration-status smoke check.
- `make test-release-notes` tests the changelog extraction used by the release workflow.
- `make e2e-docker` validates Docker metadata from inside a real Docker container.
- `make e2e-kubernetes` validates Kubernetes metadata from inside a real cluster pod.
- Direct test loop: `cd pkg && go test ./...`.
- Race check: `make test-race` or `cd pkg && go test -race ./...`.
- `make format-check` fails on unformatted code without rewriting files. `make vuln` runs `govulncheck`.
- TUI regression check: `cd pkg && go test -run 'Test(ResourceGridResponsiveAlignment|DashboardLayoutAtCommonTerminalSizes|CompactInfoPanelLeavesProcessTableSpace|TUIInputCaptureDoesNotDeadlock)' -count=1 ./...`.
- Baseline benchmarks: `cd pkg && go test -bench=. -benchmem -count=5 ./...`.
- Live integration discovery is opt-in: `TOPIC_LIVE_INTEGRATION_TESTS=1 make test-integrations`.

## Implementation Notes

- Keep cgroup parsing tolerant: missing or malformed files should degrade to empty/zero usage instead of crashing.
- Treat cgroup v1 memory limits at or above `1<<62` as unlimited.
- Container CPU usage needs a baseline: the first sample reports 0, a failed counter read keeps the existing baseline, and a counter that goes backwards rebaselines.
- Compare storage skip paths on path-segment boundaries, not raw prefixes.
- Accept container IDs from cgroup paths only when they are exactly 64 lowercase hex characters.
- Treat missing GPU tooling as a normal no-GPU environment. Tests and benchmarks should mock `nvidia-smi`.
- Keep one `GPUTotalGB` slot per reported GPU, using 0 for unparsable totals, and look totals up by GPU index, not list position.
- Docker, Kubernetes, and NVML integrations must remain optional. Test unavailable, disabled, partial, and success states with fake hooks; keep live integration tests skipped unless explicitly enabled.
- Use the E2E targets for native-environment validation; `make test-integrations` only proves deterministic behavior and JSON status shape.
- Avoid long work while holding the dynamic state mutex; collect values first, then publish a snapshot.
- Process table rows carry their PID as a cell reference. Resolve selection from that PID, not the row index, because the snapshot can change between drawing and a keypress.
- Keep resource rows on the shared label/percentage/bar/detail grid. At wide widths the two columns must retain a visible divider; narrow layouts must not overflow.
- Use the shared semantic palette for usage values: green below 70%, gold from 70% to 89.9%, and red at 90% or above.
- When changing the composed TUI, validate both tcell simulation sizes and a real pseudo-terminal interaction flow. Update `static/img/topic.png` when the README screenshot becomes materially stale.
- Text from processes, mounts, interfaces, or user input is untrusted. Pass it through `tview.Escape` after truncating and measuring it, at the point it reaches a tagged tview sink.
- Preserve the TUI behavior unless a task explicitly asks for a visual/layout change.
- Add user-facing changes to the `[Unreleased]` section of `CHANGELOG.md`. A release fails unless its version has a section with entries.
