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
- Direct test loop: `cd pkg && go test ./...`.
- Race check: `cd pkg && go test -race ./...`.
- Baseline benchmarks: `cd pkg && go test -bench=. -benchmem -count=5 ./...`.
- Live integration discovery is opt-in: `TOPIC_LIVE_INTEGRATION_TESTS=1 make test-integrations`.

## Implementation Notes

- Keep cgroup parsing tolerant: missing or malformed files should degrade to empty/zero usage instead of crashing.
- Treat missing GPU tooling as a normal no-GPU environment. Tests and benchmarks should mock `nvidia-smi`.
- Docker, Kubernetes, and NVML integrations must remain optional. Test unavailable, disabled, partial, and success states with fake hooks; keep live integration tests skipped unless explicitly enabled.
- Avoid long work while holding the dynamic state mutex; collect values first, then publish a snapshot.
- Preserve the TUI behavior unless a task explicitly asks for a visual/layout change.
