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
- Direct test loop: `cd pkg && go test ./...`.
- Race check: `cd pkg && go test -race ./...`.
- Baseline benchmarks: `cd pkg && go test -bench=. -benchmem -count=5 ./...`.

## Implementation Notes

- Keep cgroup parsing tolerant: missing or malformed files should degrade to empty/zero usage instead of crashing.
- Treat missing GPU tooling as a normal no-GPU environment. Tests and benchmarks should mock `nvidia-smi`.
- Avoid long work while holding the dynamic state mutex; collect values first, then publish a snapshot.
- Preserve the TUI behavior unless a task explicitly asks for a visual/layout change.
