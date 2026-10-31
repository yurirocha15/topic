# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `--version` prints the version of the binary.

### Changed

- Releases are built from a pushed `vX.Y.Z` tag, or from a manual workflow run given a branch and a version tag.
- Release notes come from the matching section of this changelog.
- Releases include a `SHA256SUMS` file.
- Building from source now requires Go 1.26.
- Updated to gopsutil v4, tview v0.42, and tcell v2.13.
- Container memory now reports the working set on cgroup v1 and v2. Cgroup v2 previously counted only anonymous memory, and cgroup v1 included reclaimable cache.

### Security

- Process names, users, mount paths, interface names, filter text, and dialogs are shown literally, so a process can no longer restyle or hide text on screen with display markup.
- Control characters are removed from process commands, so a process can no longer send terminal escape sequences through `--once` output.

### Fixed

- `--once` and `--json` now measure container CPU, network, and disk rates over a 250 ms window instead of reporting 0.
- SIGTERM, SIGHUP, and SIGINT now restore the terminal before `topic` exits.
- `--refresh` values of 0 or less are rejected with an error instead of crashing at startup.
- The first dashboard refresh can no longer overlap with the periodic refresh.
- Release binaries now report their release version.
- OOM and CPU throttling alerts now appear when the counters increase and clear after 30 seconds, instead of staying on for the life of the container.
- Warning alerts use the gold warning color instead of red.
- Metric history no longer crashes on a negative value.
- Process `%CPU` shows current usage instead of a lifetime average, and a reused PID no longer inherits the old process's numbers.
- Process `%MEM` uses host memory when the container has no memory limit, instead of showing 0.
- `--once` text output no longer contains display markup such as `[white]`.
- Kernel threads such as `[kthreadd]` and the `[paused]` table title are no longer hidden.
- Signals are not sent if the selected process exited or its PID was reused while a dialog was open.
- Long mount paths are shortened without splitting non-ASCII characters.
- All single-file bind mounts are hidden from storage, not only `/etc/hosts`, `/etc/hostname`, and `/etc/resolv.conf`.

## [0.0.3] - 2026-10-31

### Added

- Process details view (`Enter`) and process signaling (`k`).
- Sort mode (`s`), reverse sort (`r`), filter mode (`/`), pause (`p`), and tree view (`t`).
- Network and disk throughput, PID usage, cgroup pressure, cgroup events, metric history, and alerts.
- `--once` and `--json` for one-shot text and JSON snapshots.
- Docker, Kubernetes, and NVIDIA/NVML integration status, with `--no-docker`, `--no-kubernetes`, and `--no-nvml` to disable each one.
- `--refresh`, `--sort`, `--no-gpu`, `--ascii`, and `--no-ascii` options.
- Deterministic integration tests and Docker and Kubernetes E2E validation targets.
- MIT license.

### Changed

- The dashboard adapts to the terminal width, with aligned resource columns on wide terminals and a single column on narrow ones.
- Usage values share one color scale: green below 70%, gold from 70% to 89.9%, and red at 90% or above.

### Fixed

- `Enter` and `k` now act on the PID shown in the highlighted row, even if the list refreshed first.
- Tree view keeps tree order with the default sort, and processes with equal sort values keep a stable order.
- Cgroup v1 containers without a memory limit no longer report an 8 EiB limit.
- Container CPU no longer shows a false spike on the first sample.
- Disks mounted at paths such as `/developer` are no longer hidden by the `/dev` rule, and the `/etc/hosts`, `/etc/hostname`, and `/etc/resolv.conf` bind mounts are no longer listed as disks.
- Container IDs are detected in systemd scope cgroup paths for Docker, containerd, CRI-O, and Podman.
- GPU memory stays matched to the right GPU when `nvidia-smi` reports an unparsable total.
- Multi-column resource bars stay aligned.
- Interactive input no longer deadlocks while the screen redraws.

## [0.0.2] - 2025-11-22

### Added

- Disk usage monitoring.
- Multi-column resource layout.

### Fixed

- Containers without resource limits now show usage against the host's resources.

## [0.0.1] - 2025-08-07

### Added

- Terminal dashboard showing CPU and memory usage against cgroup limits.
- GPU usage through `nvidia-smi`.
- Process table.
- Install script for Linux, including Alpine.
- Static Linux binaries for amd64 and arm64.

[Unreleased]: https://github.com/yurirocha15/topic/compare/v0.0.3...HEAD
[0.0.3]: https://github.com/yurirocha15/topic/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/yurirocha15/topic/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/yurirocha15/topic/releases/tag/v0.0.1
