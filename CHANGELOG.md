# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `%GPUMEM` now shows how much of its GPU's memory each process uses. It previously showed how busy the GPU memory bus was, so a process holding gigabytes of GPU memory could show 0%. The process details also show the amount in MiB.
- Disks mounted at paths that contain "loop", such as `/mnt/loopback`, are no longer hidden from the storage list. Only loop devices are hidden.
- The install script no longer repeats the program name when it reports the installed version.
- The highlight in the process table now stays on the process you select instead of moving to whatever process lands on that row after a refresh.
- Signal dialogs open with `Cancel` selected, so pressing `Enter` no longer sends `SIGTERM` or `SIGKILL` by accident. The `SIGKILL` confirmation also shows the command.
- When filtering shortens the process table, a visible process stays selected instead of `Enter` and `k` reporting "No process selected".
- The mouse now works in the process table and dialogs, as the help already described.
- The sort arrow now shows the real direction: `↓` is high to low and `↑` is low to high for every column. In sort mode, `↓` and `↑` choose that direction.
- The dashboard is laid out for the terminal's real size as soon as it starts and right after a resize. The first screen previously had no labels, and a resized screen kept the old layout until the next refresh.
- Wide characters, such as Japanese or Chinese text in mount paths and interface names, no longer push rows out of alignment or past the edge of the panel.
- Long mount paths are shortened once, from the left, so the end of the path stays visible, as in `DISK …models/llama`. Two mounts with a long shared beginning were previously cut down to the same text.
- Without a GPU, the process table no longer shows empty `%GPU` and `%GPUMEM` columns, which leaves more room for commands, and sort mode skips them.
- On terminals narrower than 100 columns, the side panel is hidden so the CPU, memory, and disk rows keep their labels and details; the process table title shows `? help` instead. The top panel now takes at most half the screen, dropping history, then pressure and disk I/O, then network rows when space runs out, so the process table always keeps the other half.
- Help opens in a left-aligned box with one key per line that fits the terminal and scrolls with the arrow keys and Page Up/Down on short screens. `Esc`, `q`, or `?` closes it.
- CPU, memory, and GPU history now use a fixed 0–100% scale, so a steady 20% draws as low blocks instead of full ones. History starts empty and fills as samples arrive, instead of repeating the first reading 30 times. Network and disk I/O history still scale to their own peak.
- Memory and PID alerts now use the same levels as the colors: a warning from 70% (gold) and a critical alert from 90% (red). They were 85% and 95%, so a red bar could come with only a warning. The pressure row also turns gold when it raises an alert.
- Memory, disk, and GPU sizes are labelled GiB, because they are measured in 1024-based units. They were labelled GB, which made a 2 GiB container limit look different from what Docker and Kubernetes report.
- The network row stays on one interface instead of jumping to whichever was busiest that second. It switches only when another interface carries more than twice the traffic, shows the default-route interface when the network is idle, and its history now follows the interface it names.
- Resource details, such as `2.00 CPU limit` and `1.77 / 2.00 GiB`, are drawn in the normal text color instead of dim gray, which was hard to read on many terminal themes.

### Changed

- `topic` uses much less CPU and memory. It reads the process list straight from `/proc`, which takes about a tenth of the time and allocates far less memory. With about 730 processes, its own CPU use dropped from about 9% to about 2.5% of one core.
- Processes whose user has no account name show the numeric user ID, as `ps` does, instead of `n/a`.
- While paused, `topic` no longer redraws the screen on every refresh, so it uses almost no CPU.
- On machines with an NVIDIA GPU, the dashboard keeps one `nvidia-smi` running for GPU usage instead of starting a new one every refresh, which saves about 2% of a core.

## [0.0.4] - 2026-10-31

### Added

- `--version` prints the version of the binary.
- Releases include a `SHA256SUMS` file and signed build provenance, so downloads can be verified.

### Changed

- Building from source now requires Go 1.26.
- Container memory now reports the working set on cgroup v1 and v2. Cgroup v2 previously counted only anonymous memory, and cgroup v1 included reclaimable cache.

### Security

- Process names, users, mount paths, interface names, filter text, and dialogs are shown literally, so a process can no longer restyle or hide text on screen, or disguise text as a hyperlink, with display markup.
- Control, bidirectional, and other format characters are removed from displayed process and system names, so a process can no longer send terminal escape sequences through `--once` output or reorder how its command is displayed.
- Signals are bound to the process through a pidfd and checked against its start time, so a reused PID cannot receive a signal meant for an exited process.
- `install.sh` verifies the download against the release's `SHA256SUMS` file when one is available.

### Fixed

- `--once` and `--json` now measure container CPU, network, and disk rates over a 250 ms window instead of reporting 0.
- SIGTERM, SIGHUP, and SIGINT now restore the terminal before `topic` exits.
- `--refresh` values of 0 or less are rejected with an error instead of crashing at startup.
- OOM and CPU throttling alerts now appear when the counters increase and clear after 30 seconds, instead of staying on for the life of the container.
- Warning alerts use the gold warning color instead of red.
- Process `%CPU` shows current usage instead of a lifetime average, and a reused PID no longer inherits the old process's numbers.
- Process `%MEM` uses host memory when the container has no memory limit, instead of showing 0.
- `--once` text output no longer contains display markup such as `[white]`.
- Kernel threads such as `[kthreadd]` and the `[paused]` table title are no longer hidden.
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

[Unreleased]: https://github.com/yurirocha15/topic/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/yurirocha15/topic/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/yurirocha15/topic/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/yurirocha15/topic/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/yurirocha15/topic/releases/tag/v0.0.1
