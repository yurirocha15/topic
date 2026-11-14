// Topic is a top-like terminal monitor for containers. It reads the container's cgroup limits
// (v1 or v2) and shows CPU, memory, PIDs, pressure, disk, network, NVIDIA GPU, and per-process
// usage as a percentage of those limits instead of the host's resources. Without limits, or
// outside a container, it shows host usage.
//
// Install it with:
//
//	go install github.com/yurirocha15/topic/cmd/topic@latest
//
// Usage:
//
//	topic [flags]
//
// Common flags:
//
//	--refresh=2s      refresh interval
//	--sort=mem        initial sort: cpu, mem, gpu, gpumem, pid, user, command
//	--once            print one text snapshot and exit
//	--once --json     print one JSON snapshot for scripts
//	--no-gpu          skip GPU collection
//	--no-docker       skip the Docker metadata lookup
//	--no-kubernetes   skip the Kubernetes metadata lookup
//	--version         print the version and exit
//
// Run topic --help for every flag. Inside the dashboard, press ? for the keys.
//
// The default build is fully static. Building with -tags nvml reads GPU usage through the NVIDIA
// driver library instead of nvidia-smi; that build needs glibc. See the README at
// https://github.com/yurirocha15/topic for how each metric is measured.
package main
