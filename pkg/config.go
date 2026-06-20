package main

import (
	"flag"
	"strings"
	"time"
)

func parseConfig() AppConfig {
	refresh := flag.Duration("refresh", time.Second, "refresh interval")
	noGPU := flag.Bool("no-gpu", false, "disable GPU collection")
	noDocker := flag.Bool("no-docker", false, "disable Docker metadata integration")
	noKubernetes := flag.Bool("no-kubernetes", false, "disable Kubernetes metadata integration")
	noNVML := flag.Bool("no-nvml", false, "disable NVML GPU integration")
	once := flag.Bool("once", false, "collect one snapshot and exit")
	jsonOutput := flag.Bool("json", false, "print a JSON snapshot; implies --once")
	hideASCII := flag.Bool("no-ascii", false, "hide the ASCII art panel")
	sortColumn := flag.String("sort", "cpu", "initial process sort: cpu, mem, gpu, gpumem, pid, user, command")
	flag.Parse()

	return AppConfig{
		RefreshInterval:   *refresh,
		DisableGPU:        *noGPU,
		DisableDocker:     *noDocker,
		DisableKubernetes: *noKubernetes,
		DisableNVML:       *noNVML,
		Once:              *once || *jsonOutput,
		JSONOutput:        *jsonOutput,
		HideASCIIArt:      *hideASCII,
		InitialSort:       parseProcessSortColumn(*sortColumn),
	}
}

func parseProcessSortColumn(value string) ProcessSortColumn {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "mem", "memory":
		return SortByMemory
	case "gpu":
		return SortByGPU
	case "gpumem", "gpu_mem", "gpu-memory":
		return SortByGPUMemory
	case "pid":
		return SortByPID
	case "user":
		return SortByUser
	case "command", "cmd":
		return SortByCommand
	case resourceCPU:
		fallthrough
	default:
		return SortByCPU
	}
}
