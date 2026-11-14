package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"
)

// parseConfigArgs parses command-line arguments and reports any problem to output.
func parseConfigArgs(args []string, output io.Writer) (AppConfig, error) {
	flags := flag.NewFlagSet("topic", flag.ContinueOnError)
	flags.SetOutput(output)
	refresh := flags.Duration("refresh", time.Second, "refresh interval")
	noGPU := flags.Bool("no-gpu", false, "disable GPU collection")
	noDocker := flags.Bool("no-docker", false, "disable Docker metadata integration")
	noKubernetes := flags.Bool("no-kubernetes", false, "disable Kubernetes metadata integration")
	noNVML := flags.Bool("no-nvml", false, "disable NVML GPU integration")
	once := flags.Bool("once", false, "collect one snapshot and exit")
	jsonOutput := flags.Bool("json", false, "print a JSON snapshot; implies --once")
	showASCII := flags.Bool("ascii", false, "show the ASCII art panel at startup")
	hideASCII := flags.Bool("no-ascii", false, "hide the ASCII art panel")
	sortColumn := flags.String("sort", "cpu", "initial process sort: cpu, mem, gpu, gpumem, pid, user, command")
	showVersion := flags.Bool("version", false, "print the version and exit")
	if err := flags.Parse(args); err != nil {
		return AppConfig{}, err
	}

	// time.NewTicker panics on a non-positive interval
	if *refresh <= 0 {
		err := errors.New("--refresh must be greater than 0")
		fmt.Fprintln(output, err)
		flags.Usage()
		return AppConfig{}, err
	}

	return AppConfig{
		RefreshInterval:   *refresh,
		DisableGPU:        *noGPU,
		DisableDocker:     *noDocker,
		DisableKubernetes: *noKubernetes,
		DisableNVML:       *noNVML,
		Once:              *once || *jsonOutput,
		JSONOutput:        *jsonOutput,
		HideASCIIArt:      !*showASCII || *hideASCII,
		InitialSort:       parseProcessSortColumn(*sortColumn),
		ShowVersion:       *showVersion,
	}, nil
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
