package main

import (
	"strconv"
	"strings"
)

// getStaticGPUInfo fetches total memory for each GPU.
func getStaticGPUInfo(runner CommandRunner) (int, []float64) {
	outMem, err := runner.Output(nvidiaSMICommand, nvidiaSMIMemoryTotalQuery, nvidiaSMICSVFormat)
	if err != nil {
		return 0, nil
	}
	linesMem := strings.Split(strings.TrimSpace(string(outMem)), "\n")
	if len(linesMem) == 0 || linesMem[0] == "" {
		return 0, nil
	}
	totals := make([]float64, 0, len(linesMem))
	for _, line := range linesMem {
		mb, parseErr := strconv.ParseFloat(strings.TrimSpace(line), 64)
		if parseErr != nil {
			continue
		}
		totals = append(totals, mb/bytesPerKB) // Convert MB to GB
	}
	return len(totals), totals
}

// updateLiveGPUUsage fetches current GPU utilization and memory usage.
func updateLiveGPUUsage(gpuCount int, runner CommandRunner) []GPUUsage {
	if gpuCount == 0 {
		return nil
	}
	out, err := runner.Output(nvidiaSMICommand, nvidiaSMIUsageQuery, nvidiaSMICSVFormat)
	if err != nil {
		return nil
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	usage := make([]GPUUsage, 0, len(lines))
	for i, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) != minGPUUsageCount {
			continue
		}
		util, parseErr := strconv.Atoi(strings.TrimSpace(parts[0]))
		if parseErr != nil {
			continue
		}
		memUsedMB, parseErr := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if parseErr != nil {
			continue
		}
		usage = append(usage, GPUUsage{Index: i, Utilization: util, MemUsedGB: memUsedMB / bytesPerKB})
	}
	return usage
}

// getGPUProcessMap queries nvidia-smi pmon for apps running on the GPU and maps their PID to usage.
func getGPUProcessMap(runner CommandRunner) map[int32]GPUProcessInfo {
	// Use `pmon` with `-c 1` to get a single snapshot, and `-s um` for utilization and memory.
	out, err := runner.Output(
		nvidiaSMICommand,
		nvidiaSMIPMonCommand,
		nvidiaSMIPMonCountFlag,
		nvidiaSMIPMonSampleCount,
		nvidiaSMIPMonSelectFlag,
		nvidiaSMIPMonUsageMemory,
	)
	if err != nil {
		return nil
	}
	lines := strings.Split(string(out), "\n")
	processMap := make(map[int32]GPUProcessInfo, len(lines))
	for _, line := range lines {
		if strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < minGPUInfoCount {
			continue
		}
		var pid int64
		pid, err = strconv.ParseInt(fields[1], 10, 32)
		if err != nil {
			continue
		}
		gpuIndex, parseErr := strconv.Atoi(fields[0])
		if parseErr != nil {
			continue
		}
		gpuUtil, parseErr := parseGPUPercentField(fields[3])
		if parseErr != nil {
			continue
		}
		memUtil, parseErr := parseGPUPercentField(fields[4])
		if parseErr != nil {
			continue
		}
		processMap[int32(pid)] = GPUProcessInfo{GPUIndex: gpuIndex, GPUUtil: gpuUtil, GPUMemUtil: memUtil}
	}
	return processMap
}

func parseGPUPercentField(value string) (uint64, error) {
	if value == "-" {
		return 0, nil
	}
	return strconv.ParseUint(value, 10, 64)
}
