package main

import (
	"strconv"
	"strings"
)

// getStaticGPUInfo fetches total memory for each GPU.
func getStaticGPUInfo(runner CommandRunner) (int, []float64, error) {
	outMem, err := runner.Output(nvidiaSMICommand, nvidiaSMIMemoryTotalQuery, nvidiaSMICSVFormat)
	if err != nil {
		return 0, nil, err
	}
	var totals []float64
	for line := range strings.SplitSeq(string(outMem), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		mb, parseErr := strconv.ParseFloat(line, 64)
		if parseErr != nil {
			mb = 0 // unknown total; keep the slot so later GPUs stay aligned
		}
		totals = append(totals, mb/bytesPerKB) // Convert MB to GB
	}
	if len(totals) == 0 {
		return 0, nil, nil
	}
	return len(totals), totals, nil
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

// pmonColumns holds the data-row positions of the pmon fields topic reads. fb is -1 when it is unknown.
type pmonColumns struct {
	gpu, pid, sm, fb int
}

// pmon column names, as printed in its header.
const (
	pmonColumnGPU = "gpu"
	pmonColumnPID = "pid"
	pmonColumnSM  = "sm"
	pmonColumnFB  = "fb"
)

// Positions of the leading columns every pmon version prints, used when there is no header.
const (
	pmonDefaultGPUColumn = 0
	pmonDefaultPIDColumn = 1
	pmonDefaultSMColumn  = 3
	pmonUnknownColumn    = -1
)

// defaultPMonColumns assumes the leading columns every driver prints and leaves memory unknown.
func defaultPMonColumns() pmonColumns {
	return pmonColumns{
		gpu: pmonDefaultGPUColumn,
		pid: pmonDefaultPIDColumn,
		sm:  pmonDefaultSMColumn,
		fb:  pmonUnknownColumn,
	}
}

// parsePMonHeader reads column positions from a "# gpu pid type ..." header line.
func parsePMonHeader(line string) (pmonColumns, bool) {
	names := strings.Fields(strings.TrimPrefix(line, "#"))
	if len(names) == 0 || names[0] != pmonColumnGPU {
		return pmonColumns{}, false
	}
	columns := pmonColumns{gpu: pmonUnknownColumn, pid: pmonUnknownColumn, sm: pmonUnknownColumn, fb: pmonUnknownColumn}
	for i, name := range names {
		switch name {
		case pmonColumnGPU:
			columns.gpu = i
		case pmonColumnPID:
			columns.pid = i
		case pmonColumnSM:
			columns.sm = i
		case pmonColumnFB:
			columns.fb = i
		}
	}
	if columns.gpu < 0 || columns.pid < 0 || columns.sm < 0 {
		return pmonColumns{}, false
	}
	return columns, true
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
	columns := defaultPMonColumns()
	for _, line := range lines {
		if strings.HasPrefix(line, "#") {
			// Column order differs between driver versions, so positions come from the header.
			if header, ok := parsePMonHeader(line); ok {
				columns = header
			}
			continue
		}
		pid, info, ok := parsePMonRow(strings.Fields(line), columns)
		if !ok {
			continue
		}
		processMap[pid] = info
	}
	return processMap
}

func parsePMonRow(fields []string, columns pmonColumns) (int32, GPUProcessInfo, bool) {
	last := max(columns.gpu, columns.pid, columns.sm, columns.fb)
	if len(fields) <= last {
		return 0, GPUProcessInfo{}, false
	}
	pid, err := strconv.ParseInt(fields[columns.pid], 10, 32)
	if err != nil {
		return 0, GPUProcessInfo{}, false
	}
	gpuIndex, err := strconv.Atoi(fields[columns.gpu])
	if err != nil {
		return 0, GPUProcessInfo{}, false
	}
	gpuUtil, err := parseGPUPercentField(fields[columns.sm])
	if err != nil {
		return 0, GPUProcessInfo{}, false
	}
	var usedMemMB uint64
	if columns.fb >= 0 {
		// "-" means the driver reported no memory for this process.
		usedMemMB, err = parseGPUPercentField(fields[columns.fb])
		if err != nil {
			return 0, GPUProcessInfo{}, false
		}
	}
	return int32(pid), GPUProcessInfo{GPUIndex: gpuIndex, GPUUtil: gpuUtil, UsedMemMB: usedMemMB}, true
}

// parseGPUPercentField parses a pmon number, treating "-" as 0.
func parseGPUPercentField(value string) (uint64, error) {
	if value == "-" {
		return 0, nil
	}
	return strconv.ParseUint(value, 10, 64)
}
