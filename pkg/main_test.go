// ./pkg/main_test.go
package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/rivo/tview"
	"github.com/shirou/gopsutil/v3/process"
)

// --- Mock FileReader for Testing ---

// MockFileReader simulates reading files by returning predefined content from a map.
type MockFileReader struct {
	files map[string]string
}

// ReadFile implements the FileReader interface for MockFileReader.
func (mfs MockFileReader) ReadFile(path string) ([]byte, error) {
	if content, ok := mfs.files[path]; ok {
		return []byte(content), nil
	}
	return nil, fmt.Errorf("file not found in mock: %s", path)
}

// MockCommandRunner simulates running external commands.
type MockCommandRunner struct {
	outputs map[string]string // Maps a command string to its output
	err     error
}

// NewMockCommandRunner creates a new MockCommandRunner with predefined outputs.
func (mcr MockCommandRunner) Output(name string, arg ...string) ([]byte, error) {
	// Create a unique key for the command and its arguments.
	cmdKey := name + " " + strings.Join(arg, " ")
	if output, ok := mcr.outputs[cmdKey]; ok {
		return []byte(output), mcr.err
	}
	return nil, fmt.Errorf("mock command not found: %s", cmdKey)
}

type CountingCommandRunner struct {
	calls int
}

func (ccr *CountingCommandRunner) Output(_ string, _ ...string) ([]byte, error) {
	ccr.calls++
	return nil, errors.New("unexpected command call")
}

type MockProcessProvider struct {
	processes []ProcessHandle
	err       error
}

func (mpp MockProcessProvider) Processes() ([]ProcessHandle, error) {
	return mpp.processes, mpp.err
}

type MockProcessHandle struct {
	pid        int32
	cpuPercent float64
	rss        uint64
	user       string
	cmdline    string
	name       string
	err        error
}

func (mph MockProcessHandle) PID() int32 {
	return mph.pid
}

func (mph MockProcessHandle) CPUPercent() (float64, error) {
	return mph.cpuPercent, mph.err
}

func (mph MockProcessHandle) MemoryInfo() (*process.MemoryInfoStat, error) {
	if mph.err != nil {
		return nil, mph.err
	}
	return &process.MemoryInfoStat{RSS: mph.rss}, nil
}

func (mph MockProcessHandle) Username() (string, error) {
	if mph.user == "" {
		return "", errors.New("missing user")
	}
	return mph.user, nil
}

func (mph MockProcessHandle) Cmdline() (string, error) {
	return mph.cmdline, nil
}

func (mph MockProcessHandle) Name() (string, error) {
	if mph.name == "" {
		return "", errors.New("missing name")
	}
	return mph.name, nil
}

// MockStater simulates os.Stat for testing cgroup version detection.
type MockStater struct {
	// If true, Stat will return a "not exist" error.
	FileDoesNotExist bool
}

func (ms MockStater) Stat(_ string) (os.FileInfo, error) {
	if ms.FileDoesNotExist {
		// Return an error that satisfies os.IsNotExist.
		return nil, os.ErrNotExist
	}
	return os.FileInfo(nil), nil // Return nil, nil for a successful stat.
}

// --- Unit Tests ---

// TestMakeBar tests the makeBar function's ability to correctly render a percentage-based bar.
func TestMakeBar(t *testing.T) {
	testCases := []struct {
		name     string
		percent  float64
		width    int
		expected string
	}{
		{name: "Zero Percent (width 20)", percent: 0, width: 20, expected: "[green]░░░░░░░░░░░░░░░░░░░░[white]"},
		{name: "50 Percent (width 20)", percent: 50, width: 20, expected: "[green]▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░[white]"},
		{name: "100 Percent (width 20)", percent: 100, width: 20, expected: "[green]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[white]"},
		{name: "50 Percent (width 10)", percent: 50, width: 10, expected: "[green]▓▓▓▓▓░░░░░[white]"},
		{name: "75 Percent (width 10)", percent: 75, width: 10, expected: "[green]▓▓▓▓▓▓▓░░░[white]"},
		{name: "Zero width", percent: 100, width: 0, expected: "[green][white]"},
		{name: "Negative width", percent: 100, width: -5, expected: "[green][white]"},
		{name: "Over 100 percent is capped", percent: 125, width: 10, expected: "[green]▓▓▓▓▓▓▓▓▓▓[white]"},
		{name: "Negative percent is empty", percent: -20, width: 10, expected: "[green]░░░░░░░░░░[white]"},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := makeBar(tc.percent, tc.width)
			if actual != tc.expected {
				t.Errorf("makeBar(%.2f, %d) = \n%q\nwant:\n%q", tc.percent, tc.width, actual, tc.expected)
			}
		})
	}
}

// TestMakeBarWidth verifies that the bar's visible character width is always consistent.
func TestMakeBarDynamicWidth(t *testing.T) {
	testWidths := []int{0, 1, 10, 20, 33}

	for _, width := range testWidths {
		t.Run(fmt.Sprintf("Width_%d", width), func(t *testing.T) {
			barString := makeBar(50, width) // Use 50% as a sample percentage

			// Strip the color tags to measure the actual visible characters.
			cleanString := strings.Replace(barString, "[green]", "", 1)
			cleanString = strings.Replace(cleanString, "[white]", "", 1)

			actualWidth := len([]rune(cleanString))

			if actualWidth != width {
				t.Errorf("Bar width is incorrect. Got %d, want %d", actualWidth, width)
			}
		})
	}
}

func TestCalculateLabelInfo(t *testing.T) {
	state := &State{
		static: StaticInfo{
			ContainerCPULimit:      2,
			ContainerMemLimitBytes: 4 * bytesPerGB,
			ContainerMemLimitGB:    4,
			HostCores:              8,
			HostMemTotalGB:         16,
		},
		dynamic: DynamicInfo{
			ContainerCPUUsage:  75,
			ContainerMemUsedGB: 2,
			HostCPUUsage:       25,
			HostMemUsedGB:      8,
		},
	}

	cpuLabel, cpuInfo, cpuUsage := calculateCPULabelInfo(state)
	if !strings.Contains(cpuLabel, "75.0") || !strings.Contains(cpuInfo, "limit: 2.00 CPUs") || cpuUsage != 75 {
		t.Fatalf("Unexpected limited CPU label/info: %q %q %.1f", cpuLabel, cpuInfo, cpuUsage)
	}

	memLabel, memInfo, memPercent := calculateMEMLabelInfo(state)
	if !strings.Contains(memLabel, "50.0") || !strings.Contains(memInfo, "2.000 GB / 4.000 GB") || memPercent != 50 {
		t.Fatalf("Unexpected limited MEM label/info: %q %q %.1f", memLabel, memInfo, memPercent)
	}

	state.static.ContainerCPULimit = float64(state.static.HostCores)
	state.static.ContainerMemLimitBytes = 0
	state.static.ContainerMemLimitGB = 0
	cpuLabel, cpuInfo, cpuUsage = calculateCPULabelInfo(state)
	if !strings.Contains(cpuLabel, "25.0") || !strings.Contains(cpuInfo, "no cgroup limit") || cpuUsage != 25 {
		t.Fatalf("Unexpected host CPU label/info: %q %q %.1f", cpuLabel, cpuInfo, cpuUsage)
	}

	memLabel, memInfo, memPercent = calculateMEMLabelInfo(state)
	if !strings.Contains(memLabel, "50.0") || !strings.Contains(memInfo, "no cgroup limit") || memPercent != 50 {
		t.Fatalf("Unexpected host MEM label/info: %q %q %.1f", memLabel, memInfo, memPercent)
	}
}

func TestCalculateBarWidth(t *testing.T) {
	labels := []string{"CPU: [yellow]50.0%[white] ", "MEM: [yellow]10.0%[white] "}
	infos := []string{" [darkcyan](limit: 2.00 CPUs)[white]", " [darkcyan]1 GB / 4 GB[white]"}

	width := calculateBarWidth(120, labels, infos)
	if width <= minBarWidth {
		t.Fatalf("Expected wide terminal to allow a larger bar, got %d", width)
	}

	width = calculateBarWidth(1, labels, infos)
	if width != minBarWidth {
		t.Fatalf("Expected narrow terminal to use min bar width %d, got %d", minBarWidth, width)
	}
}

func TestStorageAndGPULabelInfo(t *testing.T) {
	state := &State{
		static: StaticInfo{
			GPUTotalGB: []float64{8},
		},
		dynamic: DynamicInfo{
			StorageUsage: []StorageUsage{
				{Path: "/", UsedGB: 10, FreeGB: 90, UsedPercent: 10},
				{Path: "/very/long/path/for/storage", UsedGB: 75, FreeGB: 25, UsedPercent: 75},
			},
			LiveGPUUsage: []GPUUsage{
				{Index: 0, Utilization: 80, MemUsedGB: 4},
				{Index: 1, Utilization: 30, MemUsedGB: 2},
			},
		},
	}

	labels, infos, percentages := calculateStorageLabelsInfo(state)
	if len(labels) != 2 || len(infos) != 2 || len(percentages) != 2 {
		t.Fatalf("Unexpected storage label slice sizes: %d %d %d", len(labels), len(infos), len(percentages))
	}
	if !strings.Contains(labels[0], "DISK /:") || !strings.Contains(infos[0], "10.00 GB / 100.00 GB") {
		t.Fatalf("Unexpected root storage label/info: %q %q", labels[0], infos[0])
	}
	if !strings.Contains(labels[1], ".../for/storage") {
		t.Fatalf("Expected long storage path to be truncated, got %q", labels[1])
	}
	if percentages[0] != 10 || percentages[1] != 75 {
		t.Fatalf("Unexpected storage percentages: %v", percentages)
	}

	labels, infos, percentages = calculateGPULabelsInfo(state)
	if len(labels) != 4 || len(infos) != 4 || len(percentages) != 4 {
		t.Fatalf("Unexpected GPU label slice sizes: %d %d %d", len(labels), len(infos), len(percentages))
	}
	if !strings.Contains(labels[0], "GPU0 Util:") || percentages[0] != 80 {
		t.Fatalf("Unexpected GPU util label/percent: %q %.1f", labels[0], percentages[0])
	}
	if !strings.Contains(labels[1], "GPU0 Mem:") ||
		!strings.Contains(infos[1], "4.00 GB / 8.00 GB") ||
		percentages[1] != 50 {
		t.Fatalf("Unexpected GPU memory label/info/percent: %q %q %.1f", labels[1], infos[1], percentages[1])
	}
	if percentages[3] != 0 {
		t.Fatalf("Expected missing GPU total to produce 0 memory percent, got %.1f", percentages[3])
	}
}

func TestBuildSectionsAndAlignedRows(t *testing.T) {
	labels := []string{
		"DISK /:        [yellow] 10.0%[white]",
		"DISK /data:    [yellow] 50.0%[white]",
		"DISK /logs:    [yellow] 90.0%[white]",
	}
	infos := []string{
		"[darkcyan]10.00 GB / 100.00 GB[white]",
		"[darkcyan]50.00 GB / 100.00 GB[white]",
		"[darkcyan]90.00 GB / 100.00 GB[white]",
	}
	percentages := []float64{10, 50, 90}
	maxLabelWidth, maxInfoWidth := calculateMaxWidthsFromSlices(labels, infos)
	layout := BarLayout{
		Columns:       2,
		BarWidth:      8,
		TotalWidth:    113,
		MaxLabelWidth: maxLabelWidth,
		MaxInfoWidth:  maxInfoWidth,
	}

	storageSection := buildStorageSection(layout, labels, infos, percentages)
	if !strings.Contains(storageSection, "DISK /:") || !strings.Contains(storageSection, "DISK /logs:") {
		t.Fatalf("Storage section is missing expected labels: %q", storageSection)
	}

	emptySection := buildStorageSection(layout, nil, nil, nil)
	if emptySection != "\n" {
		t.Fatalf("Expected empty storage section to be a separator newline, got %q", emptySection)
	}

	gpuSection := buildGPUSection(layout, labels[:2], infos[:2], percentages[:2])
	if !strings.Contains(gpuSection, "DISK /:") || !strings.Contains(gpuSection, "DISK /data:") {
		t.Fatalf("GPU section builder did not render provided bars: %q", gpuSection)
	}

	rows := makeAlignedMultiColumnBars([]BarData{
		{Label: labels[0], Percent: percentages[0], Info: infos[0]},
		{Label: labels[1], Percent: percentages[1], Info: infos[1]},
		{Label: labels[2], Percent: percentages[2], Info: infos[2]},
	}, layout)
	if len(rows) != 2 {
		t.Fatalf("Expected two aligned rows, got %d", len(rows))
	}
	if gotWidth := tview.TaggedStringWidth(rows[0]); gotWidth != layout.TotalWidth {
		t.Fatalf("Expected first row visible width %d, got %d: %q", layout.TotalWidth, gotWidth, rows[0])
	}

	emptyRows := makeAlignedMultiColumnBars(nil, layout)
	if emptyRows != nil {
		t.Fatalf("Expected nil rows for empty bars, got %v", emptyRows)
	}
}

func TestBuildCPUAndMemorySections(t *testing.T) {
	cpu := buildCPUSection(5, "CPU: ", " info", 40)
	if cpu != "CPU: [green]▓▓░░░[white] info\n" {
		t.Fatalf("Unexpected CPU section: %q", cpu)
	}
	mem := buildMemorySection(5, "MEM: ", " info", 60)
	if mem != "MEM: [green]▓▓▓░░[white] info\n" {
		t.Fatalf("Unexpected memory section: %q", mem)
	}
}

func TestUpdateInfoView(t *testing.T) {
	view := tview.NewTextView()
	height := updateInfoView(view)
	text := view.GetText(false)
	if height <= 0 {
		t.Fatalf("Expected positive info view height, got %d", height)
	}
	if !strings.Contains(text, "top inside a container") || !strings.Contains(text, "Quit: q") {
		t.Fatalf("Info view text missing expected content: %q", text)
	}
}

func TestUpdateProcessTable(t *testing.T) {
	table := tview.NewTable()
	state := &State{
		dynamic: DynamicInfo{
			Processes: []ProcessInfo{
				{PID: 123, User: "alice", CPUPercent: 12.5, MemPercent: 30, Command: "topic", GPUIndex: -1},
				{
					PID:           456,
					User:          "bob",
					CPUPercent:    1.5,
					MemPercent:    2.5,
					Command:       "python",
					GPUIndex:      0,
					GPUUtil:       70,
					GPUMemPercent: 40,
				},
			},
		},
	}

	updateProcessTable(table, state)
	if got := table.GetCell(0, 0).Text; got != "PID" {
		t.Fatalf("Expected PID header, got %q", got)
	}
	if got := table.GetCell(1, 0).Text; got != "123" {
		t.Fatalf("Expected first PID 123, got %q", got)
	}
	if got := table.GetCell(1, 4).Text; got != "-" {
		t.Fatalf("Expected no-GPU marker, got %q", got)
	}
	if got := table.GetCell(2, 4).Text; got != "70" {
		t.Fatalf("Expected GPU util 70, got %q", got)
	}

	state.dynamic.Processes = state.dynamic.Processes[:1]
	updateProcessTable(table, state)
	if rows := table.GetRowCount(); rows != 2 {
		t.Fatalf("Expected stale process rows to be removed, got %d rows", rows)
	}
}

func TestUpdateProcessListWithProvider(t *testing.T) {
	staticInfo := &StaticInfo{
		ContainerCPULimit:      2,
		ContainerMemLimitBytes: 4 * bytesPerGB,
		GPUCount:               1,
	}
	runner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi pmon -c 1 -s um": "0 1002 C 70 40 - - python",
		},
	}
	provider := MockProcessProvider{
		processes: []ProcessHandle{
			MockProcessHandle{pid: 1001, cpuPercent: 10, rss: bytesPerGB, cmdline: "", name: "worker"},
			MockProcessHandle{
				pid:        1002,
				cpuPercent: 80,
				rss:        2 * bytesPerGB,
				user:       "alice",
				cmdline:    "python train.py",
			},
			MockProcessHandle{pid: 1003, err: errors.New("skip")},
		},
	}

	processes := updateProcessListWithProvider(staticInfo, runner, provider)
	if len(processes) != 2 {
		t.Fatalf("Expected 2 valid processes, got %d", len(processes))
	}
	if processes[0].PID != 1002 || processes[1].PID != 1001 {
		t.Fatalf("Expected processes sorted by raw CPU, got %+v", processes)
	}
	if processes[0].CPUPercent != 40 || processes[0].MemPercent != 50 {
		t.Fatalf("Unexpected container percentages for GPU process: %+v", processes[0])
	}
	if processes[0].GPUIndex != 0 || processes[0].GPUUtil != 70 || processes[0].GPUMemPercent != 40 {
		t.Fatalf("Expected GPU data merged into process: %+v", processes[0])
	}
	if processes[1].User != "n/a" || processes[1].Command != "[worker]" {
		t.Fatalf("Expected username and command fallbacks, got %+v", processes[1])
	}
}

func TestUpdateProcessListWithProviderError(t *testing.T) {
	processes := updateProcessListWithProvider(
		&StaticInfo{},
		MockCommandRunner{},
		MockProcessProvider{err: errors.New("process failure")},
	)
	if processes != nil {
		t.Fatalf("Expected nil process list on provider error, got %+v", processes)
	}
}

// TestGetContainerMemLimit tests the logic for parsing memory limits from different cgroup versions.
func TestGetContainerMemLimit(t *testing.T) {
	testCases := []struct {
		name          string
		cgroupVersion CgroupVersion
		mockFiles     map[string]string
		expectedBytes int64
		expectedGB    float64
	}{
		{
			name:          "CgroupV2 with numeric limit",
			cgroupVersion: CgroupV2,
			mockFiles:     map[string]string{"/sys/fs/cgroup/memory.max": "8589934592"}, // 8 GB
			expectedBytes: 8589934592,
			expectedGB:    8.0,
		},
		{
			name:          "CgroupV2 with 'max' limit",
			cgroupVersion: CgroupV2,
			mockFiles:     map[string]string{"/sys/fs/cgroup/memory.max": "max"},
			expectedBytes: 0,
			expectedGB:    0.0,
		},
		{
			name:          "CgroupV1 with numeric limit",
			cgroupVersion: CgroupV1,
			mockFiles:     map[string]string{"/sys/fs/cgroup/memory/memory.limit_in_bytes": "4294967296"}, // 4 GB
			expectedBytes: 4294967296,
			expectedGB:    4.0,
		},
		{
			name:          "CgroupV1 with huge limit (over int64 max)",
			cgroupVersion: CgroupV1,
			// Larger than int64
			mockFiles: map[string]string{"/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854775808"},
			// Should fail parsing and return 0
			expectedBytes: 0,
			expectedGB:    0.0,
		},
		{
			name:          "File not found",
			cgroupVersion: CgroupV2,
			mockFiles:     map[string]string{}, // Empty mock filesystem
			expectedBytes: 0,
			expectedGB:    0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockFS := MockFileReader{files: tc.mockFiles}

			actualBytes, actualGB := getContainerMemLimit(tc.cgroupVersion, mockFS)

			if actualBytes != tc.expectedBytes {
				t.Errorf("Expected bytes %d, but got %d", tc.expectedBytes, actualBytes)
			}

			tolerance := 0.001
			if (actualGB-tc.expectedGB) > tolerance || (tc.expectedGB-actualGB) > tolerance {
				t.Errorf("Expected GB %.3f, but got %.3f", tc.expectedGB, actualGB)
			}
		})
	}
}

// TestGetContainerCPULimit tests the logic for parsing CPU limits.
func TestGetContainerCPULimit(t *testing.T) {
	testCases := []struct {
		name             string
		cgroupVersion    CgroupVersion
		mockFiles        map[string]string
		hostCores        int
		expectedCPULimit float64
	}{
		{
			name:             "CgroupV2 with numeric limit (2 cores)",
			cgroupVersion:    CgroupV2,
			mockFiles:        map[string]string{"/sys/fs/cgroup/cpu.max": "200000 100000"},
			hostCores:        8,
			expectedCPULimit: 2.0,
		},
		{
			name:             "CgroupV2 with 'max' limit",
			cgroupVersion:    CgroupV2,
			mockFiles:        map[string]string{"/sys/fs/cgroup/cpu.max": "max 100000"},
			hostCores:        8,
			expectedCPULimit: 8.0,
		},
		{
			name:             "CgroupV2 with malformed quota",
			cgroupVersion:    CgroupV2,
			mockFiles:        map[string]string{"/sys/fs/cgroup/cpu.max": "bad 100000"},
			hostCores:        8,
			expectedCPULimit: 8.0,
		},
		{
			name:             "CgroupV1 with numeric limit (0.5 cores)", //nolint:golines
			cgroupVersion:    CgroupV1,
			mockFiles:        map[string]string{"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "50000", "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"},
			hostCores:        4,
			expectedCPULimit: 0.5,
		},
		{
			name:             "CgroupV1 with file not found",
			cgroupVersion:    CgroupV1,
			mockFiles:        map[string]string{},
			hostCores:        4,
			expectedCPULimit: 4.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockFS := MockFileReader{files: tc.mockFiles}
			actualLimit := getContainerCPULimit(tc.cgroupVersion, tc.hostCores, mockFS)
			if actualLimit != tc.expectedCPULimit {
				t.Errorf("Expected CPU limit %.2f, but got %.2f", tc.expectedCPULimit, actualLimit)
			}
		})
	}
}

// TestUpdateContainerMemUsage tests the logic for parsing current memory usage.
func TestUpdateContainerMemUsage(t *testing.T) {
	testCases := []struct {
		name          string
		cgroupVersion CgroupVersion
		mockFiles     map[string]string
		expectedGB    float64
	}{
		{
			name:          "CgroupV2 with anon memory",
			cgroupVersion: CgroupV2,
			mockFiles:     map[string]string{"/sys/fs/cgroup/memory.stat": "anon 1073741824\nfile 512"}, // 1 GB
			expectedGB:    1.0,
		},
		{
			name:          "CgroupV1 with usage in bytes",
			cgroupVersion: CgroupV1,
			mockFiles:     map[string]string{"/sys/fs/cgroup/memory/memory.usage_in_bytes": "2147483648"}, // 2 GB
			expectedGB:    2.0,
		},
		{
			name:          "File not found",
			cgroupVersion: CgroupV2,
			mockFiles:     map[string]string{},
			expectedGB:    0.0,
		},
		{
			name:          "Malformed anon memory",
			cgroupVersion: CgroupV2,
			mockFiles:     map[string]string{"/sys/fs/cgroup/memory.stat": "anon not-a-number"},
			expectedGB:    0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockFS := MockFileReader{files: tc.mockFiles}
			actualGB := updateContainerMemUsage(tc.cgroupVersion, mockFS)
			tolerance := 0.001
			if (actualGB-tc.expectedGB) > tolerance || (tc.expectedGB-actualGB) > tolerance {
				t.Errorf("Expected GB %.3f, but got %.3f", tc.expectedGB, actualGB)
			}
		})
	}
}

// TestReadUintFromFile tests the helper for reading unsigned integers from files.
func TestReadUintFromFile(t *testing.T) {
	mockFS := MockFileReader{files: map[string]string{
		"valid":   "12345\n",
		"invalid": "not-a-number",
	}}

	// Valid number
	val, err := readUintFromFile("valid", mockFS)
	if err != nil {
		t.Errorf("Expected no error for valid file, but got %v", err)
	}
	if val != 12345 {
		t.Errorf("Expected value 12345, but got %d", val)
	}

	// Invalid number
	_, err = readUintFromFile("invalid", mockFS)
	if err == nil {
		t.Error("Expected an error for invalid file, but got nil")
	}

	// File not found
	_, err = readUintFromFile("nonexistent", mockFS)
	if err == nil {
		t.Error("Expected an error for nonexistent file, but got nil")
	}
}

// TestUpdateContainerCPUUsage tests the time-based CPU usage calculation.
func TestUpdateContainerCPUUsage(t *testing.T) {
	testCases := []struct {
		name            string
		cgroupVersion   CgroupVersion
		cpuLimit        float64
		prevUsage       uint64
		mockFiles       map[string]string
		expectedPercent float64
	}{
		{
			name:            "CgroupV2 50% usage over 1 second", //nolint:golines
			cgroupVersion:   CgroupV2,
			cpuLimit:        2.0, // 2 cores
			prevUsage:       1000000,
			mockFiles:       map[string]string{"/sys/fs/cgroup/cpu.stat": "usage_usec 2000000"}, // 1,000,000 us (1 sec) used
			expectedPercent: 50.0,
		},
		{
			name:            "CgroupV1 25% usage over 1 second", //nolint:golines
			cgroupVersion:   CgroupV1,
			cpuLimit:        4.0, // 4 cores
			prevUsage:       500000,
			mockFiles:       map[string]string{"/sys/fs/cgroup/cpuacct/cpuacct.usage": "1500000000"}, // 1,000,000,000 ns (1 sec) used
			expectedPercent: 25.0,
		},
		{
			name:          "CgroupV2 usage exceeds 100% and is capped",
			cgroupVersion: CgroupV2,
			cpuLimit:      1.0, // 1 core
			prevUsage:     0,
			// CORRECTED: The mock file content must match what the parser expects.
			mockFiles:       map[string]string{"/sys/fs/cgroup/cpu.stat": "usage_usec 1200000"}, // 1.2 seconds of usage
			expectedPercent: 100.0,                                                              // Should be capped to 100
		},
		{
			name:            "No change in usage",
			cgroupVersion:   CgroupV2,
			cpuLimit:        2.0,
			prevUsage:       1000000,
			mockFiles:       map[string]string{"/sys/fs/cgroup/cpu.stat": "usage_usec 1000000"},
			expectedPercent: 0.0,
		},
		{
			name:            "Zero CPU limit does not divide by zero",
			cgroupVersion:   CgroupV2,
			cpuLimit:        0.0,
			prevUsage:       1000000,
			mockFiles:       map[string]string{"/sys/fs/cgroup/cpu.stat": "usage_usec 2000000"},
			expectedPercent: 0.0,
		},
		{
			name:            "Malformed usage is ignored",
			cgroupVersion:   CgroupV2,
			cpuLimit:        2.0,
			prevUsage:       1000000,
			mockFiles:       map[string]string{"/sys/fs/cgroup/cpu.stat": "usage_usec nope"},
			expectedPercent: 0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockFS := MockFileReader{files: tc.mockFiles}
			state := &State{
				static: StaticInfo{
					CgroupVersion:     tc.cgroupVersion,
					ContainerCPULimit: tc.cpuLimit,
				},
				prevCPUUsage: tc.prevUsage,
				// Simulate that the previous measurement was taken exactly 1 second ago.
				prevCPUTime: time.Now().Add(-1 * time.Second),
			}

			actualPercent := updateContainerCPUUsage(state, mockFS)
			tolerance := 0.01
			if (actualPercent-tc.expectedPercent) > tolerance || (tc.expectedPercent-actualPercent) > tolerance {
				t.Errorf("Expected CPU percent %.2f, but got %.2f", tc.expectedPercent, actualPercent)
			}
		})
	}
}

func TestOSCommandRunnerTimeout(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep is required for timeout test")
	}

	runner := OSCommandRunner{Timeout: 10 * time.Millisecond}
	start := time.Now()
	_, err := runner.Output("sleep", "1")
	if err == nil {
		t.Fatal("Expected timeout error, got nil")
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Fatalf("Command timeout took too long: %s", elapsed)
	}
}

// TestGetGPUProcessMap tests the parsing of nvidia-smi pmon output.
func TestGetGPUProcessMap(t *testing.T) {
	mockPmonOutput := `# gpu     pid  type    sm   mem   enc   dec   command
# Idx       #   C/G     %     %     %     %   name
    0   20131     C    15     8     -     -   python
    0   20567     C     -     -     -     -   -
    1   34567     G    22    15     -     -   Xorg`

	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi pmon -c 1 -s um": mockPmonOutput,
		},
	}

	processMap := getGPUProcessMap(mockRunner)

	// Check if the map has the correct number of entries.
	if len(processMap) != 3 {
		t.Fatalf("Expected 3 processes in the map, but got %d", len(processMap))
	}

	// Check a specific compute process.
	p1, ok := processMap[20131]
	if !ok {
		t.Fatal("Process with PID 20131 not found in map")
	}
	if p1.GPUIndex != 0 || p1.GPUUtil != 15 || p1.GPUMemUtil != 8 {
		t.Errorf("Process 20131 has incorrect data: got %+v, want {GPUIndex:0 GPUUtil:15 GPUMemUtil:8}", p1)
	}

	// Check a specific graphics process on a different GPU.
	p2, ok := processMap[34567]
	if !ok {
		t.Fatal("Process with PID 34567 not found in map")
	}
	if p2.GPUIndex != 1 || p2.GPUUtil != 22 || p2.GPUMemUtil != 15 {
		t.Errorf("Process 34567 has incorrect data: got %+v, want {GPUIndex:1 GPUUtil:22 GPUMemUtil:15}", p2)
	}

	// Check a process with no utilization data ('-').
	p3, ok := processMap[20567]
	if !ok {
		t.Fatal("Process with PID 20567 not found in map")
	}
	if p3.GPUIndex != 0 || p3.GPUUtil != 0 || p3.GPUMemUtil != 0 {
		t.Errorf("Process 20567 has incorrect data: got %+v, want {GPUIndex:0 GPUUtil:0 GPUMemUtil:0}", p3)
	}
}

func TestGetGPUProcessMapSkipsMalformedRows(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi pmon -c 1 -s um": `# gpu pid type sm mem enc dec command
    bad 20131 C 15 8 - - python
    0 nope C 15 8 - - python
    0 20132 C nope 8 - - python
    0 20133 C 15 nope - - python
    1 20134 C 22 15 - - python`,
		},
	}

	processMap := getGPUProcessMap(mockRunner)
	if len(processMap) != 1 {
		t.Fatalf("Expected 1 valid process, got %d", len(processMap))
	}
	if _, ok := processMap[20134]; !ok {
		t.Fatal("Expected valid PID 20134 in process map")
	}
}

// TestGetStaticGPUInfo tests the parsing of nvidia-smi for static GPU info.
func TestGetStaticGPUInfo(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits": "8192\n16384\n",
		},
	}

	count, totals := getStaticGPUInfo(mockRunner)

	if count != 2 {
		t.Fatalf("Expected GPU count of 2, but got %d", count)
	}
	if len(totals) != 2 {
		t.Fatalf("Expected totals slice of length 2, but got %d", len(totals))
	}
	if totals[0] != 8.0 || totals[1] != 16.0 {
		t.Errorf("Expected totals [8.0, 16.0], but got %v", totals)
	}
}

func TestGetStaticGPUInfoSkipsMalformedRows(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits": "8192\nnot-a-number\n16384\n",
		},
	}

	count, totals := getStaticGPUInfo(mockRunner)
	if count != 2 {
		t.Fatalf("Expected GPU count of 2, got %d", count)
	}
	if len(totals) != 2 || totals[0] != 8.0 || totals[1] != 16.0 {
		t.Fatalf("Expected valid totals [8 16], got %v", totals)
	}
}

// TestUpdateLiveGPUUsage tests the parsing of nvidia-smi for live GPU usage.
func TestUpdateLiveGPUUsage(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits": "50, 2048\n75, 4096\n",
		},
	}

	usage := updateLiveGPUUsage(2, mockRunner)

	if len(usage) != 2 {
		t.Fatalf("Expected usage slice of length 2, but got %d", len(usage))
	}
	if usage[0].Utilization != 50 || usage[0].MemUsedGB != 2.0 {
		t.Errorf("GPU 0 usage mismatch: got %+v, want {Utilization:50 MemUsedGB:2.0}", usage[0])
	}
	if usage[1].Utilization != 75 || usage[1].MemUsedGB != 4.0 {
		t.Errorf("GPU 1 usage mismatch: got %+v, want {Utilization:75 MemUsedGB:4.0}", usage[1])
	}
}

func TestUpdateLiveGPUUsageSkipsMalformedRows(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits": "bad, 2048\n75, nope\n80, 1024\n",
		},
	}

	usage := updateLiveGPUUsage(3, mockRunner)
	if len(usage) != 1 {
		t.Fatalf("Expected one valid GPU usage row, got %d", len(usage))
	}
	if usage[0].Index != 2 || usage[0].Utilization != 80 || usage[0].MemUsedGB != 1.0 {
		t.Fatalf("Unexpected parsed GPU usage: %+v", usage[0])
	}
}

func TestUpdateProcessListSkipsGPUProbeWhenNoGPU(t *testing.T) {
	runner := &CountingCommandRunner{}
	staticInfo := &StaticInfo{GPUCount: 0}

	_ = updateProcessList(staticInfo, runner)
	if runner.calls != 0 {
		t.Fatalf("Expected no GPU probe calls, got %d", runner.calls)
	}
}

// TestGetStaticInfo tests the main static info gathering function.
func TestGetStaticInfo(t *testing.T) {
	testCases := []struct {
		name                  string
		mockStater            Stater
		mockFS                FileReader
		mockRunner            CommandRunner
		expectedCgroupVersion CgroupVersion
		expectedCPULimit      float64
		expectedGPUCount      int
	}{
		{
			name:       "CgroupV2 environment with GPU",
			mockStater: MockStater{FileDoesNotExist: false}, // cpu.max exists
			mockFS: MockFileReader{files: map[string]string{
				"/sys/fs/cgroup/cpu.max":    "400000 100000",
				"/sys/fs/cgroup/memory.max": "17179869184",
			}},
			mockRunner: MockCommandRunner{outputs: map[string]string{
				"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits": "8192",
			}},
			expectedCgroupVersion: CgroupV2,
			expectedCPULimit:      4.0,
			expectedGPUCount:      1,
		},
		{
			name:       "CgroupV1 environment without GPU",
			mockStater: MockStater{FileDoesNotExist: true}, // cpu.max does NOT exist
			mockFS: MockFileReader{files: map[string]string{
				"/sys/fs/cgroup/cpu/cpu.cfs_quota_us":         "50000",
				"/sys/fs/cgroup/cpu/cpu.cfs_period_us":        "100000",
				"/sys/fs/cgroup/memory/memory.limit_in_bytes": "4294967296",
			}},
			mockRunner:            MockCommandRunner{err: errors.New("nvidia-smi not found")}, // No GPU
			expectedCgroupVersion: CgroupV1,
			expectedCPULimit:      0.5,
			expectedGPUCount:      0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			info, err := getStaticInfo(tc.mockFS, tc.mockRunner, tc.mockStater)
			if err != nil {
				t.Fatalf("getStaticInfo returned an unexpected error: %v", err)
			}

			if info.CgroupVersion != tc.expectedCgroupVersion {
				t.Errorf("Expected CgroupVersion %v, got %v", tc.expectedCgroupVersion, info.CgroupVersion)
			}
			if info.ContainerCPULimit != tc.expectedCPULimit {
				t.Errorf("Expected ContainerCPULimit %.2f, got %.2f", tc.expectedCPULimit, info.ContainerCPULimit)
			}
			if info.GPUCount != tc.expectedGPUCount {
				t.Errorf("Expected GPUCount %d, got %d", tc.expectedGPUCount, info.GPUCount)
			}
		})
	}
}

// TestUpdateProcessList tests the merging of process information with GPU data.
func TestUpdateProcessList(t *testing.T) {
	pid := int32(os.Getpid())

	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi pmon -c 1 -s um": fmt.Sprintf("0 %d C 50 25 - - test", pid),
		},
	}

	staticInfo := &StaticInfo{
		ContainerCPULimit:      4.0,
		ContainerMemLimitBytes: 8 * 1024 * 1024 * 1024, // 8 GB
		GPUCount:               1,
		GPUTotalGB:             []float64{16.0},
	}

	processList := updateProcessList(staticInfo, mockRunner)

	foundTestProcess := false
	for _, p := range processList {
		if p.PID == pid {
			foundTestProcess = true
			if p.GPUIndex != 0 {
				t.Errorf("Expected GPUIndex to be 0, got %d", p.GPUIndex)
			}
			if p.GPUUtil != 50 {
				t.Errorf("Expected GPUUtil to be 50, got %d", p.GPUUtil)
			}
			if p.GPUMemPercent != 25.0 {
				t.Errorf("Expected GPUMemPercent to be 25.0, got %f", p.GPUMemPercent)
			}
			break
		}
	}

	if !foundTestProcess {
		t.Fatalf("Test process with PID %d was not found in the process list", pid)
	}
}

// TestUpdateAll tests the coordinator function that fetches all dynamic data.
func TestUpdateAll(t *testing.T) {
	mockFS := MockFileReader{files: map[string]string{
		"/sys/fs/cgroup/cpu.stat":    "usage_usec 2000000",
		"/sys/fs/cgroup/memory.stat": "anon 1073741824",
	}}
	mockRunner := MockCommandRunner{outputs: map[string]string{
		"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits": "50, 2048",
		"nvidia-smi pmon -c 1 -s um": "# gpu pid type sm mem enc dec command\n0 123 C 10 5 - - test",
	}}

	state := &State{
		static: StaticInfo{
			CgroupVersion:     CgroupV2,
			ContainerCPULimit: 2.0,
			GPUCount:          1,
		},
		prevCPUUsage: 1000000,
		prevCPUTime:  time.Now().Add(-1 * time.Second),
	}

	// Run the function under test.
	updateAll(state, mockFS, mockRunner)

	tolerance := 0.01

	// Assert that the dynamic state has been populated correctly.
	if (state.dynamic.ContainerCPUUsage-50.0) > tolerance || (50.0-state.dynamic.ContainerCPUUsage) > tolerance {
		t.Errorf("Expected ContainerCPUUsage to be close to 50.0, got %.5f", state.dynamic.ContainerCPUUsage)
	}
	if (state.dynamic.ContainerMemUsedGB-1.0) > tolerance || (1.0-state.dynamic.ContainerMemUsedGB) > tolerance {
		t.Errorf("Expected ContainerMemUsedGB to be close to 1.0, got %.2f", state.dynamic.ContainerMemUsedGB)
	}
	if len(state.dynamic.LiveGPUUsage) != 1 || state.dynamic.LiveGPUUsage[0].Utilization != 50 {
		t.Errorf("Expected LiveGPUUsage to be updated correctly, got %+v", state.dynamic.LiveGPUUsage)
	}
	// A simple check that the process list was populated.
	if len(state.dynamic.Processes) == 0 {
		t.Error("Expected Processes list to be populated, but it was empty")
	}
}

// TestShouldSkipFilesystem tests the filesystem filtering logic.
func TestShouldSkipFilesystem(t *testing.T) {
	testCases := []struct {
		name       string
		fstype     string
		mountpoint string
		expected   bool
	}{
		{name: "Skip tmpfs", fstype: "tmpfs", mountpoint: "/tmp", expected: true},
		{name: "Skip proc", fstype: "proc", mountpoint: "/proc", expected: true},
		{name: "Skip sys mount", fstype: "ext4", mountpoint: "/sys/something", expected: true},
		{name: "Skip dev mount", fstype: "ext4", mountpoint: "/dev/shm", expected: true},
		{name: "Keep root filesystem", fstype: "ext4", mountpoint: "/", expected: false},
		{name: "Keep home filesystem", fstype: "ext4", mountpoint: "/home", expected: false},
		{name: "Keep data filesystem", fstype: "xfs", mountpoint: "/data", expected: false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := shouldSkipFilesystem(tc.fstype, tc.mountpoint)
			if result != tc.expected {
				t.Errorf("shouldSkipFilesystem(%q, %q) = %v, expected %v",
					tc.fstype, tc.mountpoint, result, tc.expected)
			}
		})
	}
}
