// ./pkg/main_test.go
package main

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
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
			name:             "CgroupV1 with numeric limit (0.5 cores)",
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
			name:            "CgroupV2 50% usage over 1 second",
			cgroupVersion:   CgroupV2,
			cpuLimit:        2.0, // 2 cores
			prevUsage:       1000000,
			mockFiles:       map[string]string{"/sys/fs/cgroup/cpu.stat": "usage_usec 2000000"}, // 1,000,000 us (1 sec) used
			expectedPercent: 50.0,
		},
		{
			name:            "CgroupV1 25% usage over 1 second",
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

// TestCalculateBarLayout tests the dynamic bar layout calculation.
func TestCalculateBarLayout(t *testing.T) {
	testCases := []struct {
		name           string
		availableWidth int
		numBars        int
		expectedCols   int
		expectedWidth  int
	}{
		{name: "Wide space, single bar", availableWidth: 150, numBars: 1, expectedCols: 1, expectedWidth: 100},          // 150 - 50 = 100
		{name: "Space for two columns", availableWidth: 150, numBars: 2, expectedCols: 2, expectedWidth: 22},            // (150-5)/2 - 50 = 22.5 -> 22
		{name: "Narrow space, force single column", availableWidth: 80, numBars: 3, expectedCols: 1, expectedWidth: 30}, // 80 - 50 = 30
		{name: "Exact minimum width", availableWidth: 70, numBars: 1, expectedCols: 1, expectedWidth: 20},               // 70 - 50 = 20
		{name: "Below minimum width", availableWidth: 15, numBars: 1, expectedCols: 1, expectedWidth: 20},               // Should fallback to minBarWidth
		{name: "No bars", availableWidth: 100, numBars: 0, expectedCols: 1, expectedWidth: 20},                          // Should fallback
		{name: "Space for three columns", availableWidth: 230, numBars: 3, expectedCols: 3, expectedWidth: 23},          // (230-10)/3 - 50 = 23.33 -> 23
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			layout := calculateBarLayout(tc.availableWidth, tc.numBars)
			if layout.Columns != tc.expectedCols {
				t.Errorf("Expected %d columns, got %d", tc.expectedCols, layout.Columns)
			}
			if layout.BarWidth != tc.expectedWidth {
				t.Errorf("Expected bar width %d, got %d", tc.expectedWidth, layout.BarWidth)
			}
		})
	}
}

// TestMakeMultiColumnBars tests the multi-column bar rendering.
func TestMakeMultiColumnBars(t *testing.T) {
	testCases := []struct {
		name         string
		bars         []BarData
		layout       BarLayout
		expectedRows int
	}{
		{
			name:         "Single bar, single column",
			bars:         []BarData{{Label: "Test", Percent: 50, Info: ""}},
			layout:       BarLayout{Columns: 1, BarWidth: 20, TotalWidth: 20},
			expectedRows: 1,
		},
		{
			name: "Two bars, two columns",
			bars: []BarData{
				{Label: "Test1", Percent: 50, Info: ""},
				{Label: "Test2", Percent: 75, Info: ""},
			},
			layout:       BarLayout{Columns: 2, BarWidth: 20, TotalWidth: 45},
			expectedRows: 1,
		},
		{
			name: "Three bars, two columns",
			bars: []BarData{
				{Label: "Test1", Percent: 50, Info: ""},
				{Label: "Test2", Percent: 75, Info: ""},
				{Label: "Test3", Percent: 25, Info: ""},
			},
			layout:       BarLayout{Columns: 2, BarWidth: 20, TotalWidth: 45},
			expectedRows: 2,
		},
		{
			name:         "Empty bars",
			bars:         []BarData{},
			layout:       BarLayout{Columns: 1, BarWidth: 20, TotalWidth: 20},
			expectedRows: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := makeMultiColumnBars(tc.bars, tc.layout)
			if len(result) != tc.expectedRows {
				t.Errorf("Expected %d rows, got %d", tc.expectedRows, len(result))
			}

			// Verify each row has the expected structure
			for i, row := range result {
				// Strip color tags for length checking
				cleanRow := strings.Replace(row, "[green]", "", -1)
				cleanRow = strings.Replace(cleanRow, "[white]", "", -1)

				// Should contain bar characters
				if tc.expectedRows > 0 && !strings.ContainsAny(cleanRow, "▓░") {
					t.Errorf("Row %d should contain bar characters, got: %s", i, cleanRow)
				}
			}
		})
	}
}

// TestCalculateGPUBarLayout tests the GPU-specific bar layout calculation with column limits.
func TestCalculateGPUBarLayout(t *testing.T) {
	testCases := []struct {
		name           string
		availableWidth int
		numBars        int
		expectedCols   int
		expectedWidth  int
	}{
		{name: "Single GPU, single column", availableWidth: 100, numBars: 2, expectedCols: 1, expectedWidth: 60},        // 100 - 40 = 60
		{name: "Wide space for two columns", availableWidth: 150, numBars: 4, expectedCols: 2, expectedWidth: 32},       // (150-5)/2 - 40 = 32.5 -> 32
		{name: "Many GPUs, still max 2 columns", availableWidth: 300, numBars: 8, expectedCols: 2, expectedWidth: 107},  // (300-5)/2 - 40 = 107.5 -> 107
		{name: "Single bar, force single column", availableWidth: 200, numBars: 1, expectedCols: 1, expectedWidth: 160}, // Single bar gets single column
		{name: "Narrow space, fallback", availableWidth: 50, numBars: 4, expectedCols: 1, expectedWidth: 20},            // Too narrow, fallback
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			layout := calculateGPUBarLayout(tc.availableWidth, tc.numBars)
			if layout.Columns != tc.expectedCols {
				t.Errorf("Expected %d columns, got %d", tc.expectedCols, layout.Columns)
			}
			if layout.BarWidth != tc.expectedWidth {
				t.Errorf("Expected bar width %d, got %d", tc.expectedWidth, layout.BarWidth)
			}
		})
	}
}
