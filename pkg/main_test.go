// ./pkg/main_test.go
package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	netio "github.com/shirou/gopsutil/v3/net"
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

type fakeFileInfo struct{}

func (fakeFileInfo) Name() string {
	return "fake"
}

func (fakeFileInfo) Size() int64 {
	return 0
}

func (fakeFileInfo) Mode() os.FileMode {
	return 0
}

func (fakeFileInfo) ModTime() time.Time {
	return time.Time{}
}

func (fakeFileInfo) IsDir() bool {
	return false
}

func (fakeFileInfo) Sys() any {
	return nil
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
	ppid       int32
	cpuPercent float64
	rss        uint64
	user       string
	cmdline    string
	name       string
	createTime int64
	threads    int32
	openFiles  int
	err        error
}

func (mph MockProcessHandle) PID() int32 {
	return mph.pid
}

func (mph MockProcessHandle) Ppid() (int32, error) {
	return mph.ppid, mph.err
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

func (mph MockProcessHandle) CreateTime() (int64, error) {
	return mph.createTime, mph.err
}

func (mph MockProcessHandle) NumThreads() (int32, error) {
	return mph.threads, mph.err
}

func (mph MockProcessHandle) OpenFiles() ([]process.OpenFilesStat, error) {
	if mph.err != nil {
		return nil, mph.err
	}
	files := make([]process.OpenFilesStat, mph.openFiles)
	return files, nil
}

type MockProcessSignaler struct {
	calls []signalCall
	err   error
}

type signalCall struct {
	pid    int32
	signal os.Signal
}

func (mps *MockProcessSignaler) Signal(pid int32, signal os.Signal) error {
	mps.calls = append(mps.calls, signalCall{pid: pid, signal: signal})
	return mps.err
}

type MockHostMetricsProvider struct {
	cores      int
	cpuPercent []float64
	memory     *mem.VirtualMemoryStat
	err        error
}

func (mhp MockHostMetricsProvider) CPUCounts(_ bool) (int, error) {
	if mhp.err != nil {
		return 0, mhp.err
	}
	return mhp.cores, nil
}

func (mhp MockHostMetricsProvider) CPUPercent(_ time.Duration, _ bool) ([]float64, error) {
	if mhp.err != nil {
		return nil, mhp.err
	}
	return mhp.cpuPercent, nil
}

func (mhp MockHostMetricsProvider) VirtualMemory() (*mem.VirtualMemoryStat, error) {
	if mhp.err != nil {
		return nil, mhp.err
	}
	return mhp.memory, nil
}

type MockStorageProvider struct {
	partitions []disk.PartitionStat
	usages     map[string]*disk.UsageStat
	err        error
}

func (msp MockStorageProvider) Partitions(_ bool) ([]disk.PartitionStat, error) {
	if msp.err != nil {
		return nil, msp.err
	}
	return msp.partitions, nil
}

func (msp MockStorageProvider) Usage(path string) (*disk.UsageStat, error) {
	if msp.err != nil {
		return nil, msp.err
	}
	usage, ok := msp.usages[path]
	if !ok {
		return nil, errors.New("missing usage")
	}
	return usage, nil
}

type MockNetworkProvider struct {
	counters []netio.IOCountersStat
	err      error
}

func (mnp MockNetworkProvider) IOCounters(_ bool) ([]netio.IOCountersStat, error) {
	return mnp.counters, mnp.err
}

type MockDiskIOProvider struct {
	counters map[string]disk.IOCountersStat
	err      error
}

func (mdp MockDiskIOProvider) IOCounters(_ ...string) (map[string]disk.IOCountersStat, error) {
	return mdp.counters, mdp.err
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
	state := &State{ui: UIState{HideASCIIArt: true, ProcessSort: SortByCPU}}
	height := updateInfoView(view, state)
	text := view.GetText(false)
	if height <= 0 {
		t.Fatalf("Expected positive info view height, got %d", height)
	}
	if strings.Contains(text, "░████") {
		t.Fatalf("Default compact info view should not show ASCII art: %q", text)
	}
	if !strings.Contains(text, "top inside a container") || !strings.Contains(text, "Keys: q quit") {
		t.Fatalf("Compact info view text missing expected content: %q", text)
	}
	if height > 5 {
		t.Fatalf("Compact info view should stay short, got height %d and text %q", height, text)
	}

	state.ui.HideASCIIArt = false
	height = updateInfoView(view, state)
	text = view.GetText(false)
	if height <= 0 {
		t.Fatalf("Expected positive ASCII info view height, got %d", height)
	}
	if !strings.Contains(text, "░████") || !strings.Contains(text, "Logo: a") {
		t.Fatalf("ASCII info view did not show logo or toggle help: %q", text)
	}
}

func TestCompactInfoPanelLeavesProcessTableSpace(t *testing.T) {
	state := &State{
		static: StaticInfo{
			ContainerCPULimit:      12,
			ContainerMemLimitBytes: 30 * bytesPerGB,
			ContainerMemLimitGB:    30,
			HostCores:              12,
			GPUCount:               1,
			GPUTotalGB:             []float64{16},
		},
		ui: UIState{HideASCIIArt: true, ProcessSort: SortByCPU},
		dynamic: DynamicInfo{
			HostCPUUsage:       13.7,
			ContainerMemUsedGB: 14.9,
			StorageUsage: []StorageUsage{
				{Path: "/", UsedGB: 296.14, FreeGB: 312.86, UsedPercent: 48.6},
				{Path: "/boot/efi", UsedGB: 0.04, FreeGB: 0.15, UsedPercent: 19.4},
			},
			LiveGPUUsage: []GPUUsage{{Index: 0, Utilization: 6, MemUsedGB: 1.31}},
			NetworkUsage: []NetworkUsage{
				{
					Name:          "enp11s0",
					RXBytesPerSec: 0.10 * bytesPerSecondToMiBSecond,
					TXBytesPerSec: 0.09 * bytesPerSecondToMiBSecond,
				},
			},
			DiskIOUsage: []DiskIOUsage{
				{
					Name:             "nvme0n1p6",
					ReadBytesPerSec:  154.72 * bytesPerSecondToMiBSecond,
					WriteBytesPerSec: 13.92 * bytesPerSecondToMiBSecond,
				},
			},
			Pressure:  []PressureStat{{Resource: "cpu"}},
			Processes: []ProcessInfo{{PID: 1, User: "root", CPUPercent: 1, Command: "topic", GPUIndex: -1}},
		},
	}
	for i := range historySize {
		state.history.CPU.Add(float64(i))
		state.history.Memory.Add(float64(i))
		state.history.GPU.Add(float64(i))
	}

	resourceView := tview.NewTextView().SetDynamicColors(true).SetWrap(false)
	infoView := tview.NewTextView().SetDynamicColors(true).SetTextAlign(tview.AlignLeft)
	processTable := configureProcessTable(tview.NewTable())
	topPanel := tview.NewFlex().
		SetDirection(tview.FlexColumn).
		AddItem(resourceView, 0, 1, false).
		AddItem(infoView, compactInfoPanelWidth, 0, false)
	topPanel.SetBorder(true).SetTitle(" System ")
	mainLayout := tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(topPanel, placeholderHeight, 0, false).
		AddItem(processTable, 0, 1, true)

	screen := tcell.NewSimulationScreen("UTF-8")
	app := tview.NewApplication().SetScreen(screen).SetRoot(mainLayout, true)
	screen.SetSize(200, 50)
	leftHeight := updateResourceView(resourceView, state)
	rightHeight := updateInfoView(infoView, state)
	topPanel.ResizeItem(infoView, currentInfoPanelWidth(state), 0)
	mainLayout.ResizeItem(topPanel, max(leftHeight, rightHeight)+borderHeight, 0)
	updateProcessTable(processTable, state)
	app.ForceDraw()

	_, _, _, topHeight := topPanel.GetRect()
	_, _, _, tableHeight := processTable.GetRect()
	if topHeight > 14 {
		t.Fatalf("Expected compact top panel height to stay at or below 14 rows, got %d", topHeight)
	}
	if tableHeight <= topHeight*2 {
		t.Fatalf("Expected process table to get most of the viewport, top=%d table=%d", topHeight, tableHeight)
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
		ui: UIState{ProcessSort: SortByCPU},
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

func TestPrepareProcessRowsSortFilterReverse(t *testing.T) {
	processes := []ProcessInfo{
		{PID: 100, User: "alice", CPUPercent: 10, MemPercent: 80, Command: "worker"},
		{PID: 200, User: "bob", CPUPercent: 90, MemPercent: 20, Command: "api-server"},
		{PID: 300, User: "carol", CPUPercent: 50, MemPercent: 40, Command: "batch-worker"},
	}

	rows := prepareProcessRows(append([]ProcessInfo(nil), processes...), UIState{ProcessSort: SortByMemory})
	if rows[0].PID != 100 {
		t.Fatalf("Expected memory sort to put PID 100 first, got %+v", rows)
	}

	rows = prepareProcessRows(append([]ProcessInfo(nil), processes...), UIState{
		ProcessSort:   SortByCPU,
		ReverseSort:   true,
		ProcessFilter: "worker",
	})
	if len(rows) != 2 {
		t.Fatalf("Expected two filtered worker rows, got %+v", rows)
	}
	if rows[0].PID != 100 || rows[1].PID != 300 {
		t.Fatalf("Expected reverse CPU order for filtered rows, got %+v", rows)
	}
}

func TestBuildProcessTreeRows(t *testing.T) {
	processes := []ProcessInfo{
		{PID: 3, ParentPID: 2, CPUPercent: 20, Command: "grandchild"},
		{PID: 1, CPUPercent: 10, Command: "root"},
		{PID: 4, ParentPID: 99, CPUPercent: 5, Command: "orphan"},
		{PID: 2, ParentPID: 1, CPUPercent: 80, Command: "child"},
	}

	rows := buildProcessTreeRows(processes)
	if len(rows) != len(processes) {
		t.Fatalf("Expected %d rows, got %+v", len(processes), rows)
	}
	if rows[0].PID != 1 || rows[1].PID != 2 || rows[2].PID != 3 || rows[3].PID != 4 {
		t.Fatalf("Unexpected tree order: %+v", rows)
	}
	if !strings.Contains(rows[1].Command, "└─ child") || !strings.Contains(rows[2].Command, "└─ grandchild") {
		t.Fatalf("Expected child rows to be indented, got %+v", rows)
	}
}

func TestProcessDetailsText(t *testing.T) {
	process := ProcessInfo{
		PID:           123,
		ParentPID:     1,
		User:          "alice",
		CPUPercent:    12.5,
		MemPercent:    33.3,
		Command:       "topic",
		StartTime:     time.Unix(10, 0),
		ThreadCount:   4,
		OpenFileCount: 8,
		GPUIndex:      0,
		GPUUtil:       70,
		GPUMemPercent: 40,
	}

	text := processDetailsText(process)
	for _, want := range []string{"PID: 123", "Parent PID: 1", "alice", "GPU0", "Threads: 4", "Open files: 8", "topic"} {
		if !strings.Contains(text, want) {
			t.Fatalf("Expected process details to contain %q, got %q", want, text)
		}
	}
}

func TestSelectedProcessUsesPreparedRows(t *testing.T) {
	table := tview.NewTable()
	table.Select(1, 0)
	state := &State{
		ui: UIState{ProcessSort: SortByMemory},
		dynamic: DynamicInfo{
			Processes: []ProcessInfo{
				{PID: 1, CPUPercent: 90, MemPercent: 10, Command: "cpu"},
				{PID: 2, CPUPercent: 10, MemPercent: 90, Command: "mem"},
			},
		},
	}

	process, ok := selectedProcess(table, state)
	if !ok || process.PID != 2 {
		t.Fatalf("Expected selected row to follow prepared sort order, got process=%+v ok=%v", process, ok)
	}
}

func TestSendProcessSignal(t *testing.T) {
	pages := tview.NewPages()
	signaler := &MockProcessSignaler{}
	sendProcessSignal(pages, signaler, 123, os.Interrupt)

	if len(signaler.calls) != 1 || signaler.calls[0].pid != 123 || signaler.calls[0].signal != os.Interrupt {
		t.Fatalf("Expected signal call, got %+v", signaler.calls)
	}
	if signalLabel(os.Interrupt) != "SIGINT" {
		t.Fatalf("Expected SIGINT label for os.Interrupt")
	}
}

func TestHandleInputUpdatesUIState(t *testing.T) {
	state := &State{ui: UIState{ProcessSort: SortByCPU}}
	app := tview.NewApplication()
	pages := tview.NewPages()
	table := tview.NewTable()
	signaler := &MockProcessSignaler{}

	got := handleInput(tcell.NewEventKey(tcell.KeyRune, '/', tcell.ModNone), state, app, pages, table, signaler)
	if got != nil {
		t.Fatalf("Expected filter key to be consumed, got %v", got)
	}
	if !state.ui.SearchMode || state.ui.ProcessFilter != "" {
		t.Fatalf("Expected search mode with empty filter, got %+v", state.ui)
	}

	handleInput(tcell.NewEventKey(tcell.KeyRune, 'p', tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyRune, 'y', tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyBackspace, 0, tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyRune, 't', tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyEnter, 0, tcell.ModNone), state, app, pages, table, signaler)
	if state.ui.SearchMode || state.ui.ProcessFilter != "pt" {
		t.Fatalf("Expected typed filter pt after editing, got %+v", state.ui)
	}

	handleInput(tcell.NewEventKey(tcell.KeyRune, 's', tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyRight, 0, tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyUp, 0, tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyEnter, 0, tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyRune, 'p', tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyRune, 't', tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyRune, 'a', tcell.ModNone), state, app, pages, table, signaler)
	gotExpectedToggles := state.ui.ProcessSort == SortByMemory &&
		state.ui.ReverseSort &&
		!state.ui.SortMode &&
		state.ui.Paused &&
		state.ui.TreeMode &&
		state.ui.HideASCIIArt
	if !gotExpectedToggles {
		t.Fatalf("Expected sort/reverse/pause/ascii toggles, got %+v", state.ui)
	}

	handleInput(tcell.NewEventKey(tcell.KeyRune, '/', tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyCtrlU, 0, tcell.ModNone), state, app, pages, table, signaler)
	handleInput(tcell.NewEventKey(tcell.KeyEsc, 0, tcell.ModNone), state, app, pages, table, signaler)
	if state.ui.ProcessFilter != "" {
		t.Fatalf("Expected Ctrl+U to clear filter, got %+v", state.ui)
	}
}

func TestSortModeFollowsTableColumnOrder(t *testing.T) {
	order := []ProcessSortColumn{
		SortByPID,
		SortByUser,
		SortByCPU,
		SortByMemory,
		SortByGPU,
		SortByGPUMemory,
		SortByCommand,
	}
	for i, column := range order {
		next := order[(i+1)%len(order)]
		if got := nextProcessSortColumn(column); got != next {
			t.Fatalf("next sort after %v = %v, want %v", column, got, next)
		}
		prev := order[(i+len(order)-1)%len(order)]
		if got := previousProcessSortColumn(column); got != prev {
			t.Fatalf("previous sort before %v = %v, want %v", column, got, prev)
		}
	}
}

func TestModalInputRouting(t *testing.T) {
	app := tview.NewApplication()
	pages := tview.NewPages().AddPage("main", tview.NewBox(), true, true)
	showMessage(pages, "hello")

	handled, nextEvent := handleModalInput(tcell.NewEventKey(tcell.KeyEnter, 0, tcell.ModNone), pages, app)
	if !handled || nextEvent == nil || nextEvent.Key() != tcell.KeyEnter {
		t.Fatalf("Expected Enter to pass through to the modal, handled=%v next=%v", handled, nextEvent)
	}
	if !pages.HasPage("message") {
		t.Fatal("Expected Enter pass-through to leave modal open for tview to handle")
	}

	handled, nextEvent = handleModalInput(tcell.NewEventKey(tcell.KeyEsc, 0, tcell.ModNone), pages, app)
	if !handled || nextEvent != nil {
		t.Fatal("Expected Esc to close front modal")
	}
	if pages.HasPage("message") {
		t.Fatal("Expected message modal to be removed")
	}
	handled, nextEvent = handleModalInput(tcell.NewEventKey(tcell.KeyEsc, 0, tcell.ModNone), pages, app)
	if handled || nextEvent == nil {
		t.Fatal("Did not expect Esc to close main page")
	}
}

func TestHandleInputDoesNotReopenProcessDetailsWhenModalIsOpen(t *testing.T) {
	state := &State{
		dynamic: DynamicInfo{
			Processes: []ProcessInfo{{PID: 123, User: "root", Command: "sleep"}},
		},
	}
	app := tview.NewApplication()
	pages := tview.NewPages().AddPage("main", tview.NewBox(), true, true)
	table := tview.NewTable().SetSelectable(true, false)
	updateProcessTable(table, state)
	table.Select(1, 0)
	signaler := &MockProcessSignaler{}

	handleInput(tcell.NewEventKey(tcell.KeyEnter, 0, tcell.ModNone), state, app, pages, table, signaler)
	if !pages.HasPage("process-details") {
		t.Fatal("Expected first Enter to open process details")
	}

	nextEvent := handleInput(tcell.NewEventKey(tcell.KeyEnter, 0, tcell.ModNone), state, app, pages, table, signaler)
	if nextEvent == nil || nextEvent.Key() != tcell.KeyEnter {
		t.Fatalf("Expected second Enter to pass through to the modal, got %v", nextEvent)
	}
	if !pages.HasPage("process-details") {
		t.Fatal("Expected pass-through Enter to leave details modal available to tview")
	}
}

func TestSortModeInputRouting(t *testing.T) {
	state := &State{ui: UIState{ProcessSort: SortByUser, SortMode: true}}
	app := tview.NewApplication()
	pages := tview.NewPages()
	table := tview.NewTable()
	signaler := &MockProcessSignaler{}

	got := handleInput(tcell.NewEventKey(tcell.KeyRight, 0, tcell.ModNone), state, app, pages, table, signaler)
	if got != nil {
		t.Fatalf("Expected sort-mode Right key to be consumed, got %v", got)
	}
	if state.ui.ProcessSort != SortByCPU {
		t.Fatalf("Expected sort mode to follow table order from USER to CPU, got %v", state.ui.ProcessSort)
	}

	handleInput(tcell.NewEventKey(tcell.KeyLeft, 0, tcell.ModNone), state, app, pages, table, signaler)
	if state.ui.ProcessSort != SortByUser {
		t.Fatalf("Expected sort mode to move back from CPU to USER, got %v", state.ui.ProcessSort)
	}

	handleInput(tcell.NewEventKey(tcell.KeyDown, 0, tcell.ModNone), state, app, pages, table, signaler)
	if state.ui.ReverseSort {
		t.Fatal("Expected Down to select ascending sort direction")
	}
	handleInput(tcell.NewEventKey(tcell.KeyUp, 0, tcell.ModNone), state, app, pages, table, signaler)
	if !state.ui.ReverseSort {
		t.Fatal("Expected Up to select descending sort direction")
	}

	handleInput(tcell.NewEventKey(tcell.KeyEsc, 0, tcell.ModNone), state, app, pages, table, signaler)
	if state.ui.SortMode {
		t.Fatal("Expected Esc to leave sort mode")
	}
}

func TestTUIInputCaptureDoesNotDeadlock(t *testing.T) {
	state := &State{
		ui: UIState{ProcessSort: SortByCPU},
		dynamic: DynamicInfo{
			Processes: []ProcessInfo{
				{PID: 123, User: "root", CPUPercent: 12.3, MemPercent: 4.5, Command: "sleep"},
			},
		},
	}
	app := tview.NewApplication()
	screen := tcell.NewSimulationScreen("UTF-8")
	screen.SetSize(100, 32)
	table := tview.NewTable().
		SetSelectable(true, false).
		SetFixed(1, 0)
	table.SetBorder(true).SetTitle(" Processes ")
	mainLayout := tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(table, 0, 1, true)
	pages := tview.NewPages().
		AddPage("main", mainLayout, true, true)
	render := func() {
		updateProcessTable(table, state)
	}
	render()
	table.Select(1, 0)
	setAppInputCapture(app, state, pages, table, &MockProcessSignaler{}, render)
	app.SetScreen(screen).SetRoot(pages, true)

	done := make(chan error, 1)
	go func() {
		done <- app.Run()
	}()
	defer func() {
		go app.Stop()
		select {
		case <-done:
		case <-time.After(time.Second):
		}
	}()

	waitForTUICondition(t, app, func() bool {
		return table.HasFocus()
	}, "process table focus")

	screen.InjectKey(tcell.KeyEnter, 0, tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return pages.HasPage("process-details")
	}, "process details modal to open")

	screen.InjectKey(tcell.KeyEnter, 0, tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return !pages.HasPage("process-details")
	}, "process details modal to close")

	screen.InjectKey(tcell.KeyRune, 's', tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return state.ui.SortMode
	}, "sort mode to start")
	screen.InjectKey(tcell.KeyRight, 0, tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return state.ui.ProcessSort == SortByMemory
	}, "sort mode right arrow")
	screen.InjectKey(tcell.KeyEsc, 0, tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return !state.ui.SortMode
	}, "sort mode to exit")

	screen.InjectKey(tcell.KeyRune, '/', tcell.ModNone)
	screen.InjectKey(tcell.KeyRune, 'p', tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return state.ui.SearchMode && state.ui.ProcessFilter == "p"
	}, "filter mode typed input")
	screen.InjectKey(tcell.KeyEsc, 0, tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return !state.ui.SearchMode
	}, "filter mode to exit")

	screen.InjectKey(tcell.KeyCtrlC, 0, tcell.ModNone)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("TUI run returned error: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("TUI did not stop after Ctrl+C")
	}
}

func waitForTUICondition(t *testing.T, app *tview.Application, condition func() bool, description string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		var ok bool
		if !queueTUIUpdate(t, app, func() {
			ok = condition()
		}) {
			t.Fatalf("TUI event loop did not process update while waiting for %s", description)
		}
		if ok {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("Timed out waiting for %s", description)
}

func queueTUIUpdate(t *testing.T, app *tview.Application, update func()) bool {
	t.Helper()
	done := make(chan struct{})
	go func() {
		app.QueueUpdate(update)
		close(done)
	}()
	select {
	case <-done:
		return true
	case <-time.After(250 * time.Millisecond):
		return false
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

func TestGetGPUProcessMapEmptyOutput(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi pmon -c 1 -s um": "",
		},
	}

	processMap := getGPUProcessMap(mockRunner)
	if len(processMap) != 0 {
		t.Fatalf("Expected empty GPU process map, got %+v", processMap)
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

func TestGetStaticGPUInfoEmptyOutput(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits": "\n",
		},
	}

	count, totals := getStaticGPUInfo(mockRunner)
	if count != 0 || totals != nil {
		t.Fatalf("Expected no GPUs for empty output, got count=%d totals=%v", count, totals)
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

func TestUpdateLiveGPUUsageEmptyOutput(t *testing.T) {
	mockRunner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits": "\n",
		},
	}

	usage := updateLiveGPUUsage(2, mockRunner)
	if len(usage) != 0 {
		t.Fatalf("Expected empty live GPU usage, got %+v", usage)
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

func TestHostMetricsProviders(t *testing.T) {
	provider := MockHostMetricsProvider{
		cores:      8,
		cpuPercent: []float64{42.5},
		memory:     &mem.VirtualMemoryStat{Total: 16 * bytesPerGB, Used: 4 * bytesPerGB},
	}

	if got := updateHostCPUUsageWithProvider(provider); got != 42.5 {
		t.Fatalf("Expected host CPU 42.5, got %.1f", got)
	}
	if got := updateHostMemUsageWithProvider(provider); got != 4 {
		t.Fatalf("Expected host memory 4GB, got %.1f", got)
	}

	errProvider := MockHostMetricsProvider{err: errors.New("host failure")}
	if got := updateHostCPUUsageWithProvider(errProvider); got != 0 {
		t.Fatalf("Expected host CPU fallback 0, got %.1f", got)
	}
	if got := updateHostMemUsageWithProvider(errProvider); got != 0 {
		t.Fatalf("Expected host memory fallback 0, got %.1f", got)
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

func TestGetStaticInfoWithProviders(t *testing.T) {
	fs := MockFileReader{files: map[string]string{
		cgroupCPUMaxPath:    "200000 100000",
		cgroupMemoryMaxPath: "8589934592",
	}}
	runner := MockCommandRunner{err: errors.New("no gpu")}
	stater := MockStater{}
	hostProvider := MockHostMetricsProvider{
		cores:  12,
		memory: &mem.VirtualMemoryStat{Total: 32 * bytesPerGB},
	}
	storageProvider := MockStorageProvider{
		partitions: []disk.PartitionStat{
			{Mountpoint: "/", Fstype: "ext4"},
			{Mountpoint: "/proc", Fstype: "proc"},
		},
		usages: map[string]*disk.UsageStat{
			"/": {Total: 100 * bytesPerGB},
		},
	}

	info, err := getStaticInfoWithProviders(fs, runner, stater, hostProvider, storageProvider)
	if err != nil {
		t.Fatalf("Unexpected static info error: %v", err)
	}
	if info.HostCores != 12 || info.ContainerCPULimit != 2 || info.ContainerMemLimitGB != 8 {
		t.Fatalf("Unexpected static CPU/memory info: %+v", info)
	}
	if info.HostMemTotalGB != 32 {
		t.Fatalf("Expected host memory total 32GB, got %.1f", info.HostMemTotalGB)
	}
	if len(info.StorageMounts) != 1 || info.StorageMounts[0].Path != "/" {
		t.Fatalf("Expected only root storage mount, got %+v", info.StorageMounts)
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
			CgroupVersion:          CgroupV2,
			ContainerCPULimit:      2.0,
			ContainerMemLimitBytes: 2 * bytesPerGB,
			GPUCount:               1,
		},
		prevCPUUsage: 1000000,
		prevCPUTime:  time.Now().Add(-1 * time.Second),
	}

	// Run the function under test.
	updateAll(state, mockFS, mockRunner)

	tolerance := 0.1

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

func TestNewDynamicCollectorSetSelectsRequiredCollectors(t *testing.T) {
	state := &State{
		static: StaticInfo{
			HostCores:              8,
			ContainerCPULimit:      2,
			ContainerMemLimitBytes: bytesPerGB,
			GPUCount:               1,
		},
	}

	collectors := newDynamicCollectorSet(
		state,
		MockFileReader{},
		MockCommandRunner{},
		state.static,
		MockHostMetricsProvider{},
		MockStorageProvider{},
		MockProcessProvider{},
		MockNetworkProvider{},
		MockDiskIOProvider{},
	)
	if collectors.ContainerCPU == nil || collectors.ContainerMem == nil || collectors.LiveGPU == nil {
		t.Fatal("Expected limited container with GPU to collect container CPU, container memory, and live GPU")
	}
	if collectors.HostCPU != nil || collectors.HostMem != nil {
		t.Fatal("Did not expect host collectors when cgroup CPU and memory limits are active")
	}

	state.static.ContainerCPULimit = float64(state.static.HostCores)
	state.static.ContainerMemLimitBytes = 0
	state.static.GPUCount = 0
	collectors = newDynamicCollectorSet(
		state,
		MockFileReader{},
		MockCommandRunner{},
		state.static,
		MockHostMetricsProvider{},
		MockStorageProvider{},
		MockProcessProvider{},
		MockNetworkProvider{},
		MockDiskIOProvider{},
	)
	if collectors.ContainerCPU != nil || collectors.ContainerMem != nil || collectors.LiveGPU != nil {
		t.Fatal("Did not expect container or GPU collectors when host resources are displayed and no GPU is present")
	}
	if collectors.HostCPU == nil || collectors.HostMem == nil {
		t.Fatal("Expected host collectors when no cgroup limits are active")
	}
}

func TestCollectDynamicInfoWithTiming(t *testing.T) {
	collectors := DynamicCollectorSet{
		ContainerCPU: func() float64 { return 10 },
		ContainerMem: func() float64 { return 1.5 },
		LiveGPU:      func() []GPUUsage { return []GPUUsage{{Index: 0, Utilization: 50, MemUsedGB: 2}} },
		Storage:      func() []StorageUsage { return []StorageUsage{{Path: "/", UsedPercent: 25}} },
		Processes:    func() []ProcessInfo { return []ProcessInfo{{PID: 123, Command: "topic"}} },
		HostCPU:      func() float64 { return 20 },
		HostMem:      func() float64 { return 3.5 },
	}

	dynamic, timings := collectDynamicInfoWithTiming(collectors, time.Now)
	if dynamic.ContainerCPUUsage != 10 ||
		dynamic.ContainerMemUsedGB != 1.5 ||
		dynamic.HostCPUUsage != 20 ||
		dynamic.HostMemUsedGB != 3.5 {
		t.Fatalf("Unexpected collected scalar data: %+v", dynamic)
	}
	if len(dynamic.LiveGPUUsage) != 1 || len(dynamic.StorageUsage) != 1 || len(dynamic.Processes) != 1 {
		t.Fatalf("Unexpected collected slice data: %+v", dynamic)
	}
	expectedTimingKeys := []string{
		"container_cpu", "container_mem", "live_gpu", "storage", "processes", "host_cpu", "host_mem",
	}
	for _, key := range expectedTimingKeys {
		if _, ok := timings[key]; !ok {
			t.Fatalf("Expected timing key %q in %v", key, timings)
		}
	}
}

func TestCollectDynamicInfoWithTimingSkipsNilCollectors(t *testing.T) {
	dynamic, timings := collectDynamicInfoWithTiming(DynamicCollectorSet{
		HostCPU: func() float64 { return 42 },
	}, time.Now)

	if dynamic.HostCPUUsage != 42 {
		t.Fatalf("Expected host CPU usage from collector, got %.1f", dynamic.HostCPUUsage)
	}
	if len(timings) != 1 {
		t.Fatalf("Expected one timing entry, got %v", timings)
	}
	if _, ok := timings["host_cpu"]; !ok {
		t.Fatalf("Expected host_cpu timing entry, got %v", timings)
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

func TestStorageProviderPaths(t *testing.T) {
	provider := MockStorageProvider{
		partitions: []disk.PartitionStat{
			{Mountpoint: "/", Fstype: "ext4"},
			{Mountpoint: "/data", Fstype: "xfs"},
			{Mountpoint: "/tmp", Fstype: "tmpfs"},
			{Mountpoint: "/missing", Fstype: "ext4"},
		},
		usages: map[string]*disk.UsageStat{
			"/":     {Total: 100 * bytesPerGB, Used: 25 * bytesPerGB, Free: 75 * bytesPerGB, UsedPercent: 25},
			"/data": {Total: 200 * bytesPerGB, Used: 80 * bytesPerGB, Free: 120 * bytesPerGB, UsedPercent: 40},
		},
	}

	mounts := getStaticStorageInfoWithProvider(provider)
	if len(mounts) != 2 {
		t.Fatalf("Expected 2 static storage mounts, got %+v", mounts)
	}

	usage := updateStorageUsageWithProvider(mounts, provider)
	if len(usage) != 2 {
		t.Fatalf("Expected 2 storage usage rows, got %+v", usage)
	}
	if usage[0].Path != "/" || usage[0].UsedGB != 25 || usage[1].Path != "/data" || usage[1].UsedPercent != 40 {
		t.Fatalf("Unexpected storage usage: %+v", usage)
	}

	if got := getStaticStorageInfoWithProvider(MockStorageProvider{err: errors.New("partition failure")}); got != nil {
		t.Fatalf("Expected nil mounts on partition error, got %+v", got)
	}
	if got := updateStorageUsageWithProvider(nil, provider); got != nil {
		t.Fatalf("Expected nil usage for no mounts, got %+v", got)
	}
}

func TestNetworkAndDiskIORates(t *testing.T) {
	state := &State{
		prevNetwork: map[string]NetworkCounter{
			"eth0": {RXBytes: 1000, TXBytes: 2000},
		},
		prevDiskIO: map[string]DiskIOCounter{
			"sda": {ReadBytes: 1000, WriteBytes: 2000, ReadCount: 10, WriteCount: 20},
		},
		prevNetTime:  time.Now().Add(-time.Second),
		prevDiskTime: time.Now().Add(-time.Second),
	}

	network := updateNetworkUsageWithProvider(state, MockNetworkProvider{
		counters: []netio.IOCountersStat{
			{Name: "lo", BytesRecv: 999, BytesSent: 999},
			{Name: "eth0", BytesRecv: 3000, BytesSent: 5000, Errin: 1, Errout: 2},
		},
	})
	if len(network) != 1 || network[0].Name != "eth0" || network[0].RXBytesPerSec <= 0 || network[0].TXErrors != 2 {
		t.Fatalf("Unexpected network usage: %+v", network)
	}

	diskUsage := updateDiskIOUsageWithProvider(state, MockDiskIOProvider{
		counters: map[string]disk.IOCountersStat{
			"sda": {ReadBytes: 5000, WriteBytes: 7000, ReadCount: 30, WriteCount: 60},
		},
	})
	if len(diskUsage) != 1 || diskUsage[0].Name != "sda" ||
		diskUsage[0].ReadBytesPerSec <= 0 || diskUsage[0].WriteOpsPerSec <= 0 {
		t.Fatalf("Unexpected disk I/O usage: %+v", diskUsage)
	}
}

func TestCgroupEventsPIDsAndPressure(t *testing.T) {
	fs := MockFileReader{files: map[string]string{
		cgroupMemoryEventsPath:   "high 2\noom 3\noom_kill 1\nbad nope\n",
		cgroupCPUStatPath:        "usage_usec 1\nnr_throttled 4\n",
		cgroupPIDsCurrentPath:    "8",
		cgroupPIDsMaxPath:        "10",
		cgroupCPUPressurePath:    "some avg10=12.50 avg60=1.00 avg300=0.00 total=1\nfull avg10=3.00 avg60=0.00 avg300=0.00 total=1\n",
		cgroupMemoryPressurePath: "some avg10=0.50 avg60=0.00 avg300=0.00 total=1\n",
		cgroupIOPressurePath:     "full avg10=2.00 avg60=0.00 avg300=0.00 total=1\n",
	}}

	events := updateCgroupEvents(CgroupV2, fs)
	if events.MemoryHigh != 2 || events.MemoryOOM != 3 || events.MemoryOOMKill != 1 || events.CPUThrottledPeriods != 4 {
		t.Fatalf("Unexpected cgroup events: %+v", events)
	}
	pids := updatePIDUsage(fs)
	if pids.Current != 8 || pids.Max != 10 || pids.MaxText != "10" {
		t.Fatalf("Unexpected PID usage: %+v", pids)
	}
	pressure := updatePressure(fs)
	if len(pressure) != 3 || pressure[0].Resource != "cpu" || pressure[0].SomeAvg10 != 12.5 {
		t.Fatalf("Unexpected pressure stats: %+v", pressure)
	}
}

func TestMetricsHistoryAlertsAndRender(t *testing.T) {
	staticInfo := StaticInfo{ContainerMemLimitBytes: 10 * bytesPerGB}
	dynamic := DynamicSnapshot{
		ContainerCPUUsage:  50,
		ContainerMemUsedGB: 9.6,
		LiveGPUUsage:       []GPUUsage{{Utilization: 70}},
		NetworkUsage: []NetworkUsage{
			{
				Name:          "eth0",
				RXBytesPerSec: bytesPerSecondToMiBSecond,
				TXBytesPerSec: bytesPerSecondToMiBSecond,
			},
		},
		DiskIOUsage:  []DiskIOUsage{{Name: "sda", ReadBytesPerSec: bytesPerSecondToMiBSecond}},
		CgroupEvents: CgroupEvents{MemoryOOMKill: 1, CPUThrottledPeriods: 2},
		PIDUsage:     PIDUsage{Current: 9, Max: 10, MaxText: "10"},
		Pressure:     []PressureStat{{Resource: "cpu", SomeAvg10: 12}},
	}
	alerts := evaluateAlerts(staticInfo, dynamic)
	if len(alerts) < 4 {
		t.Fatalf("Expected multiple alerts, got %+v", alerts)
	}

	var history MetricsHistory
	updateMetricsHistory(&history, staticInfo, dynamic)
	state := &State{
		static:  staticInfo,
		history: history,
		dynamic: DynamicInfo{
			NetworkUsage: dynamic.NetworkUsage,
			DiskIOUsage:  dynamic.DiskIOUsage,
			CgroupEvents: dynamic.CgroupEvents,
			PIDUsage:     dynamic.PIDUsage,
			Pressure:     dynamic.Pressure,
			Alerts:       alerts,
		},
	}
	text := buildMetricsSection(state, 120)
	for _, want := range []string{"ALERT", "PIDS", "NET eth0", "IO sda", "PSI", "HIST CPU"} {
		if !strings.Contains(text, want) {
			t.Fatalf("Expected metrics section to contain %q, got %q", want, text)
		}
	}
}

func TestSparklineFixedWidthAndTrim(t *testing.T) {
	var ring HistoryRing
	ring.Add(25)
	if got := sparkline(ring); len([]rune(got)) != historySize {
		t.Fatalf("Expected fixed history width %d, got %q", historySize, got)
	} else if strings.Contains(got, "▁") {
		t.Fatalf("Expected first history sample to seed the ring without leading zero ramp, got %q", got)
	}
	trimmed := sparklineForWidth(ring, len("HIST CPU ")+minSparklineWidth, len("HIST CPU "))
	if len([]rune(trimmed)) != minSparklineWidth {
		t.Fatalf("Expected minimum sparkline width %d, got %q", minSparklineWidth, trimmed)
	}
	if got := sparklineForWidth(ring, 12, len("HIST CPU ")); got != "" {
		t.Fatalf("Expected sparkline to be omitted when width is too narrow, got %q", got)
	}
}

func TestIntegrationParsingAndStatusText(t *testing.T) {
	containerID := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	fs := MockFileReader{files: map[string]string{
		procSelfCgroupPath: "0::/docker/" + containerID,
	}}
	if got := containerIDFromCgroup(fs); got != containerID {
		t.Fatalf("Expected container id %q, got %q", containerID, got)
	}

	podJSON := `{
		"metadata": {"name": "topic-pod", "namespace": "default", "labels": {"app": "topic"}},
		"spec": {"nodeName": "node-a", "containers": [{"image": "topic:latest"}]}
	}`
	metadata, err := parseKubernetesPod(strings.NewReader(podJSON))
	if err != nil {
		t.Fatalf("Expected pod metadata parse to succeed: %v", err)
	}
	if metadata.Pod != "topic-pod" || metadata.Namespace != "default" || metadata.Image != "topic:latest" {
		t.Fatalf("Unexpected pod metadata: %+v", metadata)
	}

	text := integrationStatusText([]IntegrationStatus{
		{Name: integrationDocker, Available: true},
		{Name: integrationKubernetes, Detail: integrationUnavailable},
	})
	if !strings.Contains(text, "docker=+") || !strings.Contains(text, "kubernetes=-") {
		t.Fatalf("Unexpected integration status text: %q", text)
	}
}

func TestDiscoverDockerMetadataWithHooks(t *testing.T) {
	containerID := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	fs := MockFileReader{files: map[string]string{
		procSelfCgroupPath: "0::/docker/" + containerID,
	}}

	probe := integrationProbe{
		stat: func(path string) (os.FileInfo, error) {
			if path == dockerSocketPath {
				return fakeFileInfo{}, nil
			}
			return nil, os.ErrNotExist
		},
		queryDocker: func(id string) (ContainerMetadata, error) {
			if id != containerID {
				t.Fatalf("Expected docker query for container %q, got %q", containerID, id)
			}
			return ContainerMetadata{Name: "topic", Image: "topic:latest"}, nil
		},
	}
	metadata, status := discoverDockerMetadataWithProbe(fs, AppConfig{}, probe)
	if !status.Available || status.Detail != integrationAvailable {
		t.Fatalf("Expected available Docker status, got %+v", status)
	}
	if metadata.Runtime != integrationDocker || metadata.Name != "topic" || metadata.Image != "topic:latest" {
		t.Fatalf("Unexpected Docker metadata: %+v", metadata)
	}

	_, status = discoverDockerMetadataWithProbe(fs, AppConfig{DisableDocker: true}, probe)
	if status.Available || status.Detail != integrationDisabled {
		t.Fatalf("Expected disabled Docker status, got %+v", status)
	}

	unavailableProbe := integrationProbe{
		stat: func(_ string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		},
	}
	_, status = discoverDockerMetadataWithProbe(fs, AppConfig{}, unavailableProbe)
	if status.Available || status.Detail != integrationUnavailable {
		t.Fatalf("Expected unavailable Docker status, got %+v", status)
	}
}

func TestDiscoverDockerMetadataQueryError(t *testing.T) {
	fs := MockFileReader{files: map[string]string{
		procSelfCgroupPath: "0::/docker/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	}}
	probe := integrationProbe{
		stat: func(path string) (os.FileInfo, error) {
			if path == dockerSocketPath {
				return fakeFileInfo{}, nil
			}
			return nil, os.ErrNotExist
		},
		queryDocker: func(_ string) (ContainerMetadata, error) {
			return ContainerMetadata{}, errors.New("docker exploded")
		},
	}
	_, status := discoverDockerMetadataWithProbe(fs, AppConfig{}, probe)
	if status.Available || !strings.Contains(status.Detail, "docker exploded") {
		t.Fatalf("Expected Docker query error status, got %+v", status)
	}
}

func TestDiscoverKubernetesMetadataWithHooks(t *testing.T) {
	env := map[string]string{
		kubernetesServiceHostEnv: "10.0.0.1",
		kubernetesPodNameEnv:     "topic-pod",
		kubernetesNodeNameEnv:    "node-a",
	}
	fs := MockFileReader{files: map[string]string{
		kubernetesNamespacePath: "default\n",
		kubernetesTokenPath:     "token\n",
	}}
	probe := integrationProbe{
		getenv: func(key string) string {
			return env[key]
		},
		queryKubernetes: func(
			_ FileReader,
			host string,
			namespace string,
			pod string,
			token string,
		) (ContainerMetadata, error) {
			if host != "10.0.0.1" || namespace != "default" || pod != "topic-pod" || token != "token" {
				t.Fatalf(
					"Unexpected Kubernetes query args host=%q namespace=%q pod=%q token=%q",
					host,
					namespace,
					pod,
					token,
				)
			}
			return ContainerMetadata{
				Name:   "topic-pod",
				Image:  "topic:latest",
				Labels: map[string]string{"app": "topic"},
			}, nil
		},
	}
	metadata, status := discoverKubernetesMetadataWithProbe(fs, AppConfig{}, probe)
	if !status.Available || status.Detail != integrationAvailable {
		t.Fatalf("Expected available Kubernetes status, got %+v", status)
	}
	if metadata.Runtime != integrationKubernetes ||
		metadata.Namespace != "default" ||
		metadata.Pod != "topic-pod" ||
		metadata.Node != "node-a" ||
		metadata.Image != "topic:latest" ||
		metadata.Labels["app"] != "topic" {
		t.Fatalf("Unexpected Kubernetes metadata: %+v", metadata)
	}

	_, status = discoverKubernetesMetadataWithProbe(fs, AppConfig{DisableKubernetes: true}, probe)
	if status.Available || status.Detail != integrationDisabled {
		t.Fatalf("Expected disabled Kubernetes status, got %+v", status)
	}
}

func TestDiscoverKubernetesMetadataUnavailableAndPartialFiles(t *testing.T) {
	unavailableProbe := integrationProbe{
		getenv: func(_ string) string {
			return ""
		},
	}
	_, status := discoverKubernetesMetadataWithProbe(MockFileReader{}, AppConfig{}, unavailableProbe)
	if status.Available || status.Detail != integrationUnavailable {
		t.Fatalf("Expected unavailable Kubernetes status, got %+v", status)
	}

	partialProbe := integrationProbe{
		getenv: func(key string) string {
			switch key {
			case kubernetesServiceHostEnv:
				return "10.0.0.1"
			case kubernetesPodNameEnv:
				return "topic-pod"
			default:
				return ""
			}
		},
	}
	_, status = discoverKubernetesMetadataWithProbe(MockFileReader{}, AppConfig{}, partialProbe)
	if status.Available || status.Detail != "namespace not available" {
		t.Fatalf("Expected missing namespace status, got %+v", status)
	}

	fs := MockFileReader{files: map[string]string{kubernetesNamespacePath: "default"}}
	_, status = discoverKubernetesMetadataWithProbe(fs, AppConfig{}, partialProbe)
	if status.Available || status.Detail != "token not available" {
		t.Fatalf("Expected missing token status, got %+v", status)
	}
}

func TestDiscoverNVMLStatusWithHooks(t *testing.T) {
	disabled := discoverNVMLStatusWithProbe(AppConfig{DisableNVML: true}, integrationProbe{})
	if disabled.Available || disabled.Detail != integrationDisabled {
		t.Fatalf("Expected disabled NVML status, got %+v", disabled)
	}

	driverProbe := integrationProbe{
		stat: func(path string) (os.FileInfo, error) {
			if path == nvmlDriverVersionPath {
				return fakeFileInfo{}, nil
			}
			return nil, os.ErrNotExist
		},
	}
	driverStatus := discoverNVMLStatusWithProbe(AppConfig{}, driverProbe)
	if !driverStatus.Available || !strings.Contains(driverStatus.Detail, "nvidia driver") {
		t.Fatalf("Expected NVML driver status, got %+v", driverStatus)
	}

	libraryProbe := integrationProbe{
		stat: func(path string) (os.FileInfo, error) {
			if path == nvmlLibraryPath {
				return fakeFileInfo{}, nil
			}
			return nil, os.ErrNotExist
		},
	}
	libraryStatus := discoverNVMLStatusWithProbe(AppConfig{}, libraryProbe)
	if !libraryStatus.Available || !strings.Contains(libraryStatus.Detail, "NVML library") {
		t.Fatalf("Expected NVML library status, got %+v", libraryStatus)
	}

	unavailableProbe := integrationProbe{
		stat: func(_ string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		},
	}
	unavailableStatus := discoverNVMLStatusWithProbe(AppConfig{}, unavailableProbe)
	if unavailableStatus.Available || unavailableStatus.Detail != integrationUnavailable {
		t.Fatalf("Expected unavailable NVML status, got %+v", unavailableStatus)
	}
}

func TestDiscoverIntegrationsMergesFakeProviders(t *testing.T) {
	containerID := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	fs := MockFileReader{files: map[string]string{
		procSelfCgroupPath:      "0::/docker/" + containerID,
		kubernetesNamespacePath: "default",
		kubernetesTokenPath:     "token",
	}}
	probe := integrationProbe{
		stat: func(path string) (os.FileInfo, error) {
			switch path {
			case dockerSocketPath, nvmlDriverVersionPath:
				return fakeFileInfo{}, nil
			default:
				return nil, os.ErrNotExist
			}
		},
		getenv: func(key string) string {
			switch key {
			case kubernetesServiceHostEnv:
				return "10.0.0.1"
			case kubernetesPodNameEnv:
				return "topic-pod"
			default:
				return ""
			}
		},
		queryDocker: func(_ string) (ContainerMetadata, error) {
			return ContainerMetadata{Name: "docker-name", Image: "docker-image"}, nil
		},
		queryKubernetes: func(_ FileReader, _ string, _ string, _ string, _ string) (ContainerMetadata, error) {
			return ContainerMetadata{Pod: "topic-pod", Namespace: "default", Image: "kube-image"}, nil
		},
	}
	metadata, statuses := discoverIntegrationsWithProbe(fs, AppConfig{}, probe)
	if len(statuses) != integrationStatusCount {
		t.Fatalf("Expected %d integration statuses, got %+v", integrationStatusCount, statuses)
	}
	if metadata.Runtime != integrationKubernetes ||
		metadata.Name != "docker-name" ||
		metadata.Pod != "topic-pod" ||
		metadata.Image != "kube-image" {
		t.Fatalf("Expected merged integration metadata, got %+v", metadata)
	}
}

func TestLiveIntegrationDiscoveryOptIn(t *testing.T) {
	if os.Getenv("TOPIC_LIVE_INTEGRATION_TESTS") != "1" {
		t.Skip("set TOPIC_LIVE_INTEGRATION_TESTS=1 to run live Docker/Kubernetes/NVML discovery")
	}
	metadata, statuses := discoverIntegrations(OSFileReader{}, AppConfig{})
	if len(statuses) != integrationStatusCount {
		t.Fatalf("Expected %d integration statuses, got %+v", integrationStatusCount, statuses)
	}
	for _, status := range statuses {
		if status.Name == "" {
			t.Fatalf("Expected named integration status, got %+v; metadata=%+v", statuses, metadata)
		}
	}
}

func TestWriteJSONSnapshot(t *testing.T) {
	state := resourceBenchmarkState()
	state.static.Metadata = ContainerMetadata{Name: "topic", Runtime: integrationDocker}
	state.static.Integrations = []IntegrationStatus{{Name: integrationDocker, Available: true}}
	state.dynamic.Alerts = []Alert{{Level: alertWarning, Message: "test"}}

	var buffer bytes.Buffer
	if err := writeJSONSnapshot(&buffer, state); err != nil {
		t.Fatalf("Expected JSON snapshot to write: %v", err)
	}
	output := buffer.String()
	for _, want := range []string{`"timestamp"`, `"dynamic"`, `"metadata"`, `"integrations"`, `"test"`} {
		if !strings.Contains(output, want) {
			t.Fatalf("Expected JSON output to contain %q, got %q", want, output)
		}
	}
}
