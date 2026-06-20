package main

import (
	"testing"
	"time"

	"github.com/rivo/tview"
	"github.com/shirou/gopsutil/v3/disk"
	netio "github.com/shirou/gopsutil/v3/net"
)

func BenchmarkMakeBar(b *testing.B) {
	for range b.N {
		_ = makeBar(72.5, 80)
	}
}

func BenchmarkCalculateBarWidth(b *testing.B) {
	labels := []string{
		"CPU: [yellow]72.5  %[white] ",
		"MEM: [yellow]61.2  %[white] ",
		"DISK /data:    [yellow]48.8%[white]",
	}
	infos := []string{
		" [darkcyan](limit: 4.00 CPUs)[white]",
		" [darkcyan]3.672 GB / 6.000 GB[white]",
		"[darkcyan]48.80 GB / 100.00 GB[white]",
	}

	for range b.N {
		_ = calculateBarWidth(180, labels, infos)
	}
}

func BenchmarkBuildStorageSection(b *testing.B) {
	layout, labels, infos, percentages := storageSectionBenchmarkFixture()

	for range b.N {
		_ = buildStorageSection(layout, labels, infos, percentages)
	}
}

func BenchmarkBuildStorageSectionBars(b *testing.B) {
	layout, labels, infos, percentages := storageSectionBenchmarkFixture()
	bars := slicesToBars(labels, infos, percentages)

	for range b.N {
		_ = buildStorageSectionBars(layout, bars)
	}
}

func BenchmarkBuildResourceText(b *testing.B) {
	state := resourceBenchmarkState()

	for range b.N {
		_ = buildResourceText(180, state)
	}
}

func BenchmarkUpdateProcessTable(b *testing.B) {
	state := resourceBenchmarkState()
	table := tview.NewTable()

	for range b.N {
		updateProcessTable(table, state)
	}
}

func BenchmarkPrepareProcessRowsFilterSort(b *testing.B) {
	processes := benchmarkProcesses(500)
	ui := UIState{
		ProcessSort:   SortByMemory,
		ReverseSort:   true,
		ProcessFilter: "python",
	}

	for range b.N {
		_ = prepareProcessRows(append([]ProcessInfo(nil), processes...), ui)
	}
}

func BenchmarkBuildProcessTreeRows(b *testing.B) {
	processes := benchmarkProcesses(500)
	for i := range processes {
		if i > 0 {
			processes[i].ParentPID = processes[(i-1)/2].PID
		}
	}

	for range b.N {
		_ = buildProcessTreeRows(processes)
	}
}

func BenchmarkUpdateProcessListWithProvider(b *testing.B) {
	staticInfo := &StaticInfo{
		ContainerCPULimit:      4,
		ContainerMemLimitBytes: 8 * bytesPerGB,
	}
	provider := MockProcessProvider{processes: benchmarkProcessHandles(500)}
	runner := MockCommandRunner{}

	for range b.N {
		_ = updateProcessListWithProvider(staticInfo, runner, provider)
	}
}

func BenchmarkCollectDynamicInfoFake(b *testing.B) {
	collectors := DynamicCollectorSet{
		ContainerCPU: func() float64 { return 72.5 },
		ContainerMem: func() float64 { return 3.75 },
		LiveGPU: func() []GPUUsage {
			return []GPUUsage{
				{Index: 0, Utilization: 68, MemUsedGB: 12},
				{Index: 1, Utilization: 31, MemUsedGB: 4},
			}
		},
		Storage: func() []StorageUsage {
			return []StorageUsage{
				{Path: "/", UsedGB: 38.2, FreeGB: 61.8, UsedPercent: 38.2},
				{Path: "/data", UsedGB: 71.4, FreeGB: 28.6, UsedPercent: 71.4},
			}
		},
		Processes: func() []ProcessInfo { return benchmarkProcesses(100) },
		HostCPU:   func() float64 { return 19 },
		HostMem:   func() float64 { return 22 },
	}

	for range b.N {
		_ = collectDynamicInfo(collectors)
	}
}

func storageSectionBenchmarkFixture() (BarLayout, []string, []string, []float64) {
	labels := []string{
		"DISK /:        [yellow] 38.2%[white]",
		"DISK /data:    [yellow] 71.4%[white]",
		"DISK /logs:    [yellow] 12.9%[white]",
		"DISK /cache:   [yellow] 55.0%[white]",
	}
	infos := []string{
		"[darkcyan]38.20 GB / 100.00 GB[white]",
		"[darkcyan]71.40 GB / 100.00 GB[white]",
		"[darkcyan]12.90 GB / 100.00 GB[white]",
		"[darkcyan]55.00 GB / 100.00 GB[white]",
	}
	percentages := []float64{38.2, 71.4, 12.9, 55.0}
	maxLabelWidth, maxInfoWidth := calculateMaxWidthsFromSlices(labels, infos)
	layout := BarLayout{
		Columns:       2,
		BarWidth:      40,
		TotalWidth:    150,
		MaxLabelWidth: maxLabelWidth,
		MaxInfoWidth:  maxInfoWidth,
	}

	return layout, labels, infos, percentages
}

func resourceBenchmarkState() *State {
	return &State{
		static: StaticInfo{
			ContainerCPULimit:      4,
			ContainerMemLimitBytes: 8 * bytesPerGB,
			ContainerMemLimitGB:    8,
			HostCores:              16,
			HostMemTotalGB:         64,
			GPUCount:               2,
			GPUTotalGB:             []float64{24, 24},
		},
		dynamic: DynamicInfo{
			ContainerCPUUsage:  72.5,
			ContainerMemUsedGB: 3.75,
			HostCPUUsage:       19,
			HostMemUsedGB:      22,
			LiveGPUUsage: []GPUUsage{
				{Index: 0, Utilization: 68, MemUsedGB: 12},
				{Index: 1, Utilization: 31, MemUsedGB: 4},
			},
			StorageUsage: []StorageUsage{
				{Path: "/", UsedGB: 38.2, FreeGB: 61.8, UsedPercent: 38.2},
				{Path: "/data", UsedGB: 71.4, FreeGB: 28.6, UsedPercent: 71.4},
				{Path: "/logs", UsedGB: 12.9, FreeGB: 87.1, UsedPercent: 12.9},
				{Path: "/cache", UsedGB: 55, FreeGB: 45, UsedPercent: 55},
			},
			Processes: benchmarkProcesses(100),
		},
	}
}

func benchmarkProcesses(count int) []ProcessInfo {
	processes := make([]ProcessInfo, 0, count)
	for i := range count {
		processes = append(processes, ProcessInfo{
			PID:           int32(1000 + i),
			User:          "worker",
			CPUPercent:    float64(i % 100),
			MemPercent:    float64(i%50) / 2,
			Command:       "python train.py --batch-size 32 --worker",
			GPUIndex:      -1,
			GPUUtil:       uint64(i % 100),
			GPUMemPercent: float64(i % 80),
		})
		if i%3 == 0 {
			processes[i].GPUIndex = i % 2
		}
	}
	return processes
}

func benchmarkProcessHandles(count int) []ProcessHandle {
	processes := make([]ProcessHandle, 0, count)
	for i := range count {
		processes = append(processes, MockProcessHandle{
			pid:        int32(1000 + i),
			cpuPercent: float64((count - i) % 100),
			rss:        uint64((i%8)+1) * bytesPerGB / 2,
			user:       "worker",
			cmdline:    "python train.py --batch-size 32 --worker",
			name:       "python",
		})
	}
	return processes
}

func BenchmarkGetGPUProcessMap(b *testing.B) {
	runner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi pmon -c 1 -s um": `# gpu     pid  type    sm   mem   enc   dec   command
    0   20131     C    15     8     -     -   python
    0   20567     C     -     -     -     -   -
    1   34567     G    22    15     -     -   Xorg`,
		},
	}

	for range b.N {
		_ = getGPUProcessMap(runner)
	}
}

func BenchmarkUpdateLiveGPUUsage(b *testing.B) {
	runner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits": "50, 2048\n75, 4096\n",
		},
	}

	for range b.N {
		_ = updateLiveGPUUsage(2, runner)
	}
}

func BenchmarkUpdateContainerMemUsage(b *testing.B) {
	fs := MockFileReader{
		files: map[string]string{
			cgroupMemoryStatPath: "anon 1073741824\nfile 536870912\nkernel_stack 16384\n",
		},
	}

	for range b.N {
		_ = updateContainerMemUsage(CgroupV2, fs)
	}
}

func BenchmarkNetworkAndDiskCollectors(b *testing.B) {
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
	networkProvider := MockNetworkProvider{counters: []netio.IOCountersStat{
		{Name: "eth0", BytesRecv: 3000, BytesSent: 5000},
	}}
	diskProvider := MockDiskIOProvider{counters: map[string]disk.IOCountersStat{
		"sda": {ReadBytes: 5000, WriteBytes: 7000, ReadCount: 30, WriteCount: 60},
	}}

	for range b.N {
		_ = updateNetworkUsageWithProvider(state, networkProvider)
		_ = updateDiskIOUsageWithProvider(state, diskProvider)
	}
}

func BenchmarkHistorySparkline(b *testing.B) {
	var ring HistoryRing
	for i := range historySize {
		ring.Add(float64(i))
	}

	for range b.N {
		_ = sparkline(ring)
	}
}

func BenchmarkEvaluateAlerts(b *testing.B) {
	staticInfo := StaticInfo{ContainerMemLimitBytes: 10 * bytesPerGB}
	dynamic := DynamicSnapshot{
		ContainerMemUsedGB: 9.6,
		CgroupEvents:       CgroupEvents{MemoryOOMKill: 1, CPUThrottledPeriods: 2},
		PIDUsage:           PIDUsage{Current: 9, Max: 10},
		Pressure:           []PressureStat{{Resource: "cpu", SomeAvg10: 12}},
	}

	for range b.N {
		_ = evaluateAlerts(staticInfo, dynamic)
	}
}
