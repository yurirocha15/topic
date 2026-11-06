package main

import (
	"errors"
	"os"
	"testing"
)

// fakeNVML serves fixed per-GPU values. A GPU index listed in failing returns errors.
type fakeNVML struct {
	count       int
	countErr    error
	totals      []uint64
	used        []uint64
	utilization []int
	running     [][]nvmlProcessMemory
	samples     [][]nvmlProcessSample
	failing     map[int]bool

	sinceCalls [][]uint64
	shutdowns  int
}

var errFakeNVML = errors.New("fake NVML failure")

func (f *fakeNVML) DeviceCount() (int, error) { return f.count, f.countErr }

func (f *fakeNVML) MemoryInfo(index int) (uint64, uint64, error) {
	if f.failing[index] {
		return 0, 0, errFakeNVML
	}
	return f.totals[index], f.used[index], nil
}

func (f *fakeNVML) Utilization(index int) (int, error) {
	if f.failing[index] {
		return 0, errFakeNVML
	}
	return f.utilization[index], nil
}

func (f *fakeNVML) ProcessUtilization(index int, since uint64) ([]nvmlProcessSample, error) {
	if f.sinceCalls == nil {
		f.sinceCalls = make([][]uint64, f.count)
	}
	f.sinceCalls[index] = append(f.sinceCalls[index], since)
	if f.failing[index] {
		return nil, errFakeNVML
	}
	return f.samples[index], nil
}

func (f *fakeNVML) RunningProcesses(index int) ([]nvmlProcessMemory, error) {
	if f.failing[index] {
		return nil, errFakeNVML
	}
	return f.running[index], nil
}

func (f *fakeNVML) Shutdown() { f.shutdowns++ }

func newTestNVMLSource(t *testing.T, lib *fakeNVML) *nvmlGPUSource {
	t.Helper()
	source, err := newNVMLGPUSource(lib)
	if err != nil {
		t.Fatal(err)
	}
	return source
}

func TestNVMLSourceNeedsAGPU(t *testing.T) {
	for _, lib := range []*fakeNVML{{count: 0}, {countErr: errFakeNVML}} {
		if _, err := newNVMLGPUSource(lib); err == nil {
			t.Fatalf("expected an error for %+v", lib)
		}
		if lib.shutdowns != 1 {
			t.Fatalf("expected NVML to be shut down after a failed open, got %d shutdowns", lib.shutdowns)
		}
	}
}

func TestNVMLStaticInfoKeepsOneTotalPerGPU(t *testing.T) {
	lib := &fakeNVML{
		count:   3,
		totals:  []uint64{8 << 30, 0, 16 << 30},
		used:    make([]uint64, 3),
		failing: map[int]bool{1: true},
	}
	count, totals, err := newTestNVMLSource(t, lib).StaticInfo()
	if err != nil || count != 3 || len(totals) != 3 || totals[0] != 8 || totals[1] != 0 || totals[2] != 16 {
		t.Fatalf("expected 3 GPUs with totals [8 0 16] GiB, got %d %v %v", count, totals, err)
	}
}

func TestNVMLUsageSkipsGPUsThatFailAndKeepsIndexes(t *testing.T) {
	lib := &fakeNVML{
		count:       3,
		totals:      []uint64{8 << 30, 8 << 30, 8 << 30},
		used:        []uint64{1 << 30, 2 << 30, 3 << 30},
		utilization: []int{10, 20, 30},
		failing:     map[int]bool{1: true},
	}
	usage := newTestNVMLSource(t, lib).Usage()
	want := []GPUUsage{{Index: 0, Utilization: 10, MemUsedGB: 1}, {Index: 2, Utilization: 30, MemUsedGB: 3}}
	if len(usage) != len(want) || usage[0] != want[0] || usage[1] != want[1] {
		t.Fatalf("expected %+v, got %+v", want, usage)
	}
}

func TestNVMLProcessesCombineMemoryAndAveragedUtilization(t *testing.T) {
	lib := &fakeNVML{
		count: 2,
		running: [][]nvmlProcessMemory{
			{{PID: 10, UsedBytes: 512 << 20}, {PID: 11, UsedBytes: 0}},
			{{PID: 20, UsedBytes: 2 << 30}},
		},
		samples: [][]nvmlProcessSample{
			{
				{PID: 10, SMUtil: 40, TimeStamp: 100},
				{PID: 10, SMUtil: 60, TimeStamp: 200},
				{PID: 12, SMUtil: 5, TimeStamp: 150},
			},
			nil,
		},
	}
	source := newTestNVMLSource(t, lib)
	processes := source.Processes()
	want := map[int32]GPUProcessInfo{
		10: {GPUIndex: 0, GPUUtil: 50, UsedMemMB: 512},
		11: {GPUIndex: 0},
		12: {GPUIndex: 0, GPUUtil: 5}, // sampled but not in the running list
		20: {GPUIndex: 1, UsedMemMB: 2048},
	}
	if len(processes) != len(want) {
		t.Fatalf("expected %+v, got %+v", want, processes)
	}
	for pid, info := range want {
		if processes[pid] != info {
			t.Errorf("PID %d: expected %+v, got %+v", pid, info, processes[pid])
		}
	}

	// The next call asks only for newer samples and ignores any NVML repeats.
	lib.samples[0] = []nvmlProcessSample{{PID: 10, SMUtil: 90, TimeStamp: 200}, {PID: 10, SMUtil: 20, TimeStamp: 300}}
	processes = source.Processes()
	if lib.sinceCalls[0][0] != 0 || lib.sinceCalls[0][1] != 200 || lib.sinceCalls[1][1] != 0 {
		t.Fatalf("expected the second call to ask for samples after 200 on GPU 0, got %v", lib.sinceCalls)
	}
	if processes[10].GPUUtil != 20 {
		t.Fatalf("expected only the new sample to count, got %+v", processes[10])
	}
}

func TestNVMLProcessesSkipAFailingGPU(t *testing.T) {
	lib := &fakeNVML{
		count:   2,
		running: [][]nvmlProcessMemory{nil, {{PID: 20, UsedBytes: 1 << 20}}},
		samples: [][]nvmlProcessSample{nil, nil},
		failing: map[int]bool{0: true},
	}
	processes := newTestNVMLSource(t, lib).Processes()
	if len(processes) != 1 || processes[20] != (GPUProcessInfo{GPUIndex: 1, UsedMemMB: 1}) {
		t.Fatalf("expected only GPU 1's process, got %+v", processes)
	}
}

func TestCollectorsUseNVMLInsteadOfNvidiaSMI(t *testing.T) {
	pid := os.Getpid()
	lib := &fakeNVML{
		count:       1,
		totals:      []uint64{16 << 30},
		used:        []uint64{4 << 30},
		utilization: []int{75},
		running:     [][]nvmlProcessMemory{{{PID: uint32(pid), UsedBytes: 1 << 30}}},
		samples:     [][]nvmlProcessSample{{{PID: uint32(pid), SMUtil: 70, TimeStamp: 1}}},
	}
	source := newTestNVMLSource(t, lib)
	runner := &CountingCommandRunner{}
	state := &State{static: StaticInfo{GPUCount: 1, GPUTotalGB: []float64{16}}, nvml: source}
	collectors := newDynamicCollectorSet(
		state, MockFileReader{}, runner, state.static,
		MockHostMetricsProvider{}, MockStorageProvider{}, newProcfsReader(procRoot),
		MockNetworkProvider{}, MockDiskIOProvider{},
	)
	usage := collectors.LiveGPU()
	if len(usage) != 1 || usage[0].Utilization != 75 || usage[0].MemUsedGB != 4 {
		t.Fatalf("expected NVML usage, got %+v", usage)
	}
	var found bool
	for _, process := range collectors.Processes() {
		if process.PID == int32(pid) {
			found = true
			if process.GPUUtil != 70 || process.GPUMemMB != 1024 {
				t.Fatalf("expected NVML process usage, got %+v", process)
			}
		}
	}
	if !found {
		t.Fatalf("test process %d not found", pid)
	}
	if runner.calls != 0 {
		t.Fatalf("expected no nvidia-smi calls, got %d", runner.calls)
	}
}

func TestMarkNVMLInUse(t *testing.T) {
	statuses := []IntegrationStatus{
		{Name: integrationDocker, Detail: integrationUnavailable},
		{Name: integrationNVML, Detail: "nvidia driver visible; nvidia-smi fallback enabled"},
	}
	markNVMLInUse(statuses)
	if statuses[0].Available || statuses[0].Detail != integrationUnavailable {
		t.Fatalf("expected other integrations unchanged, got %+v", statuses[0])
	}
	if !statuses[1].Available || statuses[1].Detail != nvmlInUseDetail {
		t.Fatalf("expected NVML in use, got %+v", statuses[1])
	}
}
