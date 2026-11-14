package main

import (
	"testing"
	"time"
)

const testProcessPID = 1001

func refreshProcesses(
	tracker *processCPUTracker,
	static *StaticInfo,
	now time.Time,
	handles ...ProcessHandle,
) map[int32]ProcessInfo {
	provider := MockProcessProvider{processes: handles}
	processes := updateProcessListWithProvider(static, &CountingCommandRunner{}, provider, tracker, now)
	byPID := make(map[int32]ProcessInfo, len(processes))
	for _, process := range processes {
		byPID[process.PID] = process
	}
	return byPID
}

func TestProcessCPUUsesDeltaBetweenRefreshes(t *testing.T) {
	static := &StaticInfo{ContainerCPULimit: 2}
	tracker := &processCPUTracker{}
	start := time.Now()
	handle := MockProcessHandle{pid: testProcessPID, cpuSeconds: 500, createTime: 7, name: "worker"}

	first := refreshProcesses(tracker, static, start, handle)
	if got := first[testProcessPID].CPUPercent; got != 0 {
		t.Fatalf("first refresh CPUPercent = %v, want 0 without a delta", got)
	}

	// One CPU-second over two seconds is half a core, or a quarter of a 2-CPU limit.
	handle.cpuSeconds = 501
	second := refreshProcesses(tracker, static, start.Add(2*time.Second), handle)
	if got := second[testProcessPID].rawCPU; got != 50 {
		t.Errorf("rawCPU = %v, want 50", got)
	}
	if got := second[testProcessPID].CPUPercent; got != 25 {
		t.Errorf("CPUPercent = %v, want 25", got)
	}

	// An idle process drops to 0 even with a large lifetime total.
	third := refreshProcesses(tracker, static, start.Add(3*time.Second), handle)
	if got := third[testProcessPID].CPUPercent; got != 0 {
		t.Errorf("idle process CPUPercent = %v, want 0", got)
	}
}

func TestProcessCPUDoesNotInheritAReusedPID(t *testing.T) {
	static := &StaticInfo{ContainerCPULimit: 1}
	tracker := &processCPUTracker{}
	start := time.Now()

	refreshProcesses(tracker, static, start, MockProcessHandle{
		pid: testProcessPID, cpuSeconds: 10, createTime: 7, name: "old",
	})
	reused := refreshProcesses(tracker, static, start.Add(time.Second), MockProcessHandle{
		pid: testProcessPID, cpuSeconds: 20, createTime: 8, name: "new",
	})
	if got := reused[testProcessPID].CPUPercent; got != 0 {
		t.Errorf("reused PID CPUPercent = %v, want 0", got)
	}
}

func TestProcessCPUForgetsExitedProcesses(t *testing.T) {
	static := &StaticInfo{ContainerCPULimit: 1}
	tracker := &processCPUTracker{}
	start := time.Now()

	refreshProcesses(tracker, static, start, MockProcessHandle{pid: testProcessPID, cpuSeconds: 1, name: "worker"})
	refreshProcesses(tracker, static, start.Add(time.Second))
	if len(tracker.samples) != 0 {
		t.Errorf("tracker kept %d samples for exited processes", len(tracker.samples))
	}
}

func TestProcessCPUSkipsProcessWithoutTimes(t *testing.T) {
	static := &StaticInfo{ContainerCPULimit: 1}
	processes := refreshProcesses(&processCPUTracker{}, static, time.Now(), MockProcessHandle{
		pid: testProcessPID, timesMissing: true, name: "worker",
	})
	if len(processes) != 0 {
		t.Errorf("processes = %+v, want the unreadable process skipped", processes)
	}
}

func TestProcessListBreaksCPUTiesByPID(t *testing.T) {
	static := &StaticInfo{ContainerCPULimit: 1}
	provider := MockProcessProvider{processes: []ProcessHandle{
		MockProcessHandle{pid: 3, name: "c"},
		MockProcessHandle{pid: 1, name: "a"},
		MockProcessHandle{pid: 2, name: "b"},
	}}
	tracker := &processCPUTracker{}
	processes := updateProcessListWithProvider(static, &CountingCommandRunner{}, provider, tracker, time.Now())
	for index, want := range []int32{1, 2, 3} {
		if processes[index].PID != want {
			t.Fatalf("processes[%d].PID = %d, want %d", index, processes[index].PID, want)
		}
	}
}

func TestComputeContainerPercentsMemoryBase(t *testing.T) {
	const eightGiB = 8 * bytesPerGB
	tests := []struct {
		name   string
		static StaticInfo
		want   float64
	}{
		{"container limit", StaticInfo{ContainerMemLimitBytes: 4 * eightGiB, HostMemTotalGB: 128}, 25},
		{"no limit uses host memory", StaticInfo{HostMemTotalGB: 16}, 50},
		{"no limit or host total", StaticInfo{}, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, got := computeContainerPercents(0, eightGiB, &test.static)
			if got != test.want {
				t.Fatalf("memory percent = %v, want %v", got, test.want)
			}
		})
	}
}
