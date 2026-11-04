package main

import (
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Output captured from `nvidia-smi pmon -c 1 -s um` with driver 595.
const pmonDriver595Output = `# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0    1499476     G      -      -      -      -      -      -    267      0    orca --gpu-pref
    0    3521706     G     12     30      -      -      -      -   4096      0    Xorg
    1      20131     C     55      8      -      -      -      -      -      -    python train.py`

// Older drivers print fewer columns before fb.
const pmonOlderDriverOutput = `# gpu        pid  type    sm   mem   enc   dec    fb   command
# Idx          #   C/G     %     %     %     %    MB   name
    0      20131     C    15     8     -     -  2048   python`

func pmonRunner(output string) MockCommandRunner {
	return MockCommandRunner{outputs: map[string]string{"nvidia-smi pmon -c 1 -s um": output}}
}

func TestGetGPUProcessMapReadsFramebufferMemoryByHeader(t *testing.T) {
	tests := []struct {
		name   string
		output string
		want   map[int32]GPUProcessInfo
	}{
		{
			name:   "driver 595 columns",
			output: pmonDriver595Output,
			want: map[int32]GPUProcessInfo{
				1499476: {GPUIndex: 0, GPUUtil: 0, UsedMemMB: 267},
				3521706: {GPUIndex: 0, GPUUtil: 12, UsedMemMB: 4096},
				20131:   {GPUIndex: 1, GPUUtil: 55, UsedMemMB: 0},
			},
		},
		{
			name:   "older driver columns",
			output: pmonOlderDriverOutput,
			want:   map[int32]GPUProcessInfo{20131: {GPUIndex: 0, GPUUtil: 15, UsedMemMB: 2048}},
		},
		{
			name:   "no header keeps utilization and leaves memory unknown",
			output: "0 20131 C 15 8 - - python",
			want:   map[int32]GPUProcessInfo{20131: {GPUIndex: 0, GPUUtil: 15, UsedMemMB: 0}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getGPUProcessMap(pmonRunner(tt.output))
			if len(got) != len(tt.want) {
				t.Fatalf("expected %d processes, got %d: %+v", len(tt.want), len(got), got)
			}
			for pid, want := range tt.want {
				if got[pid] != want {
					t.Errorf("PID %d: expected %+v, got %+v", pid, want, got[pid])
				}
			}
		})
	}
}

func TestGetGPUProcessMapSkipsRowsMissingHeaderColumns(t *testing.T) {
	output := `# gpu pid type sm mem enc dec fb command
0 20131 C 15 8 - - nope python
0 20132 C 15 8
0 20133 C 15 8 - - 512 python`

	got := getGPUProcessMap(pmonRunner(output))
	if len(got) != 1 || got[20133].UsedMemMB != 512 {
		t.Fatalf("expected only PID 20133 with 512 MB, got %+v", got)
	}
}

func TestProcessGPUMemoryPercentUsesFramebufferShareOfGPU(t *testing.T) {
	pid := int32(os.Getpid())
	tests := []struct {
		name     string
		gpuIndex int
		usedMB   string
		totals   []float64
		want     float64
	}{
		{name: "share of the process's GPU", gpuIndex: 1, usedMB: "4096", totals: []float64{8, 16}, want: 25},
		{name: "unknown GPU total", gpuIndex: 0, usedMB: "4096", totals: []float64{0}, want: 0},
		{name: "GPU index without a total", gpuIndex: 2, usedMB: "4096", totals: []float64{16}, want: 0},
		{name: "memory not reported", gpuIndex: 0, usedMB: "-", totals: []float64{16}, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := "# gpu pid type sm mem enc dec fb command\n" +
				strings.Join(
					[]string{
						strconv.Itoa(tt.gpuIndex),
						strconv.Itoa(int(pid)),
						"C",
						"50",
						"90",
						"-",
						"-",
						tt.usedMB,
						"test",
					},
					" ",
				)
			static := &StaticInfo{GPUCount: len(tt.totals), GPUTotalGB: tt.totals}

			processes := updateProcessListWithProvider(
				static,
				pmonRunner(output),
				newProcfsReader(procRoot),
				&processCPUTracker{},
				time.Now(),
			)
			for _, process := range processes {
				if process.PID != pid {
					continue
				}
				if math.Abs(process.GPUMemPercent-tt.want) > 1e-9 {
					t.Fatalf("expected GPU memory %.1f%%, got %.1f%%", tt.want, process.GPUMemPercent)
				}
				return
			}
			t.Fatalf("test process %d not found", pid)
		})
	}
}

func TestProcessDetailsShowGPUMemoryUsed(t *testing.T) {
	details := processDetailsText(ProcessInfo{GPUIndex: 0, GPUUtil: 12, GPUMemPercent: 25, GPUMemMB: 4096})
	if !strings.Contains(details, "GPU: GPU0 util=12% mem=25.0% (4096 MB)") {
		t.Fatalf("expected GPU memory used in details, got:\n%s", details)
	}
}
