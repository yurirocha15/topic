//go:build nvml

package main

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

// TestLiveNVMLMatchesNvidiaSMI compares NVML's readings with nvidia-smi on a machine with an
// NVIDIA GPU.
func TestLiveNVMLMatchesNvidiaSMI(t *testing.T) {
	if os.Getenv("TOPIC_LIVE_INTEGRATION_TESTS") != "1" {
		t.Skip("set TOPIC_LIVE_INTEGRATION_TESTS=1 to compare NVML with nvidia-smi on a GPU machine")
	}
	runner := OSCommandRunner{Context: context.Background(), Timeout: 10 * time.Second}
	smiCount, smiTotals, err := getStaticGPUInfo(runner)
	if err != nil || smiCount == 0 {
		t.Skipf("nvidia-smi reports no GPU: %v", err)
	}
	source, err := openNVMLGPUSource()
	if err != nil {
		t.Fatalf("open NVML: %v", err)
	}
	defer source.Close()

	count, totals, _ := source.StaticInfo()
	if count != smiCount {
		t.Fatalf("NVML sees %d GPUs, nvidia-smi %d", count, smiCount)
	}
	for index := range totals {
		if math.Abs(totals[index]-smiTotals[index]) > 0.01 {
			t.Errorf("GPU %d total: NVML %.3f GiB, nvidia-smi %.3f GiB", index, totals[index], smiTotals[index])
		}
	}

	usage := source.Usage()
	smiUsage := updateLiveGPUUsage(smiCount, runner)
	if len(usage) != len(smiUsage) {
		t.Fatalf("NVML reports %d GPUs, nvidia-smi %d", len(usage), len(smiUsage))
	}
	for index := range usage {
		// Memory moves between the two reads, so allow 64 MiB.
		if math.Abs(usage[index].MemUsedGB-smiUsage[index].MemUsedGB) > 1.0/16 {
			t.Errorf("GPU %d used: NVML %.3f GiB, nvidia-smi %.3f GiB",
				index, usage[index].MemUsedGB, smiUsage[index].MemUsedGB)
		}
	}

	processes := source.Processes()
	for pid, smi := range getGPUProcessMap(runner) {
		info, ok := processes[pid]
		if !ok {
			t.Errorf("nvidia-smi lists PID %d on the GPU, NVML does not", pid)
			continue
		}
		if info.GPUIndex != smi.GPUIndex || info.UsedMemMB != smi.UsedMemMB {
			t.Errorf("PID %d: NVML %+v, nvidia-smi %+v", pid, info, smi)
		}
	}
}
