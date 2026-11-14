package main

import (
	"testing"

	"github.com/shirou/gopsutil/v4/mem"
)

// smtHostProvider reports a host with two hardware threads per core, such as a 12-core,
// 24-thread CPU.
type smtHostProvider struct {
	MockHostMetricsProvider
}

func (smtHostProvider) CPUCounts(logical bool) (int, error) {
	if logical {
		return 24, nil
	}
	return 12, nil
}

func staticInfoForCPUMax(t *testing.T, cpuMax string) StaticInfo {
	t.Helper()
	fs := MockFileReader{files: map[string]string{cgroupCPUMaxPath: cpuMax}}
	host := smtHostProvider{MockHostMetricsProvider{memory: &mem.VirtualMemoryStat{Total: 32 * bytesPerGB}}}
	info, err := getStaticInfoWithProviders(
		fs, func() (int, []float64, error) { return 0, nil, nil }, MockStater{}, host, MockStorageProvider{},
	)
	if err != nil {
		t.Fatalf("unexpected static info error: %v", err)
	}
	return info
}

// CPU limits and per-process CPU time are measured in logical CPUs, so the host must be
// counted the same way.
func TestStaticInfoCountsLogicalCPUs(t *testing.T) {
	info := staticInfoForCPUMax(t, "max 100000")
	if info.HostCores != 24 {
		t.Fatalf("expected 24 logical CPUs, got %d", info.HostCores)
	}
	if info.ContainerCPULimited || info.ContainerCPULimit != 24 {
		t.Fatalf("expected no CPU limit spanning 24 CPUs, got %+v", info)
	}
	if usesContainerCPU(info) {
		t.Fatal("expected host CPU usage without a CPU limit")
	}
}

// A limit that happens to equal the host's core count is still a limit.
func TestCPULimitEqualToHostCoresIsStillALimit(t *testing.T) {
	for _, cpuMax := range []string{"1200000 100000", "2400000 100000"} {
		info := staticInfoForCPUMax(t, cpuMax)
		if !info.ContainerCPULimited {
			t.Fatalf("cpu.max %q: expected a CPU limit, got %+v", cpuMax, info)
		}
		if !usesContainerCPU(info) {
			t.Fatalf("cpu.max %q: expected container CPU usage", cpuMax)
		}
	}
}

func TestCPULabelUsesLimitFlag(t *testing.T) {
	state := &State{
		static:  StaticInfo{HostCores: 8, ContainerCPULimit: 8, ContainerCPULimited: true},
		dynamic: DynamicInfo{ContainerCPUUsage: 75, HostCPUUsage: 25},
	}
	_, info, usage := calculateCPULabelInfo(state)
	if info != "8.00 CPU limit" || usage != 75 {
		t.Fatalf("expected the container's 8 CPU limit, got %q %.1f", info, usage)
	}

	state.static.ContainerCPULimited = false
	_, info, usage = calculateCPULabelInfo(state)
	if info != "8 CPUs, no limit" || usage != 25 {
		t.Fatalf("expected host CPUs without a limit, got %q %.1f", info, usage)
	}
}
