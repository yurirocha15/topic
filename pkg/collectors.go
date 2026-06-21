package main

import (
	"os"
	"sync"
	"time"
)

// --- Host System Functions ---

func updateHostCPUUsageWithProvider(provider HostMetricsProvider) float64 {
	if percentages, err := provider.CPUPercent(0, false); err == nil && len(percentages) > 0 {
		return percentages[0]
	}
	return 0
}

func updateHostMemUsageWithProvider(provider HostMetricsProvider) float64 {
	if memInfo, err := provider.VirtualMemory(); err == nil {
		return float64(memInfo.Used) / float64(bytesPerGB)
	}
	return 0
}

// --- Data Fetching Functions ---

func refreshDynamicState(state *State, fs FileReader, runner CommandRunner) {
	refreshDynamicStateWithProviders(
		state,
		fs,
		runner,
		OSHostMetricsProvider{},
		OSStorageProvider{},
		OSProcessProvider{},
		OSNetworkProvider{},
		OSDiskIOProvider{},
	)
}

func refreshDynamicStateWithProviders(
	state *State,
	fs FileReader,
	runner CommandRunner,
	hostProvider HostMetricsProvider,
	storageProvider StorageProvider,
	processProvider ProcessProvider,
	networkProvider NetworkProvider,
	diskIOProvider DiskIOProvider,
) {
	staticInfo := state.static
	dynamic := collectDynamicInfo(newDynamicCollectorSet(
		state,
		fs,
		runner,
		staticInfo,
		hostProvider,
		storageProvider,
		processProvider,
		networkProvider,
		diskIOProvider,
	))
	dynamic.Alerts = evaluateAlerts(staticInfo, dynamic)

	state.dynamic.mu.Lock()
	updateMetricsHistory(&state.history, staticInfo, dynamic)
	state.dynamic.ContainerCPUUsage = dynamic.ContainerCPUUsage
	state.dynamic.ContainerMemUsedGB = dynamic.ContainerMemUsedGB
	state.dynamic.LiveGPUUsage = dynamic.LiveGPUUsage
	state.dynamic.StorageUsage = dynamic.StorageUsage
	state.dynamic.NetworkUsage = dynamic.NetworkUsage
	state.dynamic.DiskIOUsage = dynamic.DiskIOUsage
	state.dynamic.CgroupEvents = dynamic.CgroupEvents
	state.dynamic.PIDUsage = dynamic.PIDUsage
	state.dynamic.Pressure = dynamic.Pressure
	state.dynamic.Alerts = dynamic.Alerts
	state.dynamic.Processes = dynamic.Processes
	state.dynamic.HostCPUUsage = dynamic.HostCPUUsage
	state.dynamic.HostMemUsedGB = dynamic.HostMemUsedGB
	state.dynamic.mu.Unlock()
}

func newDynamicCollectorSet(
	state *State,
	fs FileReader,
	runner CommandRunner,
	staticInfo StaticInfo,
	hostProvider HostMetricsProvider,
	storageProvider StorageProvider,
	processProvider ProcessProvider,
	networkProvider NetworkProvider,
	diskIOProvider DiskIOProvider,
) DynamicCollectorSet {
	collectors := DynamicCollectorSet{
		Storage: func() []StorageUsage {
			return updateStorageUsageWithProvider(staticInfo.StorageMounts, storageProvider)
		},
		Network: func() []NetworkUsage {
			return updateNetworkUsageWithProvider(state, networkProvider)
		},
		DiskIO: func() []DiskIOUsage {
			return updateDiskIOUsageWithProvider(state, diskIOProvider)
		},
		CgroupEvents: func() CgroupEvents {
			return updateCgroupEvents(staticInfo.CgroupVersion, fs)
		},
		PIDs: func() PIDUsage {
			return updatePIDUsage(fs)
		},
		Pressure: func() []PressureStat {
			return updatePressure(fs)
		},
		Processes: func() []ProcessInfo {
			return updateProcessListWithProvider(&staticInfo, runner, processProvider)
		},
	}

	if staticInfo.GPUCount > 0 {
		collectors.LiveGPU = func() []GPUUsage {
			return updateLiveGPUUsage(staticInfo.GPUCount, runner)
		}
	}

	if usesContainerCPU(staticInfo) {
		collectors.ContainerCPU = func() float64 {
			return updateContainerCPUUsage(state, fs)
		}
	} else {
		collectors.HostCPU = func() float64 {
			return updateHostCPUUsageWithProvider(hostProvider)
		}
	}

	if usesContainerMemory(staticInfo) {
		collectors.ContainerMem = func() float64 {
			return updateContainerMemUsage(staticInfo.CgroupVersion, fs)
		}
	} else {
		collectors.HostMem = func() float64 {
			return updateHostMemUsageWithProvider(hostProvider)
		}
	}

	return collectors
}

func usesContainerCPU(staticInfo StaticInfo) bool {
	return staticInfo.ContainerCPULimit > 0 && staticInfo.ContainerCPULimit != float64(staticInfo.HostCores)
}

func usesContainerMemory(staticInfo StaticInfo) bool {
	return staticInfo.ContainerMemLimitBytes > 0
}

// collectDynamicInfo runs all collectors concurrently and returns the snapshot.
func collectDynamicInfo(collectors DynamicCollectorSet) DynamicSnapshot {
	dynamic, _ := collectDynamicInfoWithTiming(collectors, nil)
	return dynamic
}

type collectorEntry struct {
	name    string
	collect func()
}

func collectDynamicInfoWithTiming(
	collectors DynamicCollectorSet,
	now func() time.Time,
) (DynamicSnapshot, map[string]time.Duration) {
	if now == nil {
		now = time.Now
	}

	var dynamic DynamicSnapshot
	var timingMu sync.Mutex
	timings := make(map[string]time.Duration, dynamicCollectorCount)
	var wg sync.WaitGroup

	run := func(name string, collect func()) {
		defer wg.Done()
		start := now()
		collect()
		elapsed := now().Sub(start)
		timingMu.Lock()
		timings[name] = elapsed
		timingMu.Unlock()
	}

	for _, entry := range dynamicCollectorEntries(collectors, &dynamic) {
		wg.Add(1)
		go run(entry.name, entry.collect)
	}
	wg.Wait()

	return dynamic, timings
}

func dynamicCollectorEntries(collectors DynamicCollectorSet, dynamic *DynamicSnapshot) []collectorEntry {
	collectorEntries := make([]collectorEntry, 0, dynamicCollectorCount)
	if collectors.ContainerCPU != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "container_cpu",
			collect: func() { dynamic.ContainerCPUUsage = collectors.ContainerCPU() },
		})
	}
	if collectors.ContainerMem != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "container_mem",
			collect: func() { dynamic.ContainerMemUsedGB = collectors.ContainerMem() },
		})
	}
	if collectors.LiveGPU != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "live_gpu",
			collect: func() { dynamic.LiveGPUUsage = collectors.LiveGPU() },
		})
	}
	if collectors.Storage != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "storage",
			collect: func() { dynamic.StorageUsage = collectors.Storage() },
		})
	}
	if collectors.Network != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "network",
			collect: func() { dynamic.NetworkUsage = collectors.Network() },
		})
	}
	if collectors.DiskIO != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "disk_io",
			collect: func() { dynamic.DiskIOUsage = collectors.DiskIO() },
		})
	}
	if collectors.CgroupEvents != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "cgroup_events",
			collect: func() { dynamic.CgroupEvents = collectors.CgroupEvents() },
		})
	}
	if collectors.PIDs != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "pids",
			collect: func() { dynamic.PIDUsage = collectors.PIDs() },
		})
	}
	if collectors.Pressure != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "pressure",
			collect: func() { dynamic.Pressure = collectors.Pressure() },
		})
	}
	if collectors.Processes != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "processes",
			collect: func() { dynamic.Processes = collectors.Processes() },
		})
	}
	if collectors.HostCPU != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "host_cpu",
			collect: func() { dynamic.HostCPUUsage = collectors.HostCPU() },
		})
	}
	if collectors.HostMem != nil {
		collectorEntries = append(collectorEntries, collectorEntry{
			name:    "host_mem",
			collect: func() { dynamic.HostMemUsedGB = collectors.HostMem() },
		})
	}
	return collectorEntries
}

func getStaticInfo(fs FileReader, runner CommandRunner, stater Stater) (StaticInfo, error) {
	return getStaticInfoWithProviders(fs, runner, stater, OSHostMetricsProvider{}, OSStorageProvider{})
}

func getStaticInfoWithProviders(
	fs FileReader,
	runner CommandRunner,
	stater Stater,
	hostProvider HostMetricsProvider,
	storageProvider StorageProvider,
) (StaticInfo, error) {
	var info StaticInfo
	var err error
	if _, err = stater.Stat(cgroupCPUMaxPath); os.IsNotExist(err) {
		info.CgroupVersion = CgroupV1
	} else {
		info.CgroupVersion = CgroupV2
	}
	info.HostCores, err = hostProvider.CPUCounts(false)
	if err != nil {
		info.HostCores = 1
	}
	info.ContainerCPULimit = getContainerCPULimit(info.CgroupVersion, info.HostCores, fs)
	info.ContainerMemLimitBytes, info.ContainerMemLimitGB = getContainerMemLimit(info.CgroupVersion, fs)
	hostMem, hostMemErr := hostProvider.VirtualMemory()
	if hostMemErr == nil {
		info.HostMemTotalGB = float64(hostMem.Total) / float64(bytesPerGB)
	} else {
		info.HostMemTotalGB = 0
	}
	info.GPUCount, info.GPUTotalGB, err = getStaticGPUInfo(runner)
	_ = err
	info.StorageMounts, err = getStaticStorageInfoWithProvider(storageProvider)
	_ = err
	return info, nil
}
