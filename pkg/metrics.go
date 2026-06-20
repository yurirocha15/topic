package main

import (
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	historySize               = 30
	pressureResourceCount     = 3
	keyedMetricFieldCount     = 2
	keyValueFieldCount        = 2
	alertCapacity             = 4
	alertMemoryWarnPercent    = 85.0
	alertMemoryCritPercent    = 95.0
	alertPIDWarnPercent       = 85.0
	alertPressureWarnAvg10    = 10.0
	bytesPerSecondToMiBSecond = 1024 * 1024
	resourceCPU               = "cpu"
	resourceMemory            = "memory"
	resourceIO                = "io"
	alertWarning              = "warning"
	alertCritical             = "critical"
)

func updateNetworkUsageWithProvider(state *State, provider NetworkProvider) []NetworkUsage {
	counters, err := provider.IOCounters(true)
	if err != nil || len(counters) == 0 {
		return nil
	}

	now := time.Now()
	elapsed := now.Sub(state.prevNetTime).Seconds()
	current := make(map[string]NetworkCounter, len(counters))
	usage := make([]NetworkUsage, 0, len(counters))
	for _, counter := range counters {
		if counter.Name == "" || counter.Name == "lo" {
			continue
		}
		next := NetworkCounter{
			RXBytes: counter.BytesRecv,
			TXBytes: counter.BytesSent,
			RXErrs:  counter.Errin,
			TXErrs:  counter.Errout,
		}
		current[counter.Name] = next
		prev, ok := state.prevNetwork[counter.Name]
		if !ok || elapsed <= 0 {
			usage = append(usage, NetworkUsage{Name: counter.Name, RXErrors: counter.Errin, TXErrors: counter.Errout})
			continue
		}
		usage = append(usage, NetworkUsage{
			Name:          counter.Name,
			RXBytesPerSec: rateUint64(prev.RXBytes, next.RXBytes, elapsed),
			TXBytesPerSec: rateUint64(prev.TXBytes, next.TXBytes, elapsed),
			RXErrors:      counter.Errin,
			TXErrors:      counter.Errout,
		})
	}
	sort.Slice(usage, func(i, j int) bool {
		return usage[i].RXBytesPerSec+usage[i].TXBytesPerSec > usage[j].RXBytesPerSec+usage[j].TXBytesPerSec
	})
	state.prevNetwork = current
	state.prevNetTime = now
	return usage
}

func updateDiskIOUsageWithProvider(state *State, provider DiskIOProvider) []DiskIOUsage {
	counters, err := provider.IOCounters()
	if err != nil || len(counters) == 0 {
		return nil
	}

	now := time.Now()
	elapsed := now.Sub(state.prevDiskTime).Seconds()
	current := make(map[string]DiskIOCounter, len(counters))
	usage := make([]DiskIOUsage, 0, len(counters))
	for name, counter := range counters {
		next := DiskIOCounter{
			ReadBytes:  counter.ReadBytes,
			WriteBytes: counter.WriteBytes,
			ReadCount:  counter.ReadCount,
			WriteCount: counter.WriteCount,
		}
		current[name] = next
		prev, ok := state.prevDiskIO[name]
		if !ok || elapsed <= 0 {
			usage = append(usage, DiskIOUsage{Name: name})
			continue
		}
		usage = append(usage, DiskIOUsage{
			Name:             name,
			ReadBytesPerSec:  rateUint64(prev.ReadBytes, next.ReadBytes, elapsed),
			WriteBytesPerSec: rateUint64(prev.WriteBytes, next.WriteBytes, elapsed),
			ReadOpsPerSec:    rateUint64(prev.ReadCount, next.ReadCount, elapsed),
			WriteOpsPerSec:   rateUint64(prev.WriteCount, next.WriteCount, elapsed),
		})
	}
	sort.Slice(usage, func(i, j int) bool {
		return usage[i].ReadBytesPerSec+usage[i].WriteBytesPerSec > usage[j].ReadBytesPerSec+usage[j].WriteBytesPerSec
	})
	state.prevDiskIO = current
	state.prevDiskTime = now
	return usage
}

func rateUint64(previous uint64, current uint64, seconds float64) float64 {
	if current < previous || seconds <= 0 {
		return 0
	}
	return float64(current-previous) / seconds
}

func updateCgroupEvents(cgroupVersion CgroupVersion, fs FileReader) CgroupEvents {
	events := CgroupEvents{}
	if cgroupVersion == CgroupV2 {
		for key, value := range readKeyedUintFile(cgroupMemoryEventsPath, fs) {
			switch key {
			case "high":
				events.MemoryHigh = value
			case "oom":
				events.MemoryOOM = value
			case "oom_kill":
				events.MemoryOOMKill = value
			}
		}
	}
	for key, value := range readKeyedUintFile(cgroupCPUStatPath, fs) {
		if key == "nr_throttled" {
			events.CPUThrottledPeriods = value
		}
	}
	return events
}

func updatePIDUsage(fs FileReader) PIDUsage {
	current, currentErr := readUintFromFile(cgroupPIDsCurrentPath, fs)
	if currentErr != nil {
		return PIDUsage{}
	}
	maxText, maxErr := readStringFromFile(cgroupPIDsMaxPath, fs)
	if maxErr != nil {
		return PIDUsage{Current: current}
	}
	if maxText == cgroupMaxToken {
		return PIDUsage{Current: current, MaxText: cgroupMaxToken}
	}
	maxValue, err := strconv.ParseUint(maxText, 10, 64)
	if err != nil {
		return PIDUsage{Current: current}
	}
	return PIDUsage{Current: current, Max: maxValue, MaxText: maxText}
}

func updatePressure(fs FileReader) []PressureStat {
	pressure := make([]PressureStat, 0, pressureResourceCount)
	for _, item := range []struct {
		resource string
		path     string
	}{
		{resourceCPU, cgroupCPUPressurePath},
		{resourceMemory, cgroupMemoryPressurePath},
		{resourceIO, cgroupIOPressurePath},
	} {
		if stat, ok := readPressureFile(item.resource, item.path, fs); ok {
			pressure = append(pressure, stat)
		}
	}
	return pressure
}

func readKeyedUintFile(path string, fs FileReader) map[string]uint64 {
	content, err := readStringFromFile(path, fs)
	if err != nil {
		return nil
	}
	values := make(map[string]uint64)
	for _, line := range strings.Split(content, "\n") {
		fields := strings.Fields(line)
		if len(fields) != keyedMetricFieldCount {
			continue
		}
		value, parseErr := strconv.ParseUint(fields[1], 10, 64)
		if parseErr != nil {
			continue
		}
		values[fields[0]] = value
	}
	return values
}

func readPressureFile(resource string, path string, fs FileReader) (PressureStat, bool) {
	content, err := readStringFromFile(path, fs)
	if err != nil {
		return PressureStat{}, false
	}
	stat := PressureStat{Resource: resource}
	for _, line := range strings.Split(content, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		switch fields[0] {
		case "some":
			stat.SomeAvg10 = pressureAvg10(fields)
		case "full":
			stat.FullAvg10 = pressureAvg10(fields)
		}
	}
	return stat, true
}

func pressureAvg10(fields []string) float64 {
	for _, field := range fields[1:] {
		keyValue := strings.SplitN(field, "=", keyValueFieldCount)
		if len(keyValue) != keyValueFieldCount || keyValue[0] != "avg10" {
			continue
		}
		value, err := strconv.ParseFloat(keyValue[1], 64)
		if err == nil {
			return value
		}
	}
	return 0
}

func evaluateAlerts(staticInfo StaticInfo, dynamic DynamicSnapshot) []Alert {
	alerts := make([]Alert, 0, alertCapacity)
	memPercent := 0.0
	if staticInfo.ContainerMemLimitBytes > 0 {
		memPercent = (dynamic.ContainerMemUsedGB * bytesPerGB / float64(staticInfo.ContainerMemLimitBytes)) * percentMultiplier
	}
	switch {
	case memPercent >= alertMemoryCritPercent:
		alerts = append(alerts, Alert{Level: alertCritical, Message: "container memory is above 95%"})
	case memPercent >= alertMemoryWarnPercent:
		alerts = append(alerts, Alert{Level: alertWarning, Message: "container memory is above 85%"})
	}
	if dynamic.CgroupEvents.MemoryOOMKill > 0 {
		alerts = append(alerts, Alert{Level: alertCritical, Message: "cgroup reports OOM kills"})
	}
	if dynamic.CgroupEvents.CPUThrottledPeriods > 0 {
		alerts = append(alerts, Alert{Level: alertWarning, Message: "cgroup reports CPU throttling"})
	}
	if dynamic.PIDUsage.Max > 0 {
		pidPercent := (float64(dynamic.PIDUsage.Current) / float64(dynamic.PIDUsage.Max)) * percentMultiplier
		if pidPercent >= alertPIDWarnPercent {
			alerts = append(alerts, Alert{Level: alertWarning, Message: "PID usage is above 85%"})
		}
	}
	for _, pressure := range dynamic.Pressure {
		if pressure.SomeAvg10 >= alertPressureWarnAvg10 || pressure.FullAvg10 >= alertPressureWarnAvg10 {
			alerts = append(alerts, Alert{
				Level:   alertWarning,
				Message: pressure.Resource + " pressure avg10 is elevated",
			})
		}
	}
	return alerts
}

func updateMetricsHistory(history *MetricsHistory, staticInfo StaticInfo, dynamic DynamicSnapshot) {
	history.CPU.Add(dynamic.ContainerCPUUsage + dynamic.HostCPUUsage)
	memPercent := 0.0
	if staticInfo.ContainerMemLimitBytes > 0 {
		memPercent = (dynamic.ContainerMemUsedGB * bytesPerGB / float64(staticInfo.ContainerMemLimitBytes)) * percentMultiplier
	} else if staticInfo.HostMemTotalGB > 0 {
		memPercent = (dynamic.HostMemUsedGB / staticInfo.HostMemTotalGB) * percentMultiplier
	}
	history.Memory.Add(memPercent)
	history.GPU.Add(maxGPUUtilization(dynamic.LiveGPUUsage))
	history.Network.Add(totalNetworkRate(dynamic.NetworkUsage) / bytesPerSecondToMiBSecond)
	history.DiskIO.Add(totalDiskIORate(dynamic.DiskIOUsage) / bytesPerSecondToMiBSecond)
}

func (ring *HistoryRing) Add(value float64) {
	if len(ring.Values) == 0 {
		ring.Values = make([]float64, historySize)
	}
	ring.Values[ring.Next] = value
	ring.Next = (ring.Next + 1) % len(ring.Values)
	if ring.Next == 0 {
		ring.Filled = true
	}
}

func (ring *HistoryRing) Ordered() []float64 {
	if len(ring.Values) == 0 {
		return nil
	}
	if !ring.Filled {
		return append([]float64(nil), ring.Values[:ring.Next]...)
	}
	ordered := make([]float64, 0, len(ring.Values))
	ordered = append(ordered, ring.Values[ring.Next:]...)
	ordered = append(ordered, ring.Values[:ring.Next]...)
	return ordered
}

func sparkline(ring HistoryRing) string {
	values := (&ring).Ordered()
	if len(values) == 0 {
		return ""
	}
	const blocks = "▁▂▃▄▅▆▇█"
	maxValue := 0.0
	for _, value := range values {
		if value > maxValue {
			maxValue = value
		}
	}
	if maxValue <= 0 {
		return strings.Repeat("▁", len(values))
	}
	var builder strings.Builder
	for _, value := range values {
		index := int((value / maxValue) * float64(len([]rune(blocks))-1))
		builder.WriteRune([]rune(blocks)[index])
	}
	return builder.String()
}

func maxGPUUtilization(usage []GPUUsage) float64 {
	maxValue := 0.0
	for _, gpu := range usage {
		if float64(gpu.Utilization) > maxValue {
			maxValue = float64(gpu.Utilization)
		}
	}
	return maxValue
}

func totalNetworkRate(usage []NetworkUsage) float64 {
	total := 0.0
	for _, network := range usage {
		total += network.RXBytesPerSec + network.TXBytesPerSec
	}
	return total
}

func totalDiskIORate(usage []DiskIOUsage) float64 {
	total := 0.0
	for _, disk := range usage {
		total += disk.ReadBytesPerSec + disk.WriteBytesPerSec
	}
	return total
}
