package main

import (
	"strconv"
	"strings"
	"time"
)

// readUintFromFile reads a uint64 value from a file, trimming whitespace.
func readUintFromFile(path string, fs FileReader) (uint64, error) {
	data, err := fs.ReadFile(path)
	if err != nil {
		return 0, err
	}
	val, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, err
	}
	return val, nil
}

// readStringFromFile reads a string from a file, trimming whitespace.
func readStringFromFile(path string, fs FileReader) (string, error) {
	data, err := fs.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

// getContainerCPULimit reads the CPU limit from cgroup files and returns it as a percentage of host cores.
func getContainerCPULimit(cgroupVersion CgroupVersion, hostCores int, fs FileReader) float64 {
	quota, period, ok := readCPUQuota(cgroupVersion, fs)
	if ok && quota > 0 && period > 0 {
		return float64(quota) / float64(period)
	}
	return float64(hostCores)
}

// readCPUQuota returns the CPU quota and period, and false if no limit is set or readable.
func readCPUQuota(cgroupVersion CgroupVersion, fs FileReader) (uint64, uint64, bool) {
	if cgroupVersion == CgroupV1 {
		quota, quotaErr := readUintFromFile(cgroupV1CPUQuotaPath, fs)
		period, periodErr := readUintFromFile(cgroupV1CPUPeriodPath, fs)
		return quota, period, quotaErr == nil && periodErr == nil
	}

	cpuMaxStr, err := readStringFromFile(cgroupCPUMaxPath, fs)
	if err != nil {
		return 0, 0, false
	}
	parts := strings.Fields(cpuMaxStr)
	if len(parts) != minCPUInfoCount || parts[0] == cgroupMaxToken {
		return 0, 0, false
	}
	quota, quotaErr := strconv.ParseUint(parts[0], 10, 64)
	period, periodErr := strconv.ParseUint(parts[1], 10, 64)
	return quota, period, quotaErr == nil && periodErr == nil
}

// getContainerMemLimit reads the memory limit from cgroup files and returns it in bytes and GB.
func getContainerMemLimit(cgroupVersion CgroupVersion, fs FileReader) (int64, float64) {
	var path string
	if cgroupVersion == CgroupV2 {
		path = cgroupMemoryMaxPath
	} else {
		path = cgroupV1MemoryLimitPath
	}
	limitStr, err := readStringFromFile(path, fs)
	if err != nil {
		return 0, 0
	}
	if limitStr == cgroupMaxToken {
		return 0, 0
	}
	limitBytes, err := strconv.ParseInt(limitStr, 10, 64)
	if err != nil {
		return 0, 0
	}
	if limitBytes <= 0 {
		return 0, 0
	}
	if cgroupVersion == CgroupV1 && limitBytes >= cgroupV1UnlimitedMemory {
		return 0, 0
	}
	return limitBytes, float64(limitBytes) / float64(bytesPerGB)
}

// readContainerCPUUsage reads the cumulative container CPU usage in microseconds.
// The boolean is false when the counter could not be read or parsed.
func readContainerCPUUsage(cgroupVersion CgroupVersion, fs FileReader) (uint64, bool) {
	if cgroupVersion == CgroupV2 {
		statStr, err := readStringFromFile(cgroupCPUStatPath, fs)
		if err != nil {
			return 0, false
		}
		for line := range strings.SplitSeq(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "usage_usec" {
				usage, parseErr := strconv.ParseUint(parts[1], 10, 64)
				return usage, parseErr == nil
			}
		}
		return 0, false
	}
	// CgroupV1: value is in nanoseconds, convert to microseconds
	ns, err := readUintFromFile(cgroupV1CPUUsagePath, fs)
	return ns / nanosecondsToMicroseconds, err == nil
}

// updateContainerCPUUsage reads the CPU usage from the cgroup filesystem.
func updateContainerCPUUsage(state *State, fs FileReader) float64 {
	currentUsage, ok := readContainerCPUUsage(state.static.CgroupVersion, fs)
	if !ok {
		return 0
	}

	now := time.Now()
	// Without a baseline, or if the counter went backwards, rebaseline instead of reporting a delta.
	if state.prevCPUUsage == 0 || currentUsage < state.prevCPUUsage {
		state.prevCPUUsage = currentUsage
		state.prevCPUTime = now
		return 0
	}
	timeDelta := now.Sub(state.prevCPUTime).Seconds()

	var cpuPercent float64
	if timeDelta > 0 && currentUsage > state.prevCPUUsage && state.static.ContainerCPULimit > 0 {
		usageDelta := float64(currentUsage - state.prevCPUUsage) // in microseconds
		// usageDelta is how many microseconds of CPU time were used.
		// timeDelta * 1e6 is how many microseconds passed in wall-clock time.
		// Normalize by the number of cores to get a percentage of the container's total capacity.
		cpuPercent = (usageDelta / (timeDelta * secondToMicroseconds)) * percentMultiplier / state.static.ContainerCPULimit
	}

	state.prevCPUUsage = currentUsage
	state.prevCPUTime = now

	if cpuPercent > percentMultiplier {
		return percentMultiplier
	}
	return cpuPercent
}

// updateContainerMemUsage reads the working set: usage minus the inactive file cache the kernel can reclaim.
func updateContainerMemUsage(cgroupVersion CgroupVersion, fs FileReader) float64 {
	usagePath, statPath, inactiveField := cgroupMemoryCurrentPath, cgroupMemoryStatPath, "inactive_file"
	if cgroupVersion == CgroupV1 {
		usagePath, statPath, inactiveField = cgroupV1MemoryUsagePath, cgroupV1MemoryStatPath, "total_inactive_file"
	}
	usage, err := readUintFromFile(usagePath, fs)
	if err != nil {
		return 0
	}
	inactive := readMemoryStatField(statPath, inactiveField, fs)
	if inactive >= usage {
		return 0
	}
	return float64(usage-inactive) / float64(bytesPerGB)
}

// readMemoryStatField returns a memory.stat value, or 0 if it is missing or malformed.
func readMemoryStatField(path, field string, fs FileReader) uint64 {
	statStr, err := readStringFromFile(path, fs)
	if err != nil {
		return 0
	}
	for line := range strings.SplitSeq(statStr, "\n") {
		parts := strings.Fields(line)
		if len(parts) == 2 && parts[0] == field {
			value, parseErr := strconv.ParseUint(parts[1], 10, 64)
			if parseErr != nil {
				return 0
			}
			return value
		}
	}
	return 0
}
