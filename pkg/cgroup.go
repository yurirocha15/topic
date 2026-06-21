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
	var quota, period uint64
	if cgroupVersion == CgroupV2 {
		cpuMaxStr, err := readStringFromFile(cgroupCPUMaxPath, fs)
		if err != nil {
			return float64(hostCores)
		}
		parts := strings.Fields(cpuMaxStr)
		if len(parts) == minCPUInfoCount {
			if parts[0] == cgroupMaxToken {
				return float64(hostCores)
			}
			quota, err = strconv.ParseUint(parts[0], 10, 64)
			if err != nil {
				return float64(hostCores)
			}
			period, err = strconv.ParseUint(parts[1], 10, 64)
			if err != nil {
				return float64(hostCores)
			}
		}
	} else { // CgroupV1
		q, err1 := readUintFromFile(cgroupV1CPUQuotaPath, fs)
		p, err2 := readUintFromFile(cgroupV1CPUPeriodPath, fs)
		if err1 == nil && err2 == nil {
			quota, period = q, p
		} else {
			return float64(hostCores)
		}
	}
	if quota > 0 && period > 0 {
		return float64(quota) / float64(period)
	}
	return float64(hostCores)
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
	return limitBytes, float64(limitBytes) / float64(bytesPerGB)
}

// updateContainerCPUUsage reads the CPU usage from the cgroup filesystem.
func updateContainerCPUUsage(state *State, fs FileReader) float64 {
	var currentUsage uint64

	if state.static.CgroupVersion == CgroupV2 {
		statStr, err := readStringFromFile(cgroupCPUStatPath, fs)
		if err != nil {
			return 0
		}
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "usage_usec" {
				currentUsage, err = strconv.ParseUint(parts[1], 10, 64)
				if err != nil {
					return 0
				}
				break
			}
		}
	} else { // CgroupV1
		// value is in nanoseconds, convert to microseconds
		ns, err := readUintFromFile(cgroupV1CPUUsagePath, fs)
		if err == nil {
			currentUsage = ns / nanosecondsToMicroseconds
		}
	}

	now := time.Now()
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

// updateContainerMemUsage reads the memory usage from the cgroup filesystem.
func updateContainerMemUsage(cgroupVersion CgroupVersion, fs FileReader) float64 {
	if cgroupVersion == CgroupV2 {
		// 'anon' memory is a good proxy for non-cache process memory
		statStr, err := readStringFromFile(cgroupMemoryStatPath, fs)
		if err != nil {
			return 0
		}
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "anon" {
				bytes, parseErr := strconv.ParseUint(parts[1], 10, 64)
				if parseErr != nil {
					return 0
				}
				return float64(bytes) / float64(bytesPerGB)
			}
		}
	} else { // CgroupV1
		// memory.usage_in_bytes includes file cache, but is the standard.
		bytes, err := readUintFromFile(cgroupV1MemoryUsagePath, fs)
		if err == nil {
			return float64(bytes) / float64(bytesPerGB)
		}
	}
	return 0
}
