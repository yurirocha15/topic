package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/shirou/gopsutil/v4/disk"
)

const (
	diskSectorBytes      = 512 // /proc/diskstats counts 512-byte sectors on every device
	diskStatsFieldCount  = 14
	diskStatsNameField   = 2
	diskStatsFirstCount  = 3
	diskReadsField       = 3
	diskSectorsReadField = 5
	diskWritesField      = 7
	diskSectorsWritten   = 9
)

// readDiskStats reads I/O counters from /proc/diskstats. It matches gopsutil's disk.IOCounters for
// the counters topic uses, without gopsutil's per-device serial number and label lookups.
func readDiskStats(root string, names ...string) (map[string]disk.IOCountersStat, error) {
	data, err := os.ReadFile(filepath.Join(root, "diskstats"))
	if err != nil {
		return nil, fmt.Errorf("read diskstats: %w", err)
	}
	return parseDiskStats(data, names...), nil
}

func parseDiskStats(data []byte, names ...string) map[string]disk.IOCountersStat {
	counters := map[string]disk.IOCountersStat{}
	for line := range bytes.Lines(data) {
		fields := bytes.Fields(line)
		if len(fields) < diskStatsFieldCount {
			continue
		}
		name := string(fields[diskStatsNameField])
		if len(names) > 0 && !slices.Contains(names, name) {
			continue
		}
		values := make([]uint64, diskStatsFieldCount)
		valid, idle := true, true
		for index := diskStatsFirstCount; index < diskStatsFieldCount; index++ {
			value, parseErr := strconv.ParseUint(string(fields[index]), 10, 64)
			if parseErr != nil {
				valid = false
				break
			}
			values[index] = value
			idle = idle && value == 0
		}
		// Like gopsutil, skip malformed lines and devices that have never done any I/O
		if !valid || idle {
			continue
		}
		counters[name] = disk.IOCountersStat{
			Name:       name,
			ReadCount:  values[diskReadsField],
			ReadBytes:  values[diskSectorsReadField] * diskSectorBytes,
			WriteCount: values[diskWritesField],
			WriteBytes: values[diskSectorsWritten] * diskSectorBytes,
		}
	}
	return counters
}
