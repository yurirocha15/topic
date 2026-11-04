package main

import (
	"testing"

	"github.com/shirou/gopsutil/v4/disk"
)

func TestParseDiskStatsReadsCountersTopicUses(t *testing.T) {
	data := []byte(`   7       0 loop0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 259       0 nvme0n1 1200 30 96000 400 800 20 64000 300 0 500 700 0 0 0 0 0 0
 259       1 nvme0n1p1 10 0 80 1 0 0 0 0 0 2 1
   8       0 sda 5 0 bad 1 0 0 0 0 0 1 1
   8      16 sdb 1 2
`)
	got := parseDiskStats(data)
	want := map[string]disk.IOCountersStat{
		"nvme0n1": {
			Name:       "nvme0n1",
			ReadCount:  1200,
			ReadBytes:  96000 * 512,
			WriteCount: 800,
			WriteBytes: 64000 * 512,
		},
		"nvme0n1p1": {Name: "nvme0n1p1", ReadCount: 10, ReadBytes: 80 * 512},
	}
	if len(got) != len(want) {
		t.Fatalf("expected %d devices, got %v", len(want), got)
	}
	for name, counters := range want {
		if got[name] != counters {
			t.Errorf("%s: expected %+v, got %+v", name, counters, got[name])
		}
	}

	if filtered := parseDiskStats(data, "nvme0n1p1"); len(filtered) != 1 || filtered["nvme0n1p1"].ReadCount != 10 {
		t.Fatalf("expected only the requested device, got %v", filtered)
	}
}

func TestReadDiskStatsMatchesGopsutil(t *testing.T) {
	reference, err := disk.IOCounters()
	if err != nil {
		t.Skipf("gopsutil cannot read disk counters here: %v", err)
	}
	got, err := readDiskStats(procRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(reference) {
		t.Fatalf("expected %d devices like gopsutil, got %d", len(reference), len(got))
	}
	for name, want := range reference {
		counters, ok := got[name]
		if !ok {
			t.Errorf("missing device %s", name)
			continue
		}
		// Counters only grow, and this read comes after gopsutil's.
		if counters.ReadCount < want.ReadCount || counters.WriteCount < want.WriteCount ||
			counters.ReadBytes < want.ReadBytes || counters.WriteBytes < want.WriteBytes {
			t.Errorf("%s: %+v is behind gopsutil's %+v", name, counters, want)
		}
	}
}
