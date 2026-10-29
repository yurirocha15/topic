package main

import (
	"math"
	"testing"
	"time"
)

const (
	testV2CPUStat  = "/sys/fs/cgroup/cpu.stat"
	testV1CPUUsage = "/sys/fs/cgroup/cpuacct/cpuacct.usage"
)

func TestGetContainerMemLimitV1(t *testing.T) {
	cases := []struct {
		name      string
		value     string
		wantBytes int64
		wantGB    float64
	}{
		{"kernel page-aligned unlimited value is unlimited", "9223372036854771712", 0, 0},
		{"max int64 is unlimited", "9223372036854775807", 0, 0},
		{"huge non-canonical value is unlimited", "9223372036854710272", 0, 0},
		{"exactly 1<<62 is unlimited", "4611686018427387904", 0, 0},
		{"real limit returns bytes and GB", "536870912", 536870912, 0.5},
		{"large real limit 1<<50 returns itself", "1125899906842624", 1125899906842624, 1048576},
		{"one below 1<<62 returns itself", "4611686018427387903", 4611686018427387903, 4294967296},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fs := MockFileReader{files: map[string]string{cgroupV1MemoryLimitPath: tc.value}}
			gotBytes, gotGB := getContainerMemLimit(CgroupV1, fs)
			if gotBytes != tc.wantBytes || math.Abs(gotGB-tc.wantGB) > 1e-9 {
				t.Errorf("got (%d, %f), want (%d, %f)", gotBytes, gotGB, tc.wantBytes, tc.wantGB)
			}
		})
	}
}

func newCPUState(version CgroupVersion) *State {
	return &State{static: StaticInfo{CgroupVersion: version, ContainerCPULimit: 1}}
}

func approx(got, want float64) bool { return math.Abs(got-want) <= 1.0 }

func TestUpdateContainerCPUUsageFirstSampleReturnsZero(t *testing.T) {
	for _, v := range []CgroupVersion{CgroupV2, CgroupV1} {
		name := "v2"
		path, val := testV2CPUStat, "usage_usec 5000000"
		if v == CgroupV1 {
			name, path, val = "v1", testV1CPUUsage, "5000000000"
		}
		t.Run(name, func(t *testing.T) {
			state := newCPUState(v)
			state.prevCPUTime = time.Now().Add(-50 * time.Millisecond)
			got := updateContainerCPUUsage(state, MockFileReader{files: map[string]string{path: val}})
			if got != 0 {
				t.Errorf("first sample returned %.2f, want 0", got)
			}
		})
	}
}

func TestUpdateContainerCPUUsageSecondSampleComputesDelta(t *testing.T) {
	for _, v := range []CgroupVersion{CgroupV2, CgroupV1} {
		name := "v2"
		path, first, second := testV2CPUStat, "usage_usec 5000000", "usage_usec 5500000"
		if v == CgroupV1 {
			name, path, first, second = "v1", testV1CPUUsage, "5000000000", "5500000000"
		}
		t.Run(name, func(t *testing.T) {
			files := map[string]string{path: first}
			fs := MockFileReader{files: files}
			state := newCPUState(v)
			updateContainerCPUUsage(state, fs)
			files[path] = second
			state.prevCPUTime = time.Now().Add(-1 * time.Second)
			got := updateContainerCPUUsage(state, fs)
			if !approx(got, 50) {
				t.Errorf("second sample returned %.2f, want ~50", got)
			}
		})
	}
}

func TestUpdateContainerCPUUsageReadFailureKeepsBaseline(t *testing.T) {
	cases := []struct {
		name        string
		version     CgroupVersion
		path        string
		base, after string
		broken      string // empty means the file is missing
		wantPercent float64
	}{
		// 100000 us over 2s on 1 CPU = 5%
		{"v2 file missing", CgroupV2, testV2CPUStat, "usage_usec 5000000", "usage_usec 5100000", "", 5},
		{
			"v2 no usage_usec line",
			CgroupV2,
			testV2CPUStat,
			"usage_usec 5000000",
			"usage_usec 5100000",
			"nr_periods 1\nthrottled_usec 0",
			5,
		},
		{
			"v2 malformed usage_usec",
			CgroupV2,
			testV2CPUStat,
			"usage_usec 5000000",
			"usage_usec 5100000",
			"usage_usec abc",
			5,
		},
		{"v1 file missing", CgroupV1, testV1CPUUsage, "5000000000", "5100000000", "", 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			files := map[string]string{tc.path: tc.base}
			fs := MockFileReader{files: files}
			state := newCPUState(tc.version)
			updateContainerCPUUsage(state, fs)

			if tc.broken == "" {
				delete(files, tc.path)
			} else {
				files[tc.path] = tc.broken
			}
			state.prevCPUTime = time.Now().Add(-1 * time.Second)
			if got := updateContainerCPUUsage(state, fs); got != 0 {
				t.Fatalf("read failure returned %.2f, want 0", got)
			}
			if state.prevCPUUsage != 5000000 {
				t.Errorf("baseline after failed read = %d, want 5000000", state.prevCPUUsage)
			}

			files[tc.path] = tc.after
			state.prevCPUTime = time.Now().Add(-2 * time.Second)
			got := updateContainerCPUUsage(state, fs)
			if !approx(got, tc.wantPercent) {
				t.Errorf("after recovery got %.2f, want ~%.0f (no spike from reset baseline)", got, tc.wantPercent)
			}
		})
	}
}

func TestUpdateContainerCPUUsageCounterRegressionResetsBaseline(t *testing.T) {
	for _, v := range []CgroupVersion{CgroupV2, CgroupV1} {
		name := "v2"
		path := testV2CPUStat
		hi, lo, next := "usage_usec 5000000", "usage_usec 1000000", "usage_usec 1500000"
		if v == CgroupV1 {
			name, path = "v1", testV1CPUUsage
			hi, lo, next = "5000000000", "1000000000", "1500000000"
		}
		t.Run(name, func(t *testing.T) {
			files := map[string]string{path: hi}
			fs := MockFileReader{files: files}
			state := newCPUState(v)
			updateContainerCPUUsage(state, fs)

			files[path] = lo
			state.prevCPUTime = time.Now().Add(-1 * time.Second)
			if got := updateContainerCPUUsage(state, fs); got != 0 {
				t.Fatalf("counter regression returned %.2f, want 0", got)
			}

			files[path] = next
			state.prevCPUTime = time.Now().Add(-1 * time.Second)
			got := updateContainerCPUUsage(state, fs)
			if !approx(got, 50) {
				t.Errorf("after reset got %.2f, want ~50 (baseline must be the new lower value)", got)
			}
		})
	}
}
