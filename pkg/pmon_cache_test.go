package main

import (
	"testing"
	"time"
)

func TestPMonRunsAtMostEveryThreeSeconds(t *testing.T) {
	var cache gpuProcessCache
	queries := 0
	query := func() map[int32]GPUProcessInfo {
		queries++
		return map[int32]GPUProcessInfo{int32(queries): {GPUUtil: uint64(queries)}}
	}
	start := time.Unix(1000, 0)

	for _, step := range []struct {
		after   time.Duration
		queries int
	}{
		{0, 1},                 // first refresh queries
		{time.Second, 1},       // reused
		{2 * time.Second, 1},   // reused
		{3 * time.Second, 2},   // three seconds old: query again
		{4 * time.Second, 2},   // reused
		{6 * time.Second, 3},   // query again
		{5 * time.Second, 4},   // the clock went back: query again
		{5*time.Second + 1, 4}, // reused
		{time.Minute, 5},       // long pause: query again
		{time.Minute + time.Second, 5},
	} {
		processes := cache.get(start.Add(step.after), query)
		if queries != step.queries {
			t.Fatalf("at +%v: expected %d pmon runs, got %d", step.after, step.queries, queries)
		}
		if processes[int32(queries)].GPUUtil != uint64(queries) {
			t.Fatalf("at +%v: expected the latest pmon result, got %+v", step.after, processes)
		}
	}
}

func TestPMonFailureIsNotRetriedEveryRefresh(t *testing.T) {
	var cache gpuProcessCache
	queries := 0
	failing := func() map[int32]GPUProcessInfo { queries++; return nil }
	start := time.Unix(1000, 0)
	cache.get(start, failing)
	cache.get(start.Add(time.Second), failing)
	if queries != 1 {
		t.Fatalf("expected one pmon run within three seconds of a failure, got %d", queries)
	}
}

func TestStaticBuildCollectorReusesPMon(t *testing.T) {
	runner := &CountingCommandRunner{}
	state := &State{static: StaticInfo{GPUCount: 1, GPUTotalGB: []float64{16}}}
	for range 3 {
		collectors := newDynamicCollectorSet(
			state, MockFileReader{}, runner, state.static,
			MockHostMetricsProvider{}, MockStorageProvider{}, newProcfsReader(procRoot),
			MockNetworkProvider{}, MockDiskIOProvider{},
		)
		collectors.Processes()
	}
	if runner.calls != 1 {
		t.Fatalf("expected one pmon run for three quick refreshes, got %d", runner.calls)
	}
}
