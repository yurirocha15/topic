package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
	"github.com/shirou/gopsutil/v4/mem"
	netio "github.com/shirou/gopsutil/v4/net"
)

func TestParseConfigArgsDefaults(t *testing.T) {
	var stderr bytes.Buffer
	config, err := parseConfigArgs(nil, &stderr)
	if err != nil {
		t.Fatalf("parseConfigArgs() error = %v", err)
	}
	if config.RefreshInterval != time.Second {
		t.Errorf("RefreshInterval = %v, want 1s", config.RefreshInterval)
	}
	if config.InitialSort != SortByCPU || !config.HideASCIIArt || config.Once || config.ShowVersion {
		t.Errorf("unexpected defaults: %+v", config)
	}
}

func TestParseConfigArgsFlags(t *testing.T) {
	var stderr bytes.Buffer
	config, err := parseConfigArgs([]string{"--refresh=2s", "--json", "--sort=mem", "--ascii", "--no-gpu"}, &stderr)
	if err != nil {
		t.Fatalf("parseConfigArgs() error = %v", err)
	}
	if config.RefreshInterval != 2*time.Second {
		t.Errorf("RefreshInterval = %v, want 2s", config.RefreshInterval)
	}
	if !config.JSONOutput || !config.Once {
		t.Errorf("--json should imply --once: %+v", config)
	}
	if config.InitialSort != SortByMemory || config.HideASCIIArt || !config.DisableGPU {
		t.Errorf("flags not applied: %+v", config)
	}
}

func TestParseConfigArgsRejectsNonPositiveRefresh(t *testing.T) {
	for _, value := range []string{"0", "0s", "-1s"} {
		t.Run(value, func(t *testing.T) {
			var stderr bytes.Buffer
			if _, err := parseConfigArgs([]string{"--refresh=" + value}, &stderr); err == nil {
				t.Fatal("expected an error for a non-positive refresh interval")
			}
			if !strings.Contains(stderr.String(), "--refresh must be greater than 0") {
				t.Errorf("stderr = %q, want the refresh error", stderr.String())
			}
		})
	}
}

func TestParseConfigArgsUnknownFlagAndHelp(t *testing.T) {
	var stderr bytes.Buffer
	if _, err := parseConfigArgs([]string{"--nope"}, &stderr); err == nil || errors.Is(err, flag.ErrHelp) {
		t.Fatalf("unknown flag error = %v", err)
	}
	if _, err := parseConfigArgs([]string{"--help"}, &stderr); !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("--help error = %v, want flag.ErrHelp", err)
	}
	if !strings.Contains(stderr.String(), "-refresh") {
		t.Errorf("usage should list flags, got %q", stderr.String())
	}
}

func TestVersionStringPrefersInjectedVersion(t *testing.T) {
	original := Version
	t.Cleanup(func() { Version = original })

	Version = "v1.2.3"
	if got := versionString(); got != "v1.2.3" {
		t.Errorf("versionString() = %q, want v1.2.3", got)
	}
	Version = defaultVersion
	if got := versionString(); got != defaultVersion {
		t.Errorf("versionString() = %q, want %q for a test binary", got, defaultVersion)
	}
}

func TestRunExitCodes(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		want       int
		wantStdout string
		wantStderr string
	}{
		{name: "version", args: []string{"--version"}, want: exitOK, wantStdout: "topic dev\n"},
		{name: "help", args: []string{"--help"}, want: exitOK, wantStderr: "-refresh"},
		{name: "zero refresh", args: []string{"--refresh=0"}, want: exitUsage, wantStderr: "--refresh must be"},
		{name: "negative refresh", args: []string{"--refresh=-1s"}, want: exitUsage, wantStderr: "--refresh must be"},
		{name: "unknown flag", args: []string{"--nope"}, want: exitUsage, wantStderr: "-nope"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if got := run(test.args, &stdout, &stderr); got != test.want {
				t.Errorf("run() = %d, want %d (stderr %q)", got, test.want, stderr.String())
			}
			if test.wantStdout != "" && stdout.String() != test.wantStdout {
				t.Errorf("stdout = %q, want %q", stdout.String(), test.wantStdout)
			}
			if !strings.Contains(stderr.String(), test.wantStderr) {
				t.Errorf("stderr = %q, want it to contain %q", stderr.String(), test.wantStderr)
			}
		})
	}
}

func TestSampleOnceRefreshesAcrossTheWindow(t *testing.T) {
	var events []string
	sampleOnce(
		func() { events = append(events, "refresh") },
		func(window time.Duration) {
			if window != onceSampleWindow {
				t.Errorf("window = %v, want %v", window, onceSampleWindow)
			}
			events = append(events, "wait")
		},
	)
	if got := strings.Join(events, ","); got != "refresh,wait,refresh" {
		t.Errorf("events = %q, want refresh,wait,refresh", got)
	}
}

func TestSampleOnceReportsDeltaMetrics(t *testing.T) {
	const cpuStatPath = "/sys/fs/cgroup/cpu.stat"
	state := &State{static: StaticInfo{CgroupVersion: CgroupV2, ContainerCPULimit: 2, HostCores: 8}}
	files := map[string]string{cpuStatPath: "usage_usec 5000000"}
	network := &MockNetworkProvider{counters: []netio.IOCountersStat{{Name: "eth0", BytesRecv: 1000, BytesSent: 1000}}}

	refresh := func() {
		refreshDynamicStateWithProviders(
			state,
			MockFileReader{files: files},
			&CountingCommandRunner{},
			MockHostMetricsProvider{memory: &mem.VirtualMemoryStat{}},
			MockStorageProvider{},
			MockProcessProvider{},
			network,
			MockDiskIOProvider{},
		)
	}
	sampleOnce(refresh, func(window time.Duration) {
		time.Sleep(window / 10)
		files[cpuStatPath] = "usage_usec 5020000"
		network.counters = []netio.IOCountersStat{{Name: "eth0", BytesRecv: 9000, BytesSent: 5000}}
	})

	if state.dynamic.ContainerCPUUsage <= 0 {
		t.Errorf("ContainerCPUUsage = %v, want a measured value", state.dynamic.ContainerCPUUsage)
	}
	if len(state.dynamic.NetworkUsage) != 1 || state.dynamic.NetworkUsage[0].RXBytesPerSec <= 0 {
		t.Errorf("NetworkUsage = %+v, want a measured rate", state.dynamic.NetworkUsage)
	}
}

func newTestTUI() (*tview.Application, AppConfig, *State) {
	screen := tcell.NewSimulationScreen("UTF-8")
	screen.SetSize(100, 32)
	return tview.NewApplication().SetScreen(screen),
		AppConfig{RefreshInterval: time.Millisecond},
		&State{ui: UIState{ProcessSort: SortByCPU, HideASCIIArt: true}}
}

func waitForRunTUI(t *testing.T, done <-chan error) {
	t.Helper()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("runTUI() error = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runTUI did not stop")
	}
}

func TestRunTUIStopsWhenContextIsCancelled(t *testing.T) {
	app, config, state := newTestTUI()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var refreshes atomic.Int32
	done := make(chan error, 1)
	go func() { done <- runTUI(ctx, app, config, state, func() { refreshes.Add(1) }) }()

	deadline := time.Now().Add(5 * time.Second)
	for refreshes.Load() < 3 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if refreshes.Load() < 3 {
		t.Fatal("periodic refresh did not run")
	}
	cancel()
	waitForRunTUI(t, done)
}

func TestRunTUIStopsWhenContextIsCancelledBeforeStartup(t *testing.T) {
	app, config, state := newTestTUI()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	done := make(chan error, 1)
	go func() { done <- runTUI(ctx, app, config, state, func() {}) }()
	waitForRunTUI(t, done)
}

func TestRunTUIDoesNotOverlapInitialRefresh(t *testing.T) {
	app, config, state := newTestTUI()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var active, overlaps, refreshes atomic.Int32
	refresh := func() {
		if active.Add(1) > 1 {
			overlaps.Add(1)
		}
		defer active.Add(-1)
		// The first refresh outlasts several ticks, like a slow nvidia-smi call.
		if refreshes.Add(1) == 1 {
			time.Sleep(50 * time.Millisecond)
		}
	}

	done := make(chan error, 1)
	go func() { done <- runTUI(ctx, app, config, state, refresh) }()

	deadline := time.Now().Add(5 * time.Second)
	for refreshes.Load() < 3 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	cancel()
	waitForRunTUI(t, done)
	if got := overlaps.Load(); got != 0 {
		t.Errorf("refresh overlapped %d times", got)
	}
}

func TestOSCommandRunnerStopsWithContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	start := time.Now()
	runner := OSCommandRunner{Context: ctx, Timeout: time.Minute}
	if _, err := runner.Output("sleep", "30"); err == nil {
		t.Fatal("expected an error from a cancelled command")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("command took %v to stop", elapsed)
	}
}

func TestOSCommandRunnerWorksWithoutContext(t *testing.T) {
	output, err := OSCommandRunner{Timeout: 5 * time.Second}.Output("echo", "ok")
	if err != nil || strings.TrimSpace(string(output)) != "ok" {
		t.Fatalf("Output() = %q, %v", output, err)
	}
}
