package main

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeStreamStarter hands out pipes the test writes nvidia-smi output into.
type fakeStreamStarter struct {
	mu      sync.Mutex
	starts  [][]string
	writers []*io.PipeWriter
	err     error
}

func (f *fakeStreamStarter) Start(name string, arg ...string) (io.ReadCloser, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.starts = append(f.starts, append([]string{name}, arg...))
	if f.err != nil {
		return nil, f.err
	}
	reader, writer := io.Pipe()
	f.writers = append(f.writers, writer)
	return reader, nil
}

func (f *fakeStreamStarter) startCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.starts)
}

func (f *fakeStreamStarter) write(t *testing.T, lines ...string) {
	t.Helper()
	f.mu.Lock()
	writer := f.writers[len(f.writers)-1]
	f.mu.Unlock()
	if _, err := io.WriteString(writer, strings.Join(lines, "\n")+"\n"); err != nil {
		t.Fatalf("write stream: %v", err)
	}
}

// fakeClock is a settable clock shared with the stream's reader goroutine.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

func newTestGPUStream(t *testing.T, starter StreamStarter) (*gpuUsageStream, *fakeClock) {
	t.Helper()
	clock := &fakeClock{now: time.Unix(1000, 0)}
	stream := newGPUUsageStream(starter, time.Second)
	stream.now = clock.Now
	t.Cleanup(stream.Stop)
	return stream, clock
}

// waitForGPUStream polls until the stream reports usage, which happens once the reader goroutine
// has recorded the written lines.
func waitForGPUStream(t *testing.T, stream *gpuUsageStream, gpuCount int) []GPUUsage {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if usage, ok := stream.Usage(gpuCount); ok {
			return usage
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("stream never reported usage")
	return nil
}

func TestGPUStreamStartsOneLongRunningQuery(t *testing.T) {
	starter := &fakeStreamStarter{}
	stream, _ := newTestGPUStream(t, starter)

	if _, ok := stream.Usage(2); ok {
		t.Fatal("expected no usage before nvidia-smi prints anything")
	}
	want := []string{
		"nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
		"--format=csv,noheader,nounits", "-lms", "1000",
	}
	if starter.startCount() != 1 || strings.Join(starter.starts[0], " ") != strings.Join(want, " ") {
		t.Fatalf("expected one start with %v, got %v", want, starter.starts)
	}

	starter.write(t, "0, 35, 2048", "garbage", "1, 90, 8192")
	usage := waitForGPUStream(t, stream, 2)
	if usage[0].Index != 0 || usage[0].Utilization != 35 || usage[0].MemUsedGB != 2 ||
		usage[1].Index != 1 || usage[1].Utilization != 90 || usage[1].MemUsedGB != 8 {
		t.Fatalf("unexpected usage %+v", usage)
	}
	if starter.startCount() != 1 {
		t.Fatalf("expected the running stream to be reused, got %d starts", starter.startCount())
	}
}

func TestGPUStreamNeedsEveryGPU(t *testing.T) {
	starter := &fakeStreamStarter{}
	stream, _ := newTestGPUStream(t, starter)
	stream.Usage(2)
	starter.write(t, "0, 35, 2048")
	waitForGPUStream(t, stream, 1)
	if _, ok := stream.Usage(2); ok {
		t.Fatal("expected no usage while GPU 1 has not been reported")
	}
}

func TestGPUStreamTreatsOldValuesAsMissing(t *testing.T) {
	starter := &fakeStreamStarter{}
	stream, clock := newTestGPUStream(t, starter)
	stream.Usage(1)
	starter.write(t, "0, 35, 2048")
	waitForGPUStream(t, stream, 1)

	clock.Advance(4 * time.Second)
	if _, ok := stream.Usage(1); ok {
		t.Fatal("expected values older than three intervals to be ignored")
	}
}

func TestGPUStreamRestartsAfterExitAtMostEvery30Seconds(t *testing.T) {
	starter := &fakeStreamStarter{}
	stream, clock := newTestGPUStream(t, starter)
	stream.Usage(1)
	starter.write(t, "0, 35, 2048")
	waitForGPUStream(t, stream, 1)

	// nvidia-smi exits
	if err := starter.writers[0].Close(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for stream.isRunning() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if stream.isRunning() {
		t.Fatal("stream still running after its output ended")
	}

	clock.Advance(10 * time.Second)
	stream.Usage(1)
	if starter.startCount() != 1 {
		t.Fatalf("expected no restart within 30 s, got %d starts", starter.startCount())
	}
	clock.Advance(25 * time.Second)
	stream.Usage(1)
	if starter.startCount() != 2 {
		t.Fatalf("expected a restart after 30 s, got %d starts", starter.startCount())
	}
}

func TestGPUStreamStartFailureFallsBack(t *testing.T) {
	starter := &fakeStreamStarter{err: errors.New("no nvidia-smi")}
	stream, _ := newTestGPUStream(t, starter)
	if _, ok := stream.Usage(1); ok {
		t.Fatal("expected no usage when nvidia-smi cannot start")
	}
	stream.Usage(1)
	if starter.startCount() != 1 {
		t.Fatalf("expected one attempt per 30 s, got %d", starter.startCount())
	}
}

func TestGPUStreamStopEndsTheCommandAndStaysStopped(t *testing.T) {
	starter := &fakeStreamStarter{}
	stream, clock := newTestGPUStream(t, starter)
	stream.Usage(1)
	stream.Stop()
	if stream.isRunning() {
		t.Fatal("expected Stop to end the stream")
	}
	clock.Advance(time.Minute)
	stream.Usage(1)
	if starter.startCount() != 1 {
		t.Fatalf("expected no restart after Stop, got %d starts", starter.startCount())
	}
}

func TestLiveGPUCollectorUsesStreamInsteadOfOneShotQuery(t *testing.T) {
	starter := &fakeStreamStarter{}
	stream, _ := newTestGPUStream(t, starter)
	stream.Usage(1)
	starter.write(t, "0, 35, 2048")
	waitForGPUStream(t, stream, 1)

	runner := &CountingCommandRunner{}
	state := &State{static: StaticInfo{GPUCount: 1, GPUTotalGB: []float64{16}}, gpuStream: stream}
	collectors := newDynamicCollectorSet(
		state, MockFileReader{}, runner, state.static,
		MockHostMetricsProvider{}, MockStorageProvider{}, MockProcessProvider{},
		MockNetworkProvider{}, MockDiskIOProvider{},
	)
	usage := collectors.LiveGPU()
	if len(usage) != 1 || usage[0].Utilization != 35 {
		t.Fatalf("expected streamed usage, got %+v", usage)
	}
	if runner.calls != 0 {
		t.Fatalf("expected no one-shot nvidia-smi calls, got %d", runner.calls)
	}
}

func TestOSCommandRunnerStartStopsCommandOnClose(t *testing.T) {
	output, err := OSCommandRunner{Context: context.Background()}.Start("sh", "-c", "echo ready; exec sleep 30")
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, len("ready\n"))
	if _, err = io.ReadFull(output, buffer); err != nil || string(buffer) != "ready\n" {
		t.Fatalf("expected the command's output, got %q (%v)", buffer, err)
	}
	start := time.Now()
	if err = output.Close(); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("Close took %v; the command was not stopped", elapsed)
	}
}
