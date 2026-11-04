package main

import (
	"bufio"
	"io"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	nvidiaSMIStreamQuery = "--query-gpu=index,utilization.gpu,memory.used"
	nvidiaSMILoopFlag    = "-lms"
	// gpuStreamStaleIntervals is how many refresh intervals a value may age before it is replaced
	// by a one-shot query.
	gpuStreamStaleIntervals = 3
	// gpuStreamRestartDelay limits how often a stream that exited is started again.
	gpuStreamRestartDelay = 30 * time.Second
	gpuStreamFieldCount   = 3
	minGPUStreamInterval  = time.Millisecond
)

// StreamStarter starts a long-running command and returns its output. Closing the output stops
// the command.
type StreamStarter interface {
	Start(name string, arg ...string) (io.ReadCloser, error)
}

// gpuUsageStream keeps one nvidia-smi running that prints GPU usage every refresh interval, so a
// refresh reads the latest values instead of starting nvidia-smi each time.
type gpuUsageStream struct {
	starter  StreamStarter
	interval time.Duration
	now      func() time.Time

	mu        sync.Mutex
	samples   map[int]gpuStreamSample
	output    io.ReadCloser
	running   bool
	stopped   bool
	lastStart time.Time
	finished  chan struct{}
}

type gpuStreamSample struct {
	usage GPUUsage
	at    time.Time
}

func newGPUUsageStream(starter StreamStarter, interval time.Duration) *gpuUsageStream {
	return &gpuUsageStream{
		starter:  starter,
		interval: max(interval, minGPUStreamInterval),
		now:      time.Now,
		samples:  map[int]gpuStreamSample{},
	}
}

// Usage returns the latest usage of every GPU. It reports false until the stream has a fresh value
// for each GPU, so the caller can run a one-shot query instead. It starts the stream when needed.
func (s *gpuUsageStream) Usage(gpuCount int) ([]GPUUsage, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	if !s.running && !s.stopped && (s.lastStart.IsZero() || now.Sub(s.lastStart) >= gpuStreamRestartDelay) {
		s.startLocked(now)
	}

	oldest := now.Add(-gpuStreamStaleIntervals * s.interval)
	usage := make([]GPUUsage, 0, gpuCount)
	for index := range gpuCount {
		sample, ok := s.samples[index]
		if !ok || sample.at.Before(oldest) {
			return nil, false
		}
		usage = append(usage, sample.usage)
	}
	return usage, true
}

func (s *gpuUsageStream) startLocked(now time.Time) {
	s.lastStart = now
	milliseconds := strconv.FormatInt(s.interval.Milliseconds(), 10)
	output, err := s.starter.Start(
		nvidiaSMICommand,
		nvidiaSMIStreamQuery,
		nvidiaSMICSVFormat,
		nvidiaSMILoopFlag,
		milliseconds,
	)
	if err != nil {
		return
	}
	s.output = output
	s.running = true
	s.finished = make(chan struct{})
	go s.read(output, s.finished)
}

// read records each line of output until the command exits.
func (s *gpuUsageStream) read(output io.ReadCloser, finished chan struct{}) {
	defer close(finished)
	scanner := bufio.NewScanner(output)
	for scanner.Scan() {
		usage, ok := parseGPUStreamLine(scanner.Text())
		if !ok {
			continue
		}
		s.mu.Lock()
		s.samples[usage.Index] = gpuStreamSample{usage: usage, at: s.now()}
		s.mu.Unlock()
	}
	_ = output.Close()
	s.mu.Lock()
	s.running = false
	s.output = nil
	s.mu.Unlock()
}

// parseGPUStreamLine parses an "index, utilization, memory used (MB)" line.
func parseGPUStreamLine(line string) (GPUUsage, bool) {
	fields := strings.Split(line, ",")
	if len(fields) != gpuStreamFieldCount {
		return GPUUsage{}, false
	}
	index, err := strconv.Atoi(strings.TrimSpace(fields[0]))
	if err != nil || index < 0 {
		return GPUUsage{}, false
	}
	utilization, err := strconv.Atoi(strings.TrimSpace(fields[1]))
	if err != nil {
		return GPUUsage{}, false
	}
	memoryUsedMB, err := strconv.ParseFloat(strings.TrimSpace(fields[2]), 64)
	if err != nil {
		return GPUUsage{}, false
	}
	return GPUUsage{Index: index, Utilization: utilization, MemUsedGB: memoryUsedMB / bytesPerKB}, true
}

// Stop ends the stream and waits for its reader to finish. The stream does not start again.
func (s *gpuUsageStream) Stop() {
	s.mu.Lock()
	s.stopped = true
	output, finished := s.output, s.finished
	s.mu.Unlock()
	if output == nil {
		return
	}
	_ = output.Close()
	<-finished
}

func (s *gpuUsageStream) isRunning() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.running
}
