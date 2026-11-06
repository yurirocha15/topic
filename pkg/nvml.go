package main

import (
	"errors"
	"sync"
)

// NVML reads GPU usage inside the topic process, which costs far less CPU than starting
// nvidia-smi. Only builds with the nvml tag can load it, because loading a C library makes the
// binary depend on glibc; the static build keeps using nvidia-smi.

var errNVMLNoGPU = errors.New("NVML reports no GPUs")

// nvmlProcessSample is one per-process utilization sample from NVML.
type nvmlProcessSample struct {
	PID       uint32
	SMUtil    uint32
	TimeStamp uint64 // microseconds, as NVML reports it
}

// nvmlProcessMemory is a process running on a GPU and the GPU memory it uses.
type nvmlProcessMemory struct {
	PID       uint32
	UsedBytes uint64
}

// nvmlLibrary is the part of NVML topic uses. GPUs are addressed by their NVML index, which is
// the index nvidia-smi shows.
type nvmlLibrary interface {
	DeviceCount() (int, error)
	MemoryInfo(index int) (total uint64, used uint64, err error)
	Utilization(index int) (int, error)
	// ProcessUtilization returns the samples taken after since, a timestamp from an earlier
	// sample, or every buffered sample when since is 0.
	ProcessUtilization(index int, since uint64) ([]nvmlProcessSample, error)
	RunningProcesses(index int) ([]nvmlProcessMemory, error)
	Shutdown()
}

// nvmlGPUSource collects GPU usage through NVML. It is safe for concurrent use.
type nvmlGPUSource struct {
	mu       sync.Mutex
	lib      nvmlLibrary
	count    int
	lastSeen []uint64 // newest process sample timestamp per GPU
}

// openNVMLGPUSource loads NVML, or returns an error when this build, the driver, or the machine
// cannot use it.
func openNVMLGPUSource() (*nvmlGPUSource, error) {
	lib, err := openNVML()
	if err != nil {
		return nil, err
	}
	return newNVMLGPUSource(lib)
}

func newNVMLGPUSource(lib nvmlLibrary) (*nvmlGPUSource, error) {
	count, err := lib.DeviceCount()
	if err == nil && count == 0 {
		err = errNVMLNoGPU
	}
	if err != nil {
		lib.Shutdown()
		return nil, err
	}
	return &nvmlGPUSource{lib: lib, count: count, lastSeen: make([]uint64, count)}, nil
}

// Close releases NVML.
func (s *nvmlGPUSource) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lib.Shutdown()
}

// StaticInfo returns the GPU count and each GPU's total memory in GiB, 0 when it is unknown.
func (s *nvmlGPUSource) StaticInfo() (int, []float64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	totals := make([]float64, s.count)
	for index := range totals {
		if total, _, err := s.lib.MemoryInfo(index); err == nil {
			totals[index] = float64(total) / bytesPerGB
		}
	}
	return s.count, totals, nil
}

// Usage returns the utilization and memory use of each GPU that reports both.
func (s *nvmlGPUSource) Usage() []GPUUsage {
	s.mu.Lock()
	defer s.mu.Unlock()
	usage := make([]GPUUsage, 0, s.count)
	for index := range s.count {
		utilization, err := s.lib.Utilization(index)
		if err != nil {
			continue
		}
		_, used, err := s.lib.MemoryInfo(index)
		if err != nil {
			continue
		}
		usage = append(usage, GPUUsage{Index: index, Utilization: utilization, MemUsedGB: float64(used) / bytesPerGB})
	}
	return usage
}

// Processes maps the PID of each process running on a GPU to its GPU, its average SM utilization
// since the previous call, and the GPU memory it uses.
func (s *nvmlGPUSource) Processes() map[int32]GPUProcessInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	processes := map[int32]GPUProcessInfo{}
	for index := range s.count {
		running, err := s.lib.RunningProcesses(index)
		if err != nil {
			running = nil
		}
		samples, err := s.lib.ProcessUtilization(index, s.lastSeen[index])
		if err != nil {
			samples = nil
		}
		for _, process := range running {
			processes[int32(process.PID)] = GPUProcessInfo{ //nolint:gosec // Linux PIDs fit in int32.
				GPUIndex:  index,
				UsedMemMB: process.UsedBytes / bytesPerMB,
			}
		}
		for pid, utilization := range s.averageUtilization(index, samples) {
			info := processes[pid]
			info.GPUIndex = index
			info.GPUUtil = utilization
			processes[pid] = info
		}
	}
	return processes
}

// averageUtilization averages each process's samples and remembers the newest timestamp, so the
// next call only gets newer samples.
func (s *nvmlGPUSource) averageUtilization(index int, samples []nvmlProcessSample) map[int32]uint64 {
	type total struct{ sum, count uint64 }
	totals := map[int32]total{}
	for _, sample := range samples {
		if sample.TimeStamp <= s.lastSeen[index] {
			continue
		}
		pid := int32(sample.PID) //nolint:gosec // Linux PIDs fit in int32.
		entry := totals[pid]
		entry.sum += uint64(sample.SMUtil)
		entry.count++
		totals[pid] = entry
	}
	for _, sample := range samples {
		s.lastSeen[index] = max(s.lastSeen[index], sample.TimeStamp)
	}
	averages := make(map[int32]uint64, len(totals))
	for pid, entry := range totals {
		averages[pid] = entry.sum / entry.count
	}
	return averages
}
