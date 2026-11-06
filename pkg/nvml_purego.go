//go:build nvml

package main

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

// nvmlLibraryName is the versioned name the NVIDIA driver installs and the NVIDIA container
// runtime mounts into containers.
const nvmlLibraryName = "libnvidia-ml.so.1"

// NVML return codes topic handles.
const (
	nvmlSuccess          nvmlReturn = 0
	nvmlErrorNotFound    nvmlReturn = 6
	nvmlInsufficientSize nvmlReturn = 7
)

// nvmlValueNotAvailable is what NVML reports for a process's memory when the driver cannot tell.
const nvmlValueNotAvailable = ^uint64(0)

// nvmlInitialBufferLength is the number of processes or samples the buffers start with. They grow
// when NVML asks for more room.
const nvmlInitialBufferLength = 64

type nvmlReturn int32

// nvmlError is a failed NVML call.
type nvmlError struct {
	code nvmlReturn
}

func (e nvmlError) Error() string {
	return fmt.Sprintf("NVML error %d", int32(e.code))
}

func (code nvmlReturn) err() error {
	if code == nvmlSuccess {
		return nil
	}
	return nvmlError{code: code}
}

// The structs below match NVML's C layout on amd64 and arm64.

type nvmlMemoryC struct {
	Total, Free, Used uint64
}

// nvmlMemoryV2C is nvmlMemory_v2_t. Its Used leaves out memory the driver reserves, matching
// nvidia-smi's memory.used.
type nvmlMemoryV2C struct {
	Version                     uint32
	Total, Reserved, Free, Used uint64
}

// nvmlMemoryV2Version is NVML_STRUCT_VERSION(Memory, 2): the struct size with the version in the
// top byte.
const nvmlMemoryV2Version = uint32(unsafe.Sizeof(nvmlMemoryV2C{})) | 2<<24

type nvmlUtilizationC struct {
	GPU, Memory uint32
}

type nvmlProcessUtilizationSampleC struct {
	PID                               uint32
	TimeStamp                         uint64
	SMUtil, MemUtil, EncUtil, DecUtil uint32
}

// nvmlProcessInfoC is nvmlProcessInfo_t as the _v2 and _v3 process queries fill it.
type nvmlProcessInfoC struct {
	PID                              uint32
	UsedGPUMemory                    uint64
	GPUInstanceID, ComputeInstanceID uint32
}

type nvmlDevice uintptr

type nvmlProcessQuery func(device nvmlDevice, count *uint32, infos *nvmlProcessInfoC) nvmlReturn

// puregoNVML calls libnvidia-ml without cgo. Callers must not use it concurrently; nvmlGPUSource
// holds a lock around every call.
type puregoNVML struct {
	devices []nvmlDevice // nil after Shutdown

	shutdown           func() nvmlReturn
	memoryInfo         func(device nvmlDevice, memory *nvmlMemoryC) nvmlReturn
	memoryInfoV2       func(device nvmlDevice, memory *nvmlMemoryV2C) nvmlReturn // nil before R510
	utilizationRates   func(device nvmlDevice, utilization *nvmlUtilizationC) nvmlReturn
	processUtilization func(
		device nvmlDevice, samples *nvmlProcessUtilizationSampleC, count *uint32, lastSeen uint64,
	) nvmlReturn
	computeProcesses  nvmlProcessQuery
	graphicsProcesses nvmlProcessQuery

	samples   []nvmlProcessUtilizationSampleC
	processes []nvmlProcessInfoC
}

func openNVML() (nvmlLibrary, error) {
	library, err := purego.Dlopen(nvmlLibraryName, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, err
	}
	// The library stays loaded until topic exits; unloading driver libraries is not reliably safe.
	return loadNVML(library)
}

func loadNVML(library uintptr) (*puregoNVML, error) {
	nvml := &puregoNVML{
		samples:   make([]nvmlProcessUtilizationSampleC, nvmlInitialBufferLength),
		processes: make([]nvmlProcessInfoC, nvmlInitialBufferLength),
	}
	var (
		initialize  func() nvmlReturn
		deviceCount func(count *uint32) nvmlReturn
		deviceAt    func(index uint32, device *nvmlDevice) nvmlReturn
	)
	err := errors.Join(
		bindNVML(library, &initialize, "nvmlInit_v2"),
		bindNVML(library, &nvml.shutdown, "nvmlShutdown"),
		bindNVML(library, &deviceCount, "nvmlDeviceGetCount_v2"),
		bindNVML(library, &deviceAt, "nvmlDeviceGetHandleByIndex_v2"),
		bindNVML(library, &nvml.memoryInfo, "nvmlDeviceGetMemoryInfo"),
		bindNVML(library, &nvml.utilizationRates, "nvmlDeviceGetUtilizationRates"),
		bindNVML(library, &nvml.processUtilization, "nvmlDeviceGetProcessUtilization"),
		// Drivers before R510 have only the _v2 process queries, which fill the same struct.
		bindNVML(library, &nvml.computeProcesses,
			"nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2"),
		bindNVML(library, &nvml.graphicsProcesses,
			"nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses_v2"),
	)
	if err != nil {
		return nil, err
	}
	if bindNVML(library, &nvml.memoryInfoV2, "nvmlDeviceGetMemoryInfo_v2") != nil {
		nvml.memoryInfoV2 = nil
	}
	if err = initialize().err(); err != nil {
		return nil, fmt.Errorf("initialize NVML: %w", err)
	}
	var count uint32
	if err = deviceCount(&count).err(); err != nil {
		_ = nvml.shutdown()
		return nil, fmt.Errorf("count GPUs: %w", err)
	}
	nvml.devices = make([]nvmlDevice, count)
	for index := range nvml.devices {
		// A GPU without a handle stays 0, and every query for it fails.
		_ = deviceAt(uint32(index), &nvml.devices[index])
	}
	return nvml, nil
}

// bindNVML points function at the first of names the library exports.
func bindNVML(library uintptr, function any, names ...string) error {
	for _, name := range names {
		if address, err := purego.Dlsym(library, name); err == nil {
			purego.RegisterFunc(function, address)
			return nil
		}
	}
	return fmt.Errorf("NVML has no %s", names[0])
}

func (nvml *puregoNVML) device(index int) (nvmlDevice, error) {
	if index < 0 || index >= len(nvml.devices) || nvml.devices[index] == 0 {
		return 0, fmt.Errorf("no NVML handle for GPU %d", index)
	}
	return nvml.devices[index], nil
}

func (nvml *puregoNVML) DeviceCount() (int, error) {
	return len(nvml.devices), nil
}

func (nvml *puregoNVML) MemoryInfo(index int) (uint64, uint64, error) {
	device, err := nvml.device(index)
	if err != nil {
		return 0, 0, err
	}
	if nvml.memoryInfoV2 != nil {
		memory := nvmlMemoryV2C{Version: nvmlMemoryV2Version}
		if nvml.memoryInfoV2(device, &memory) == nvmlSuccess {
			return memory.Total, memory.Used, nil
		}
	}
	var memory nvmlMemoryC
	if err = nvml.memoryInfo(device, &memory).err(); err != nil {
		return 0, 0, err
	}
	return memory.Total, memory.Used, nil
}

func (nvml *puregoNVML) Utilization(index int) (int, error) {
	device, err := nvml.device(index)
	if err != nil {
		return 0, err
	}
	var utilization nvmlUtilizationC
	if err = nvml.utilizationRates(device, &utilization).err(); err != nil {
		return 0, err
	}
	return int(utilization.GPU), nil
}

func (nvml *puregoNVML) ProcessUtilization(index int, since uint64) ([]nvmlProcessSample, error) {
	device, err := nvml.device(index)
	if err != nil {
		return nil, err
	}
	count := uint32(len(nvml.samples)) //nolint:gosec // The buffer length fits in uint32.
	code := nvml.processUtilization(device, &nvml.samples[0], &count, since)
	if code == nvmlInsufficientSize {
		nvml.samples = make([]nvmlProcessUtilizationSampleC, count+nvmlInitialBufferLength)
		count = uint32(len(nvml.samples)) //nolint:gosec // The buffer length fits in uint32.
		code = nvml.processUtilization(device, &nvml.samples[0], &count, since)
	}
	if code == nvmlErrorNotFound {
		return nil, nil // no samples since the last call
	}
	if err = code.err(); err != nil {
		return nil, err
	}
	samples := make([]nvmlProcessSample, 0, count)
	for _, sample := range nvml.samples[:min(int(count), len(nvml.samples))] {
		samples = append(samples, nvmlProcessSample{
			PID:       sample.PID,
			SMUtil:    sample.SMUtil,
			TimeStamp: sample.TimeStamp,
		})
	}
	return samples, nil
}

// RunningProcesses lists compute and graphics processes. A process that is both is listed once.
func (nvml *puregoNVML) RunningProcesses(index int) ([]nvmlProcessMemory, error) {
	device, err := nvml.device(index)
	if err != nil {
		return nil, err
	}
	compute, computeErr := nvml.runningProcesses(nvml.computeProcesses, device)
	graphics, graphicsErr := nvml.runningProcesses(nvml.graphicsProcesses, device)
	if computeErr != nil && graphicsErr != nil {
		return nil, errors.Join(computeErr, graphicsErr)
	}
	processes := compute
	listed := make(map[uint32]bool, len(compute))
	for _, process := range compute {
		listed[process.PID] = true
	}
	for _, process := range graphics {
		if !listed[process.PID] {
			processes = append(processes, process)
		}
	}
	return processes, nil
}

func (nvml *puregoNVML) runningProcesses(query nvmlProcessQuery, device nvmlDevice) ([]nvmlProcessMemory, error) {
	count := uint32(len(nvml.processes)) //nolint:gosec // The buffer length fits in uint32.
	code := query(device, &count, &nvml.processes[0])
	if code == nvmlInsufficientSize {
		nvml.processes = make([]nvmlProcessInfoC, count+nvmlInitialBufferLength)
		count = uint32(len(nvml.processes)) //nolint:gosec // The buffer length fits in uint32.
		code = query(device, &count, &nvml.processes[0])
	}
	if err := code.err(); err != nil {
		return nil, err
	}
	processes := make([]nvmlProcessMemory, 0, count)
	for _, process := range nvml.processes[:min(int(count), len(nvml.processes))] {
		used := process.UsedGPUMemory
		if used == nvmlValueNotAvailable {
			used = 0
		}
		processes = append(processes, nvmlProcessMemory{PID: process.PID, UsedBytes: used})
	}
	return processes, nil
}

func (nvml *puregoNVML) Shutdown() {
	if nvml.devices == nil {
		return
	}
	_ = nvml.shutdown()
	nvml.devices = nil
}
