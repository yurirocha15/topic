package main

import (
	"context"
	"os"
	"os/exec"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

// FileReader defines an interface for reading files.
type FileReader interface {
	ReadFile(path string) ([]byte, error)
}

// OSFileReader is the default implementation of FileReader using the os package.
type OSFileReader struct{}

func (fs OSFileReader) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

// CommandRunner defines an interface for running external commands, allowing for mocking.
type CommandRunner interface {
	Output(name string, arg ...string) ([]byte, error)
}

// OSCommandRunner is the default implementation using the os/exec package.
type OSCommandRunner struct {
	Timeout time.Duration
}

func (ocr OSCommandRunner) Output(name string, arg ...string) ([]byte, error) {
	timeout := ocr.Timeout
	if timeout <= 0 {
		timeout = commandTimeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return exec.CommandContext(ctx, name, arg...).Output()
}

// Stater defines an interface for checking file status.
type Stater interface {
	Stat(path string) (os.FileInfo, error)
}

// OSStater is the default implementation of Stater using the os package.
type OSStater struct{}

func (s OSStater) Stat(path string) (os.FileInfo, error) {
	return os.Stat(path)
}

// ProcessHandle exposes the process data needed by the process table.
type ProcessHandle interface {
	PID() int32
	CPUPercent() (float64, error)
	MemoryInfo() (*process.MemoryInfoStat, error)
	Username() (string, error)
	Cmdline() (string, error)
	Name() (string, error)
}

// ProcessProvider lists processes from the operating system.
type ProcessProvider interface {
	Processes() ([]ProcessHandle, error)
}

// HostMetricsProvider exposes host CPU and memory metrics.
type HostMetricsProvider interface {
	CPUCounts(logical bool) (int, error)
	CPUPercent(interval time.Duration, perCPU bool) ([]float64, error)
	VirtualMemory() (*mem.VirtualMemoryStat, error)
}

// StorageProvider exposes mounted filesystem metadata and usage.
type StorageProvider interface {
	Partitions(all bool) ([]disk.PartitionStat, error)
	Usage(path string) (*disk.UsageStat, error)
}

// OSProcessProvider reads process information through gopsutil.
type OSProcessProvider struct{}

func (p OSProcessProvider) Processes() ([]ProcessHandle, error) {
	procs, err := process.Processes()
	if err != nil {
		return nil, err
	}
	handles := make([]ProcessHandle, 0, len(procs))
	for _, proc := range procs {
		handles = append(handles, gopsutilProcessHandle{Process: proc})
	}
	return handles, nil
}

type gopsutilProcessHandle struct {
	*process.Process
}

func (p gopsutilProcessHandle) PID() int32 {
	return p.Pid
}

// OSHostMetricsProvider reads host metrics through gopsutil.
type OSHostMetricsProvider struct{}

func (p OSHostMetricsProvider) CPUCounts(logical bool) (int, error) {
	return cpu.Counts(logical)
}

func (p OSHostMetricsProvider) CPUPercent(interval time.Duration, perCPU bool) ([]float64, error) {
	return cpu.Percent(interval, perCPU)
}

func (p OSHostMetricsProvider) VirtualMemory() (*mem.VirtualMemoryStat, error) {
	return mem.VirtualMemory()
}

// OSStorageProvider reads storage metadata through gopsutil.
type OSStorageProvider struct{}

func (p OSStorageProvider) Partitions(all bool) ([]disk.PartitionStat, error) {
	return disk.Partitions(all)
}

func (p OSStorageProvider) Usage(path string) (*disk.UsageStat, error) {
	return disk.Usage(path)
}
