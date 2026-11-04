package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"time"

	"github.com/shirou/gopsutil/v4/cpu"
	"github.com/shirou/gopsutil/v4/disk"
	"github.com/shirou/gopsutil/v4/mem"
	netio "github.com/shirou/gopsutil/v4/net"
	"github.com/shirou/gopsutil/v4/process"
)

// The OS* types below are thin wrappers around stdlib and gopsutil calls.
// They exist solely to enable dependency injection and mocking in tests.
// Direct forwarding is intentional — the wrappers provide seam points.

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
	Context context.Context // Stops running commands on shutdown; defaults to context.Background
	Timeout time.Duration
}

func (ocr OSCommandRunner) Output(name string, arg ...string) ([]byte, error) {
	timeout := ocr.Timeout
	if timeout <= 0 {
		timeout = commandTimeout
	}
	parent := ocr.Context
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, timeout)
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
	ParentPID() (int32, error)
	Times() (*cpu.TimesStat, error)
	MemoryInfo() (*process.MemoryInfoStat, error)
	Username() (string, error)
	Cmdline() (string, error)
	Name() (string, error)
	CreateTime() (int64, error)
	NumThreads() (int32, error)
}

// ProcessProvider lists processes from the operating system.
type ProcessProvider interface {
	Processes() ([]ProcessHandle, error)
}

// errProcessChanged reports that a PID no longer belongs to the expected process.
var errProcessChanged = errors.New("process is no longer running")

// ProcessSignaler sends signals to operating-system processes.
type ProcessSignaler interface {
	// Signal signals the process only if it is still the one that started at startTime.
	Signal(pid int32, startTime time.Time, signal os.Signal) error
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

type NetworkProvider interface {
	IOCounters(pernic bool) ([]netio.IOCountersStat, error)
}

type DiskIOProvider interface {
	IOCounters(names ...string) (map[string]disk.IOCountersStat, error)
}

// OSProcessSignaler sends a signal to a process by PID.
type OSProcessSignaler struct{}

// Signal opens the process first. On Linux the handle is a pidfd, so the start time check
// below applies to the exact process that receives the signal, even if the PID is reused.
func (s OSProcessSignaler) Signal(pid int32, startTime time.Time, signal os.Signal) error {
	proc, err := os.FindProcess(int(pid))
	if err != nil {
		return errProcessChanged
	}
	defer func() { _ = proc.Release() }()

	createTime, err := readProcessStartTime(procRoot, pid)
	if err != nil || startTime.IsZero() || createTime != startTime.UnixMilli() {
		return errProcessChanged
	}
	if err = proc.Signal(signal); errors.Is(err, os.ErrProcessDone) {
		return errProcessChanged
	}
	return err
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

type OSNetworkProvider struct{}

func (p OSNetworkProvider) IOCounters(pernic bool) ([]netio.IOCountersStat, error) {
	return netio.IOCounters(pernic)
}

type OSDiskIOProvider struct{}

func (p OSDiskIOProvider) IOCounters(names ...string) (map[string]disk.IOCountersStat, error) {
	return readDiskStats(procRoot, names...)
}
