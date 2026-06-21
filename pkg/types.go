package main

import (
	"sync"
	"time"
)

// CgroupVersion denotes the cgroup version (1 or 2).
type CgroupVersion int

const (
	CgroupV1 CgroupVersion = 1
	CgroupV2 CgroupVersion = 2
)

// StaticInfo holds information that is fetched once at startup.
type StaticInfo struct {
	CgroupVersion          CgroupVersion
	ContainerCPULimit      float64
	ContainerMemLimitBytes int64
	ContainerMemLimitGB    float64
	HostCores              int
	HostMemTotalGB         float64
	GPUCount               int
	GPUTotalGB             []float64
	StorageMounts          []StorageMount
	Metadata               ContainerMetadata
	Integrations           []IntegrationStatus
}

type AppConfig struct {
	RefreshInterval   time.Duration
	DisableGPU        bool
	DisableDocker     bool
	DisableKubernetes bool
	DisableNVML       bool
	Once              bool
	JSONOutput        bool
	HideASCIIArt      bool
	InitialSort       ProcessSortColumn
}

type ContainerMetadata struct {
	Runtime   string            `json:"runtime,omitempty"`
	ID        string            `json:"id,omitempty"`
	Name      string            `json:"name,omitempty"`
	Image     string            `json:"image,omitempty"`
	Namespace string            `json:"namespace,omitempty"`
	Pod       string            `json:"pod,omitempty"`
	Node      string            `json:"node,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// MergeFrom copies non-empty fields from other into m.
func (m *ContainerMetadata) MergeFrom(other ContainerMetadata) {
	if other.Runtime != "" {
		m.Runtime = other.Runtime
	}
	if other.ID != "" {
		m.ID = other.ID
	}
	if other.Name != "" {
		m.Name = other.Name
	}
	if other.Image != "" {
		m.Image = other.Image
	}
	if other.Namespace != "" {
		m.Namespace = other.Namespace
	}
	if other.Pod != "" {
		m.Pod = other.Pod
	}
	if other.Node != "" {
		m.Node = other.Node
	}
	if len(other.Labels) > 0 {
		m.Labels = other.Labels
	}
}

type IntegrationStatus struct {
	Name      string `json:"name"`
	Available bool   `json:"available"`
	Detail    string `json:"detail,omitempty"`
}

// GPUUsage holds live usage data for a single GPU.
type GPUUsage struct {
	Index       int
	Utilization int
	MemUsedGB   float64
}

// StorageMount holds information about a mounted filesystem.
type StorageMount struct {
	Path    string
	TotalGB float64
	Fstype  string
}

// StorageUsage holds current usage data for a mounted filesystem.
type StorageUsage struct {
	Path        string
	UsedGB      float64
	FreeGB      float64
	UsedPercent float64
}

type NetworkUsage struct {
	Name          string
	RXBytesPerSec float64
	TXBytesPerSec float64
	RXErrors      uint64
	TXErrors      uint64
}

type DiskIOUsage struct {
	Name             string
	ReadBytesPerSec  float64
	WriteBytesPerSec float64
	ReadOpsPerSec    float64
	WriteOpsPerSec   float64
}

type CgroupEvents struct {
	MemoryHigh          uint64
	MemoryOOM           uint64
	MemoryOOMKill       uint64
	CPUThrottledPeriods uint64
}

type PIDUsage struct {
	Current uint64
	Max     uint64
	MaxText string
}

type PressureStat struct {
	Resource  string
	SomeAvg10 float64
	FullAvg10 float64
}

type Alert struct {
	Level   string
	Message string
}

type HistoryRing struct {
	Values []float64
	Next   int
	Filled bool
}

type MetricsHistory struct {
	CPU     HistoryRing
	Memory  HistoryRing
	GPU     HistoryRing
	Network HistoryRing
	DiskIO  HistoryRing
}

// BarLayout holds information about how bars should be arranged.
type BarLayout struct {
	Columns       int
	BarWidth      int
	TotalWidth    int
	MaxLabelWidth int
	MaxInfoWidth  int
}

// BarData holds information for rendering a single bar.
type BarData struct {
	Label      string
	LabelWidth int
	Percent    float64
	Info       string
	InfoWidth  int
}

// GPUProcessInfo holds usage data for a process on a GPU.
type GPUProcessInfo struct {
	GPUIndex   int
	UsedMemMB  uint64
	GPUUtil    uint64
	GPUMemUtil uint64
}

// ProcessInfo holds information for a single monitored process.
type ProcessInfo struct {
	PID           int32
	ParentPID     int32
	User          string
	CPUPercent    float64
	MemPercent    float64
	Command       string
	StartTime     time.Time
	ThreadCount   int32
	OpenFileCount int
	rawCPU        float64
	GPUIndex      int
	GPUUtil       uint64
	GPUMemPercent float64
}

type ProcessSortColumn int

const (
	SortByCPU ProcessSortColumn = iota
	SortByMemory
	SortByGPU
	SortByGPUMemory
	SortByPID
	SortByUser
	SortByCommand
)

// UIState holds interactive view preferences.
type UIState struct {
	ProcessSort   ProcessSortColumn
	ReverseSort   bool
	ProcessFilter string
	SearchMode    bool
	SortMode      bool
	Paused        bool
	TreeMode      bool
	SelectedPID   int32
	HideASCIIArt  bool
}

// DynamicInfo holds all the data that is refreshed on each tick.
type DynamicInfo struct {
	mu                 sync.Mutex
	ContainerCPUUsage  float64
	ContainerMemUsedGB float64
	HostCPUUsage       float64
	HostMemUsedGB      float64
	LiveGPUUsage       []GPUUsage
	StorageUsage       []StorageUsage
	NetworkUsage       []NetworkUsage
	DiskIOUsage        []DiskIOUsage
	CgroupEvents       CgroupEvents
	PIDUsage           PIDUsage
	Pressure           []PressureStat
	Alerts             []Alert
	Processes          []ProcessInfo
}

type DynamicSnapshot struct {
	ContainerCPUUsage  float64
	ContainerMemUsedGB float64
	HostCPUUsage       float64
	HostMemUsedGB      float64
	LiveGPUUsage       []GPUUsage
	StorageUsage       []StorageUsage
	NetworkUsage       []NetworkUsage
	DiskIOUsage        []DiskIOUsage
	CgroupEvents       CgroupEvents
	PIDUsage           PIDUsage
	Pressure           []PressureStat
	Alerts             []Alert
	Processes          []ProcessInfo
}

type DynamicCollectorSet struct {
	ContainerCPU func() float64
	ContainerMem func() float64
	LiveGPU      func() []GPUUsage
	Storage      func() []StorageUsage
	Network      func() []NetworkUsage
	DiskIO       func() []DiskIOUsage
	CgroupEvents func() CgroupEvents
	PIDs         func() PIDUsage
	Pressure     func() []PressureStat
	Processes    func() []ProcessInfo
	HostCPU      func() float64
	HostMem      func() float64
}

// State holds the application's entire state.
// Fields are unexported to enforce snapshot-based access; use accessor
// methods to read values and the refresh functions to update them.
type State struct {
	static       StaticInfo
	dynamic      DynamicInfo
	ui           UIState
	prevCPUUsage uint64
	prevCPUTime  time.Time
	prevNetwork  map[string]NetworkCounter
	prevDiskIO   map[string]DiskIOCounter
	prevNetTime  time.Time
	prevDiskTime time.Time
	history      MetricsHistory
}

// Static returns a copy of the static configuration.
func (s *State) Static() StaticInfo {
	s.dynamic.mu.Lock()
	defer s.dynamic.mu.Unlock()
	return s.static
}

type NetworkCounter struct {
	RXBytes uint64
	TXBytes uint64
	RXErrs  uint64
	TXErrs  uint64
}

type DiskIOCounter struct {
	ReadBytes  uint64
	WriteBytes uint64
	ReadCount  uint64
	WriteCount uint64
}
