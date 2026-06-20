package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

const onceTextWidth = 120

type ExportSnapshot struct {
	Timestamp time.Time         `json:"timestamp"`
	Static    StaticSnapshot    `json:"static"`
	Dynamic   DynamicSnapshot   `json:"dynamic"`
	History   ExportHistory     `json:"history"`
	Metadata  ContainerMetadata `json:"metadata,omitempty"`
}

type StaticSnapshot struct {
	CgroupVersion          CgroupVersion       `json:"cgroupVersion"`
	ContainerCPULimit      float64             `json:"containerCpuLimit"`
	ContainerMemLimitBytes int64               `json:"containerMemLimitBytes"`
	ContainerMemLimitGB    float64             `json:"containerMemLimitGb"`
	HostCores              int                 `json:"hostCores"`
	HostMemTotalGB         float64             `json:"hostMemTotalGb"`
	GPUCount               int                 `json:"gpuCount"`
	GPUTotalGB             []float64           `json:"gpuTotalGb,omitempty"`
	StorageMounts          []StorageMount      `json:"storageMounts,omitempty"`
	Integrations           []IntegrationStatus `json:"integrations,omitempty"`
}

type ExportHistory struct {
	CPU     []float64 `json:"cpu,omitempty"`
	Memory  []float64 `json:"memory,omitempty"`
	GPU     []float64 `json:"gpu,omitempty"`
	Network []float64 `json:"network,omitempty"`
	DiskIO  []float64 `json:"diskIo,omitempty"`
}

func snapshotFromState(state *State) ExportSnapshot {
	state.dynamic.mu.Lock()
	dynamic := DynamicSnapshot{
		ContainerCPUUsage:  state.dynamic.ContainerCPUUsage,
		ContainerMemUsedGB: state.dynamic.ContainerMemUsedGB,
		HostCPUUsage:       state.dynamic.HostCPUUsage,
		HostMemUsedGB:      state.dynamic.HostMemUsedGB,
		LiveGPUUsage:       append([]GPUUsage(nil), state.dynamic.LiveGPUUsage...),
		StorageUsage:       append([]StorageUsage(nil), state.dynamic.StorageUsage...),
		NetworkUsage:       append([]NetworkUsage(nil), state.dynamic.NetworkUsage...),
		DiskIOUsage:        append([]DiskIOUsage(nil), state.dynamic.DiskIOUsage...),
		CgroupEvents:       state.dynamic.CgroupEvents,
		PIDUsage:           state.dynamic.PIDUsage,
		Pressure:           append([]PressureStat(nil), state.dynamic.Pressure...),
		Alerts:             append([]Alert(nil), state.dynamic.Alerts...),
		Processes:          append([]ProcessInfo(nil), state.dynamic.Processes...),
	}
	history := ExportHistory{
		CPU:     (&state.history.CPU).Ordered(),
		Memory:  (&state.history.Memory).Ordered(),
		GPU:     (&state.history.GPU).Ordered(),
		Network: (&state.history.Network).Ordered(),
		DiskIO:  (&state.history.DiskIO).Ordered(),
	}
	state.dynamic.mu.Unlock()

	staticInfo := state.static
	return ExportSnapshot{
		Timestamp: time.Now(),
		Static: StaticSnapshot{
			CgroupVersion:          staticInfo.CgroupVersion,
			ContainerCPULimit:      staticInfo.ContainerCPULimit,
			ContainerMemLimitBytes: staticInfo.ContainerMemLimitBytes,
			ContainerMemLimitGB:    staticInfo.ContainerMemLimitGB,
			HostCores:              staticInfo.HostCores,
			HostMemTotalGB:         staticInfo.HostMemTotalGB,
			GPUCount:               staticInfo.GPUCount,
			GPUTotalGB:             append([]float64(nil), staticInfo.GPUTotalGB...),
			StorageMounts:          append([]StorageMount(nil), staticInfo.StorageMounts...),
			Integrations:           append([]IntegrationStatus(nil), staticInfo.Integrations...),
		},
		Dynamic:  dynamic,
		History:  history,
		Metadata: staticInfo.Metadata,
	}
}

func writeJSONSnapshot(writer io.Writer, state *State) error {
	encoder := json.NewEncoder(writer)
	encoder.SetIndent("", "  ")
	//nolint:musttag // ExportSnapshot contains exported nested app structs intentionally encoded by field name.
	return encoder.Encode(snapshotFromState(state))
}

func onceText(state *State) string {
	var builder strings.Builder
	builder.WriteString(buildResourceText(onceTextWidth, state))
	snapshot := snapshotFromState(state)
	builder.WriteString("\nProcesses:\n")
	for _, process := range snapshot.Dynamic.Processes {
		fmt.Fprintf(
			&builder,
			"%d %.1f%% %.1f%% %s\n",
			process.PID,
			process.CPUPercent,
			process.MemPercent,
			process.Command,
		)
	}
	return builder.String()
}
