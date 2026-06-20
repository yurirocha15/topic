package main

import (
	"fmt"
	"strings"

	"github.com/rivo/tview"
)

// updateResourceView updates the current usage statistics panel.
func updateResourceView(view *tview.TextView, state *State) int {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()

	// Get the available width inside the view, minus padding.
	_, _, availableWidth, _ := view.GetInnerRect()
	availableWidth -= borderHeight // Account for horizontal padding within the box

	finalText := buildResourceText(availableWidth, state)
	view.SetText(finalText)
	return strings.Count(finalText, "\n")
}

func buildResourceText(availableWidth int, state *State) string {
	var builder strings.Builder

	cpuLabel, cpuInfo, cpuUsage := calculateCPULabelInfo(state)
	memLabel, memInfo, memPercent := calculateMEMLabelInfo(state)
	cpuMemBarWidth := calculateBarWidth(availableWidth, []string{cpuLabel, memLabel}, []string{cpuInfo, memInfo})
	builder.WriteString(buildCPUSection(cpuMemBarWidth, cpuLabel, cpuInfo, cpuUsage))
	builder.WriteString(buildMemorySection(cpuMemBarWidth, memLabel, memInfo, memPercent))

	// Calculate unified bar width for DISK and GPU sections.
	var storageBars []BarData
	var gpuBars []BarData
	var diskGPUBars []BarData

	if len(state.dynamic.StorageUsage) > 0 {
		storageBars = calculateStorageBars(state)
		diskGPUBars = append(diskGPUBars, storageBars...)
	}
	if state.static.GPUCount > 0 {
		gpuBars = calculateGPUBars(state)
		diskGPUBars = append(diskGPUBars, gpuBars...)
	}

	// Determine if DISK/GPU should use multi-column layout
	var sharedLayout BarLayout
	if len(diskGPUBars) > 0 {
		// Calculate unified max widths from all combined labels and infos for proper alignment
		maxLabelWidth, maxInfoWidth := calculateMaxWidthsFromBars(diskGPUBars)

		// Determine the layout for all DISK/GPU entries combined
		totalEntries := len(diskGPUBars)
		maxColumns := maxDiskColumns
		if totalEntries == 1 {
			maxColumns = 1
		}

		// Force single column if terminal is too narrow for multi-column
		if availableWidth < minTotalWidthForMultiCol || maxColumns == 1 {
			// Single column layout
			sharedLayout = BarLayout{
				Columns:       1,
				BarWidth:      calculateBarWidthFromBars(availableWidth, diskGPUBars),
				TotalWidth:    availableWidth,
				MaxLabelWidth: maxLabelWidth,
				MaxInfoWidth:  maxInfoWidth,
			}
		} else {
			// Multi-column layout - calculate based on actual content dimensions
			spacingWidth := (maxColumns - 1) * columnSpacing
			availableContentWidth := availableWidth - spacingWidth

			// Calculate actual space required per column
			actualSpacePerColumn := maxLabelWidth + spacesAroundBar + minBarWidth + spacesAroundBar + maxInfoWidth

			// Check if we have enough space for the requested columns
			if availableContentWidth < actualSpacePerColumn*maxColumns {
				// Not enough space for multi-column, force single column
				sharedLayout = BarLayout{
					Columns:       1,
					BarWidth:      calculateBarWidthFromBars(availableWidth, diskGPUBars),
					TotalWidth:    availableWidth,
					MaxLabelWidth: maxLabelWidth,
					MaxInfoWidth:  maxInfoWidth,
				}
			} else {
				// We can fit multi-column layout
				widthPerColumn := availableContentWidth / maxColumns
				barWidth := calculateBarWidthFromBars(widthPerColumn, diskGPUBars)
				totalWidth := maxColumns*widthPerColumn + spacingWidth

				sharedLayout = BarLayout{
					Columns:       maxColumns,
					BarWidth:      barWidth,
					TotalWidth:    totalWidth,
					MaxLabelWidth: maxLabelWidth,
					MaxInfoWidth:  maxInfoWidth,
				}
			}
		}
	}

	// Build sections with unified layout and bar width
	if len(storageBars) > 0 {
		builder.WriteString(buildStorageSectionBars(sharedLayout, storageBars))
	}
	if len(gpuBars) > 0 {
		builder.WriteString(buildGPUSectionBars(sharedLayout, gpuBars))
	}

	return builder.String()
}

// calculateCPULabelInfo calculates the CPU label, info text, and usage percentage.
func calculateCPULabelInfo(state *State) (string, string, float64) {
	var usage float64
	var label, info string
	if state.static.ContainerCPULimit == float64(state.static.HostCores) {
		// Running outside container or no cgroup limit - use host metrics
		usage = state.dynamic.HostCPUUsage
		label = fmt.Sprintf("CPU: [yellow]%-6.1f%%[white] ", usage)
		info = fmt.Sprintf(" [darkcyan](no cgroup limit, %d host cores)[white]", state.static.HostCores)
	} else {
		// Running inside container with limits
		usage = state.dynamic.ContainerCPUUsage
		label = fmt.Sprintf("CPU: [yellow]%-6.1f%%[white] ", usage)
		info = fmt.Sprintf(" [darkcyan](limit: %.2f CPUs)[white]", state.static.ContainerCPULimit)
	}
	return label, info, usage
}

// calculateMEMLabelInfo calculates the MEM label, info text, and percentage.
func calculateMEMLabelInfo(state *State) (string, string, float64) {
	var percent float64
	var label, info string
	if state.static.ContainerMemLimitGB == 0 {
		// Running outside container or no memory limit - use host metrics
		if state.static.HostMemTotalGB > 0 {
			percent = (state.dynamic.HostMemUsedGB / state.static.HostMemTotalGB) * percentMultiplier
		}
		label = fmt.Sprintf("MEM: [yellow]%-6.1f%%[white] ", percent)
		info = fmt.Sprintf(
			" [darkcyan]%.3f GB / %.3f GB (no cgroup limit)[white]",
			state.dynamic.HostMemUsedGB,
			state.static.HostMemTotalGB,
		)
	} else {
		// Running inside container with limits
		percent = (state.dynamic.ContainerMemUsedGB * bytesPerGB / float64(state.static.ContainerMemLimitBytes)) * percentMultiplier
		label = fmt.Sprintf("MEM: [yellow]%-6.1f%%[white] ", percent)
		info = fmt.Sprintf(
			" [darkcyan]%.3f GB / %.3f GB[white]",
			state.dynamic.ContainerMemUsedGB,
			state.static.ContainerMemLimitGB,
		)
	}
	return label, info, percent
}

// calculateBarWidth calculates optimal bar width given available width and label/info text arrays.
// This is a general function that can be reused across different sections.
func calculateBarWidth(availableWidth int, labels []string, infos []string) int {
	// Find maximum label width
	maxLabelWidth := 0
	for _, label := range labels {
		if width := tview.TaggedStringWidth(label); width > maxLabelWidth {
			maxLabelWidth = width
		}
	}

	// Find maximum info width
	maxInfoWidth := 0
	for _, info := range infos {
		if width := tview.TaggedStringWidth(info); width > maxInfoWidth {
			maxInfoWidth = width
		}
	}

	// Calculate consistent bar width
	barWidth := availableWidth - maxLabelWidth - maxInfoWidth
	if barWidth < minBarWidth {
		barWidth = minBarWidth
	}

	return barWidth
}

func calculateBarWidthFromBars(availableWidth int, bars []BarData) int {
	maxLabelWidth, maxInfoWidth := calculateMaxWidthsFromBars(bars)
	barWidth := availableWidth - maxLabelWidth - maxInfoWidth
	if barWidth < minBarWidth {
		barWidth = minBarWidth
	}
	return barWidth
}

func newBarData(label string, percent float64, info string) BarData {
	return BarData{
		Label:      label,
		LabelWidth: tview.TaggedStringWidth(label),
		Percent:    percent,
		Info:       info,
		InfoWidth:  tview.TaggedStringWidth(info),
	}
}

// calculateStorageLabelsInfo calculates all storage labels, info texts, and percentages.
func calculateStorageLabelsInfo(state *State) ([]string, []string, []float64) {
	return barsToSlices(calculateStorageBars(state))
}

func calculateStorageBars(state *State) []BarData {
	bars := make([]BarData, 0, len(state.dynamic.StorageUsage))
	for _, storage := range state.dynamic.StorageUsage {
		// Shorten long mount paths for display
		displayPath := storage.Path
		if len(displayPath) > maxDisplayPathLength {
			displayPath = "..." + displayPath[len(displayPath)-12:]
		}

		// Format as: "DISK /:" with right-aligned percentage
		label := fmt.Sprintf("DISK %s:", displayPath)
		percentage := fmt.Sprintf("%5.1f%%", storage.UsedPercent)
		formattedLabel := fmt.Sprintf("%-*s[yellow]%s[white]", minLabelWidth, label, percentage)

		info := fmt.Sprintf("[darkcyan]%.2f GB / %.2f GB[white]", storage.UsedGB, storage.UsedGB+storage.FreeGB)

		bars = append(bars, newBarData(formattedLabel, storage.UsedPercent, info))
	}
	return bars
}

// calculateGPULabelsInfo calculates all GPU labels, info texts, and percentages.
func calculateGPULabelsInfo(state *State) ([]string, []string, []float64) {
	return barsToSlices(calculateGPUBars(state))
}

func calculateGPUBars(state *State) []BarData {
	bars := make([]BarData, 0, len(state.dynamic.LiveGPUUsage)*barsPerGPU)
	for i, gpu := range state.dynamic.LiveGPUUsage {
		// GPU Utilization - Format as: "GPU0 Util:" with right-aligned percentage
		utilLabel := fmt.Sprintf("GPU%d Util:", i)
		utilPercentage := fmt.Sprintf("%5d%%", gpu.Utilization)
		formattedUtilLabel := fmt.Sprintf("%-*s[yellow]%s[white]", minLabelWidth, utilLabel, utilPercentage)

		bars = append(bars, newBarData(formattedUtilLabel, float64(gpu.Utilization), ""))

		// GPU Memory - Format as: "GPU0 Mem:" with right-aligned percentage
		gpuMemPercent := 0.0
		if len(state.static.GPUTotalGB) > i && state.static.GPUTotalGB[i] > 0 {
			gpuMemPercent = (gpu.MemUsedGB / state.static.GPUTotalGB[i]) * percentMultiplier
		}
		memLabel := fmt.Sprintf("GPU%d Mem:", i)
		memPercentage := fmt.Sprintf("%5.0f%%", gpuMemPercent)
		formattedMemLabel := fmt.Sprintf("%-*s[yellow]%s[white]", minLabelWidth, memLabel, memPercentage)

		var memInfo string
		if len(state.static.GPUTotalGB) > i {
			memInfo = fmt.Sprintf("[darkcyan]%.2f GB / %.2f GB[white]", gpu.MemUsedGB, state.static.GPUTotalGB[i])
		}

		bars = append(bars, newBarData(formattedMemLabel, gpuMemPercent, memInfo))
	}
	return bars
}

func barsToSlices(bars []BarData) ([]string, []string, []float64) {
	labels := make([]string, 0, len(bars))
	infos := make([]string, 0, len(bars))
	percentages := make([]float64, 0, len(bars))
	for _, bar := range bars {
		labels = append(labels, bar.Label)
		infos = append(infos, bar.Info)
		percentages = append(percentages, bar.Percent)
	}
	return labels, infos, percentages
}

func slicesToBars(labels []string, infos []string, percentages []float64) []BarData {
	bars := make([]BarData, 0, len(labels))
	for i := range labels {
		bars = append(bars, newBarData(labels[i], percentages[i], infos[i]))
	}
	return bars
}

// buildCPUSection creates the CPU usage display section.
func buildCPUSection(barWidth int, cpuLabel string, cpuInfo string, cpuUsage float64) string {
	return cpuLabel + makeBar(cpuUsage, barWidth) + cpuInfo + "\n"
}

// buildMemorySection creates the memory usage display section.
func buildMemorySection(barWidth int, memLabel string, memInfo string, memPercent float64) string {
	return memLabel + makeBar(memPercent, barWidth) + memInfo + "\n"
}

// buildStorageSection creates the storage usage display section.
func buildStorageSection(layout BarLayout, labels []string, infos []string, percentages []float64) string {
	return buildBarSection(layout, slicesToBars(labels, infos, percentages))
}

func buildStorageSectionBars(layout BarLayout, bars []BarData) string {
	return buildBarSection(layout, bars)
}

// buildGPUSection creates the GPU usage display section.
func buildGPUSection(layout BarLayout, labels []string, infos []string, percentages []float64) string {
	return buildBarSection(layout, slicesToBars(labels, infos, percentages))
}

func buildGPUSectionBars(layout BarLayout, bars []BarData) string {
	return buildBarSection(layout, bars)
}

func buildBarSection(layout BarLayout, bars []BarData) string {
	var builder strings.Builder
	builder.WriteString("\n")

	if len(bars) == 0 {
		return builder.String()
	}

	// Use the pre-calculated layout from updateResourceView
	if layout.Columns == 1 {
		for _, bar := range bars {
			builder.WriteString(bar.Label + " " + makeBar(bar.Percent, layout.BarWidth) + " " + bar.Info + "\n")
		}
	} else {
		barRows := makeAlignedMultiColumnBars(bars, layout)
		for _, row := range barRows {
			builder.WriteString(row + "\n")
		}
	}

	return builder.String()
}
