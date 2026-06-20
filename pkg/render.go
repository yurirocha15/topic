package main

import (
	"fmt"
	"strconv"
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
	builder.WriteString(buildMetricsSection(state, availableWidth))

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

func buildMetricsSection(state *State, availableWidth int) string {
	if len(state.dynamic.NetworkUsage) == 0 &&
		len(state.dynamic.DiskIOUsage) == 0 &&
		state.dynamic.PIDUsage.Current == 0 &&
		len(state.dynamic.Pressure) == 0 &&
		len(state.dynamic.Alerts) == 0 {
		return ""
	}

	var builder strings.Builder
	builder.WriteString("\n")
	if len(state.dynamic.Alerts) > 0 {
		for _, alert := range state.dynamic.Alerts {
			builder.WriteString("[red]ALERT[white] ")
			builder.WriteString(alert.Level)
			builder.WriteString(": ")
			builder.WriteString(alert.Message)
			builder.WriteString("\n")
		}
	}
	if pid := state.dynamic.PIDUsage; pid.Current > 0 {
		maxText := pid.MaxText
		if maxText == "" && pid.Max > 0 {
			maxText = strconv.FormatUint(pid.Max, 10)
		}
		if maxText == "" {
			maxText = "unknown"
		}
		fmt.Fprintf(&builder, "PIDS: [yellow]%d / %s[white]\n", pid.Current, maxText)
	}
	if len(state.dynamic.NetworkUsage) > 0 {
		network := state.dynamic.NetworkUsage[0]
		const networkSparklinePrefixWidth = 36
		fmt.Fprintf(
			&builder,
			"NET %s: [yellow]RX %.2f MiB/s TX %.2f MiB/s[white] %s\n",
			network.Name,
			network.RXBytesPerSec/bytesPerSecondToMiBSecond,
			network.TXBytesPerSec/bytesPerSecondToMiBSecond,
			sparklineForWidth(state.history.Network, availableWidth, networkSparklinePrefixWidth),
		)
	}
	if len(state.dynamic.DiskIOUsage) > 0 {
		disk := state.dynamic.DiskIOUsage[0]
		const diskIOSparklinePrefixWidth = 32
		fmt.Fprintf(
			&builder,
			"IO %s: [yellow]R %.2f MiB/s W %.2f MiB/s[white] %s\n",
			disk.Name,
			disk.ReadBytesPerSec/bytesPerSecondToMiBSecond,
			disk.WriteBytesPerSec/bytesPerSecondToMiBSecond,
			sparklineForWidth(state.history.DiskIO, availableWidth, diskIOSparklinePrefixWidth),
		)
	}
	if len(state.dynamic.Pressure) > 0 {
		builder.WriteString("PSI:")
		for _, pressure := range state.dynamic.Pressure {
			fmt.Fprintf(
				&builder,
				" %s some %.1f full %.1f",
				pressure.Resource,
				pressure.SomeAvg10,
				pressure.FullAvg10,
			)
		}
		builder.WriteString("\n")
	}
	if state.history.CPU.Next > 0 || state.history.CPU.Filled {
		writeHistoryLine(&builder, "CPU", state.history.CPU, availableWidth)
		writeHistoryLine(&builder, "MEM", state.history.Memory, availableWidth)
		writeHistoryLine(&builder, "GPU", state.history.GPU, availableWidth)
	}
	return builder.String()
}

func writeHistoryLine(builder *strings.Builder, label string, ring HistoryRing, availableWidth int) {
	fmt.Fprintf(builder, "HIST %s %s\n", label, sparklineForWidth(ring, availableWidth, len("HIST XXX ")))
}

func sparklineForWidth(ring HistoryRing, availableWidth int, prefixWidth int) string {
	width := availableWidth - prefixWidth
	if width <= 0 {
		return ""
	}
	if width > historySize {
		width = historySize
	}
	if width < minSparklineWidth {
		if availableWidth < prefixWidth+minSparklineWidth {
			return ""
		}
		width = minSparklineWidth
	}
	return trimSparkline(sparkline(ring), width)
}

func trimSparkline(value string, width int) string {
	runes := []rune(value)
	if len(runes) <= width {
		return value
	}
	return string(runes[len(runes)-width:])
}

// makeAlignedMultiColumnBars creates properly aligned bars in columns with consistent spacing.
func makeAlignedMultiColumnBars(bars []BarData, layout BarLayout) []string {
	if len(bars) == 0 {
		return nil
	}
	numRows := (len(bars) + layout.Columns - 1) / layout.Columns

	// Use the unified max widths from layout for consistent alignment across sections
	maxLabelWidth := layout.MaxLabelWidth
	maxInfoWidth := layout.MaxInfoWidth

	// Use the unified bar width from layout without recalculation
	// This ensures consistent bar width across different sections (DISK/GPU)
	actualBarWidth := layout.BarWidth

	result := make([]string, 0, numRows)
	for row := range numRows {
		rowContent := buildAlignedRow(bars, BarLayout{
			Columns:    layout.Columns,
			BarWidth:   actualBarWidth,
			TotalWidth: layout.TotalWidth,
		}, row, maxLabelWidth, maxInfoWidth)
		result = append(result, rowContent)
	}

	return result
}

// calculateMaxWidthsFromSlices calculates max widths from separate label and info slices for unified alignment.
func calculateMaxWidthsFromSlices(labels []string, infos []string) (int, int) {
	maxLabelWidth := 0
	maxInfoWidth := 0

	for _, label := range labels {
		labelWidth := tview.TaggedStringWidth(label)
		if labelWidth > maxLabelWidth {
			maxLabelWidth = labelWidth
		}
	}

	for _, info := range infos {
		infoWidth := tview.TaggedStringWidth(info)
		if infoWidth > maxInfoWidth {
			maxInfoWidth = infoWidth
		}
	}

	return maxLabelWidth, maxInfoWidth
}

func calculateMaxWidthsFromBars(bars []BarData) (int, int) {
	maxLabelWidth := 0
	maxInfoWidth := 0

	for _, bar := range bars {
		labelWidth := barLabelWidth(bar)
		if labelWidth > maxLabelWidth {
			maxLabelWidth = labelWidth
		}
		infoWidth := barInfoWidth(bar)
		if infoWidth > maxInfoWidth {
			maxInfoWidth = infoWidth
		}
	}

	return maxLabelWidth, maxInfoWidth
}

func barLabelWidth(bar BarData) int {
	if bar.LabelWidth > 0 {
		return bar.LabelWidth
	}
	return tview.TaggedStringWidth(bar.Label)
}

func barInfoWidth(bar BarData) int {
	if bar.InfoWidth > 0 || bar.Info == "" {
		return bar.InfoWidth
	}
	return tview.TaggedStringWidth(bar.Info)
}

func buildAlignedRow(
	bars []BarData,
	layout BarLayout,
	row int,
	maxLabelWidth int,
	maxInfoWidth int,
) string {
	// Calculate the width each column should occupy
	spacingWidth := (layout.Columns - 1) * columnSpacing
	columnWidth := (layout.TotalWidth - spacingWidth) / layout.Columns
	spacing := strings.Repeat(" ", columnSpacing)

	var builder strings.Builder

	for col := range layout.Columns {
		if col > 0 {
			builder.WriteString(spacing)
		}

		barIndex := row*layout.Columns + col
		if barIndex >= len(bars) {
			// Fill empty columns with spaces to maintain alignment
			builder.WriteString(strings.Repeat(" ", columnWidth))
			continue
		}

		builder.WriteString(formatAlignedBar(
			bars[barIndex],
			layout.BarWidth,
			maxLabelWidth,
			maxInfoWidth,
			columnWidth,
		))
	}

	return builder.String()
}

func formatAlignedBar(bar BarData, barWidth int, maxLabelWidth int, maxInfoWidth int, columnWidth int) string {
	// Pad label to consistent width for alignment
	paddedLabel := bar.Label
	labelPadding := maxLabelWidth - barLabelWidth(bar)
	if labelPadding > 0 {
		paddedLabel += strings.Repeat(" ", labelPadding)
	}

	// Create the bar with consistent width
	barContent := makeBar(bar.Percent, barWidth)

	// Pad info to consistent width (if info exists)
	if bar.Info != "" {
		paddedInfo := bar.Info
		infoPadding := maxInfoWidth - barInfoWidth(bar)
		if infoPadding > 0 {
			paddedInfo += strings.Repeat(" ", infoPadding)
		}
		// Complete bar: label + space + bar + space + info
		content := paddedLabel + " " + barContent + " " + paddedInfo

		// Pad the entire content to fill the column width
		contentWidth := tview.TaggedStringWidth(content)
		if columnWidth > contentWidth {
			content += strings.Repeat(" ", columnWidth-contentWidth)
		}
		return content
	}

	// No info, just label + bar
	content := paddedLabel + " " + barContent

	// Pad to fill the column width
	contentWidth := tview.TaggedStringWidth(content)
	if columnWidth > contentWidth {
		content += strings.Repeat(" ", columnWidth-contentWidth)
	}
	return content
}

// makeBar creates a visual bar representation of a percentage.
func makeBar(percent float64, barWidth int) string {
	// Ensure barWidth is not negative.
	if barWidth < 0 {
		barWidth = 0
	}

	filledWidth := int((float64(barWidth) * percent) / percentMultiplier)
	if filledWidth > barWidth {
		filledWidth = barWidth
	}
	if filledWidth < 0 {
		filledWidth = 0
	}

	// Use characters that provide better visual separation between bars
	var builder strings.Builder
	builder.Grow(len("[green]") + len("[white]") + barWidth*len("▓"))
	builder.WriteString("[green]")
	for range filledWidth {
		builder.WriteString("▓")
	}
	for range barWidth - filledWidth {
		builder.WriteString("░")
	}
	builder.WriteString("[white]")
	return builder.String()
}
