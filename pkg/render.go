package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/rivo/tview"
)

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
	resourceBars := make([]BarData, 0, 2+len(state.dynamic.StorageUsage)+len(state.dynamic.LiveGPUUsage)*barsPerGPU)
	cpuLabel, cpuInfo, cpuUsage := calculateCPULabelInfo(state)
	resourceBars = append(resourceBars, newBarData(cpuLabel, cpuUsage, cpuInfo))
	memLabel, memInfo, memPercent := calculateMEMLabelInfo(state)
	resourceBars = append(resourceBars, newBarData(memLabel, memPercent, memInfo))
	resourceBars = append(resourceBars, calculateStorageBars(state)...)
	if state.static.GPUCount > 0 {
		resourceBars = append(resourceBars, calculateGPUBars(state)...)
	}

	var builder strings.Builder
	layout := calculateResourceLayout(availableWidth, resourceBars)
	for _, row := range makeAlignedMultiColumnBars(resourceBars, layout) {
		builder.WriteString(row)
		builder.WriteByte('\n')
	}
	builder.WriteString(buildMetricsSection(state, availableWidth))

	return builder.String()
}

func calculateResourceLayout(availableWidth int, bars []BarData) BarLayout {
	if availableWidth < 0 {
		availableWidth = 0
	}
	maxLabelWidth, maxInfoWidth := calculateMaxWidthsFromBars(bars)
	if maxLabelWidth > maxMetricLabelWidth {
		maxLabelWidth = maxMetricLabelWidth
	}

	columns := 1
	if len(bars) > 1 && availableWidth >= minMetricColumnWidth*2+columnSpacing {
		columns = 2
	}
	spacingWidth := (columns - 1) * columnSpacing
	columnWidth := 0
	if availableWidth > spacingWidth {
		columnWidth = (availableWidth - spacingWidth) / columns
	}

	maxLabelThatFits := columnWidth - metricPercentWidth -
		metricFieldSpacing*metricLeadingFieldGaps - compactBarWidth
	if maxLabelWidth > maxLabelThatFits {
		maxLabelWidth = max(0, maxLabelThatFits)
	}

	remainingWidth := columnWidth - maxLabelWidth - metricPercentWidth -
		metricFieldSpacing*metricLeadingFieldGaps
	barWidth := min(maxBarWidth, max(0, remainingWidth))
	renderedInfoWidth := 0
	availableInfoWidth := remainingWidth - metricFieldSpacing - minBarWidth
	if maxInfoWidth > 0 && availableInfoWidth >= minMetricInfoWidth {
		renderedInfoWidth = min(maxInfoWidth, availableInfoWidth)
		barWidth = min(maxBarWidth, remainingWidth-metricFieldSpacing-renderedInfoWidth)
	}

	return BarLayout{
		Columns:       columns,
		BarWidth:      max(0, barWidth),
		TotalWidth:    columnWidth*columns + spacingWidth,
		MaxLabelWidth: maxLabelWidth,
		MaxInfoWidth:  renderedInfoWidth,
	}
}

// calculateCPULabelInfo calculates the CPU label, info text, and usage percentage.
func calculateCPULabelInfo(state *State) (string, string, float64) {
	var usage float64
	var label, info string
	if state.static.ContainerCPULimit == float64(state.static.HostCores) {
		// Running outside container or no cgroup limit - use host metrics
		usage = state.dynamic.HostCPUUsage
		label = metricCPULabel
		info = fmt.Sprintf("%d cores, no limit", state.static.HostCores)
	} else {
		// Running inside container with limits
		usage = state.dynamic.ContainerCPUUsage
		label = metricCPULabel
		info = fmt.Sprintf("%.2f CPU limit", state.static.ContainerCPULimit)
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
		label = metricMemoryLabel
		info = fmt.Sprintf(
			"%.2f / %.2f GB, no limit",
			state.dynamic.HostMemUsedGB,
			state.static.HostMemTotalGB,
		)
	} else {
		// Running inside container with limits
		percent = (state.dynamic.ContainerMemUsedGB * bytesPerGB / float64(state.static.ContainerMemLimitBytes)) * percentMultiplier
		label = metricMemoryLabel
		info = fmt.Sprintf(
			"%.2f / %.2f GB",
			state.dynamic.ContainerMemUsedGB,
			state.static.ContainerMemLimitGB,
		)
	}
	return label, info, percent
}

// calculateBarWidth calculates optimal bar width given available width and label/info text arrays.
// This is a general function that can be reused across different sections.
func calculateBarWidth(availableWidth int, labels []string, infos []string) int {
	count := max(len(labels), len(infos))
	bars := make([]BarData, 0, count)
	for index := range count {
		label := ""
		if index < len(labels) {
			label = labels[index]
		}
		info := ""
		if index < len(infos) {
			info = infos[index]
		}
		bars = append(bars, newBarData(label, 0, info))
	}
	return calculateResourceLayout(availableWidth, bars).BarWidth
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

		label := fmt.Sprintf("DISK %s", displayPath)
		info := fmt.Sprintf("%.2f / %.2f GB", storage.UsedGB, storage.UsedGB+storage.FreeGB)

		bars = append(bars, newBarData(label, storage.UsedPercent, info))
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
		utilLabel := fmt.Sprintf("GPU%d UTIL", i)
		bars = append(bars, newBarData(utilLabel, float64(gpu.Utilization), ""))

		// GPU Memory - Format as: "GPU0 Mem:" with right-aligned percentage
		gpuMemPercent := 0.0
		if len(state.static.GPUTotalGB) > i && state.static.GPUTotalGB[i] > 0 {
			gpuMemPercent = (gpu.MemUsedGB / state.static.GPUTotalGB[i]) * percentMultiplier
		}
		memLabel := fmt.Sprintf("GPU%d MEM", i)

		var memInfo string
		if len(state.static.GPUTotalGB) > i {
			memInfo = fmt.Sprintf("%.2f / %.2f GB", gpu.MemUsedGB, state.static.GPUTotalGB[i])
		}

		bars = append(bars, newBarData(memLabel, gpuMemPercent, memInfo))
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
	count := min(len(labels), len(infos), len(percentages))
	bars := make([]BarData, 0, count)
	for i := range count {
		bars = append(bars, newBarData(labels[i], percentages[i], infos[i]))
	}
	return bars
}

func buildStorageSection(layout BarLayout, labels []string, infos []string, percentages []float64) string {
	return buildBarSection(layout, slicesToBars(labels, infos, percentages))
}

func buildStorageSectionBars(layout BarLayout, bars []BarData) string {
	return buildBarSection(layout, bars)
}

func buildGPUSection(layout BarLayout, labels []string, infos []string, percentages []float64) string {
	return buildBarSection(layout, slicesToBars(labels, infos, percentages))
}

func buildBarSection(layout BarLayout, bars []BarData) string {
	var builder strings.Builder
	builder.WriteString("\n")

	if len(bars) == 0 {
		return builder.String()
	}

	for _, row := range makeAlignedMultiColumnBars(bars, layout) {
		builder.WriteString(row)
		builder.WriteByte('\n')
	}

	return builder.String()
}

func buildMetricsSection(state *State, availableWidth int) string {
	hasActivity := hasActivityMetrics(&state.dynamic)
	hasHistory := state.history.CPU.Next > 0 || state.history.CPU.Filled
	if !hasActivity && !hasHistory && len(state.dynamic.Alerts) == 0 {
		return ""
	}

	var builder strings.Builder
	writeAlertMetrics(&builder, state.dynamic.Alerts, availableWidth)
	if hasActivity {
		writeActivityMetrics(&builder, state, availableWidth)
	}
	if hasHistory {
		writeHistoryMetrics(&builder, state, availableWidth)
	}
	return builder.String()
}

func hasActivityMetrics(dynamic *DynamicInfo) bool {
	return len(dynamic.NetworkUsage) > 0 ||
		len(dynamic.DiskIOUsage) > 0 ||
		dynamic.PIDUsage.Current > 0 ||
		len(dynamic.Pressure) > 0
}

func writeAlertMetrics(builder *strings.Builder, alerts []Alert, availableWidth int) {
	for _, alert := range alerts {
		writeMetricLine(
			builder,
			"ALERT "+strings.ToUpper(alert.Level),
			alert.Message,
			"",
			availableWidth,
			"[red]",
		)
	}
}

func writeActivityMetrics(builder *strings.Builder, state *State, availableWidth int) {
	writePIDMetric(builder, state.dynamic.PIDUsage, availableWidth)
	if len(state.dynamic.NetworkUsage) > 0 {
		network := state.dynamic.NetworkUsage[0]
		writeMetricLine(
			builder,
			"NET "+network.Name,
			throughputText(
				"RX",
				network.RXBytesPerSec/bytesPerSecondToMiBSecond,
				"TX",
				network.TXBytesPerSec/bytesPerSecondToMiBSecond,
			),
			sparkline(state.history.Network),
			availableWidth,
			themeAccentTag,
		)
	}
	if len(state.dynamic.DiskIOUsage) > 0 {
		disk := state.dynamic.DiskIOUsage[0]
		writeMetricLine(
			builder,
			"IO "+disk.Name,
			throughputText(
				"R",
				disk.ReadBytesPerSec/bytesPerSecondToMiBSecond,
				"W",
				disk.WriteBytesPerSec/bytesPerSecondToMiBSecond,
			),
			sparkline(state.history.DiskIO),
			availableWidth,
			themeAccentTag,
		)
	}
	if len(state.dynamic.Pressure) > 0 {
		writeMetricLine(
			builder,
			"PSI SOME/FULL",
			pressureText(state.dynamic.Pressure),
			"",
			availableWidth,
			themeAccentTag,
		)
	}
}

func writePIDMetric(builder *strings.Builder, pid PIDUsage, availableWidth int) {
	if pid.Current == 0 {
		return
	}
	maxText := pid.MaxText
	if maxText == "" && pid.Max > 0 {
		maxText = strconv.FormatUint(pid.Max, 10)
	}
	if maxText == "" {
		maxText = "unknown"
	}
	writeMetricLine(
		builder,
		"PIDS",
		fmt.Sprintf("%d / %s", pid.Current, maxText),
		"",
		availableWidth,
		themeAccentTag,
	)
}

func writeHistoryMetrics(builder *strings.Builder, state *State, availableWidth int) {
	writeHistoryLine(builder, "HIST "+metricCPULabel, state.history.CPU, availableWidth)
	writeHistoryLine(builder, "HIST "+metricMemoryLabel, state.history.Memory, availableWidth)
	if state.static.GPUCount > 0 {
		writeHistoryLine(builder, "HIST GPU", state.history.GPU, availableWidth)
	}
}

func writeHistoryLine(builder *strings.Builder, label string, ring HistoryRing, availableWidth int) {
	writeMetricLine(builder, label, "", sparkline(ring), availableWidth, themeAccentTag)
}

func writeMetricLine(
	builder *strings.Builder,
	label string,
	value string,
	trend string,
	availableWidth int,
	valueTag string,
) {
	if availableWidth <= 0 {
		return
	}
	labelWidth := min(activityLabelWidth, max(0, availableWidth-metricFieldSpacing))
	label = truncatePlainText(label, labelWidth)
	valueWidthAvailable := max(0, availableWidth-labelWidth-metricFieldSpacing)
	value = truncatePlainText(value, valueWidthAvailable)

	trendWidth := valueWidthAvailable - visibleRuneCount(value)
	if value != "" && trend != "" {
		trendWidth -= metricFieldSpacing
	}
	if trendWidth < minSparklineWidth {
		trend = ""
	} else {
		trend = trimSparkline(trend, min(historySize, trendWidth))
	}

	builder.WriteString(themeLabelTag)
	builder.WriteString(padPlainText(label, labelWidth))
	builder.WriteString(themeResetTag)
	builder.WriteString(strings.Repeat(" ", metricFieldSpacing))
	if value != "" {
		builder.WriteString(valueTag)
		builder.WriteString(value)
		builder.WriteString(themeResetTag)
	}
	if trend != "" {
		if value != "" {
			builder.WriteString(strings.Repeat(" ", metricFieldSpacing))
		}
		builder.WriteString(themeAccentTag)
		builder.WriteString(trend)
		builder.WriteString(themeResetTag)
	}
	builder.WriteByte('\n')
}

func throughputText(firstLabel string, first float64, secondLabel string, second float64) string {
	return fmt.Sprintf("%-2s %7.2f  %-2s %7.2f MiB/s", firstLabel, first, secondLabel, second)
}

func pressureText(pressure []PressureStat) string {
	var builder strings.Builder
	for index, stat := range pressure {
		if index > 0 {
			builder.WriteString("  ")
		}
		fmt.Fprintf(
			&builder,
			"%s %.1f/%.1f",
			strings.ToUpper(stat.Resource),
			stat.SomeAvg10,
			stat.FullAvg10,
		)
	}
	return builder.String()
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
	if layout.Columns <= 0 {
		layout.Columns = 1
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
	spacing := "  │  "

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
	label := padPlainText(truncatePlainText(bar.Label, maxLabelWidth), maxLabelWidth)
	info := padPlainText(truncatePlainText(bar.Info, maxInfoWidth), maxInfoWidth)

	var builder strings.Builder
	builder.WriteString(themeLabelTag)
	builder.WriteString(label)
	builder.WriteString(themeResetTag)
	builder.WriteString(strings.Repeat(" ", metricFieldSpacing))
	builder.WriteString(formatUsagePercent(bar.Percent))
	builder.WriteString(strings.Repeat(" ", metricFieldSpacing))
	builder.WriteString(makeBar(bar.Percent, barWidth))
	if maxInfoWidth > 0 {
		builder.WriteString(strings.Repeat(" ", metricFieldSpacing))
		builder.WriteString(themeMutedTag)
		builder.WriteString(info)
		builder.WriteString(themeResetTag)
	}

	content := builder.String()
	contentWidth := tview.TaggedStringWidth(content)
	if columnWidth > contentWidth {
		content += strings.Repeat(" ", columnWidth-contentWidth)
	}
	return content
}

func formatUsagePercent(percent float64) string {
	value := fmt.Sprintf("%5.1f%%", min(max(percent, 0), usageDisplayMaximum))
	if percent > usageDisplayMaximum {
		value = " 999+%"
	}
	return "[" + usageColorName(percent) + "]" + value + themeResetTag
}

func truncatePlainText(value string, width int) string {
	if width <= 0 {
		return ""
	}
	runes := []rune(value)
	if len(runes) <= width {
		return value
	}
	if width == 1 {
		return "…"
	}
	return string(runes[:width-1]) + "…"
}

func padPlainText(value string, width int) string {
	padding := width - visibleRuneCount(value)
	if padding <= 0 {
		return value
	}
	return value + strings.Repeat(" ", padding)
}

func visibleRuneCount(value string) int {
	return len([]rune(value))
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

	var builder strings.Builder
	colorName := usageColorName(percent)
	builder.Grow(len(colorName) + 2 + len(themeTrackTag) + len(themeResetTag) + barWidth*len("━"))
	builder.WriteByte('[')
	builder.WriteString(colorName)
	builder.WriteByte(']')
	for range filledWidth {
		builder.WriteString("━")
	}
	builder.WriteString(themeTrackTag)
	for range barWidth - filledWidth {
		builder.WriteString("─")
	}
	builder.WriteString(themeResetTag)
	return builder.String()
}
