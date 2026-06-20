package main

import (
	"sort"
	"strconv"
	"strings"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

func updateProcessTable(table *tview.Table, state *State) {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()

	// --- Create Header ---
	headers := []string{"PID", "USER", "%CPU", "%MEM", "%GPU", "%GPUMEM", "COMMAND"}
	for i, header := range headers {
		setTableCell(table, 0, i, header, tcell.ColorYellow, false, 0)
	}

	// --- Populate Data ---
	for r, p := range state.dynamic.Processes {
		// PID
		columnIdx := 0
		setTableCell(table, r+1, columnIdx, strconv.Itoa(int(p.PID)), tcell.ColorWhite, true, 0)
		// USER
		columnIdx++
		setTableCell(table, r+1, columnIdx, p.User, tcell.ColorGreen, true, 0)
		// %CPU
		columnIdx++
		setTableCell(
			table,
			r+1,
			columnIdx,
			strconv.FormatFloat(p.CPUPercent, 'f', 1, 64),
			tcell.ColorAqua,
			true,
			0,
		)
		// %MEM
		columnIdx++
		setTableCell(
			table,
			r+1,
			columnIdx,
			strconv.FormatFloat(p.MemPercent, 'f', 1, 64),
			tcell.ColorAqua,
			true,
			0,
		)

		// %GPU and %GPUMEM
		columnIdx++
		if p.GPUIndex != -1 {
			setTableCell(
				table,
				r+1,
				columnIdx,
				strconv.FormatUint(p.GPUUtil, 10),
				tcell.ColorFuchsia,
				true,
				0,
			)
			columnIdx++
			setTableCell(
				table,
				r+1,
				columnIdx,
				strconv.FormatFloat(p.GPUMemPercent, 'f', 1, 64),
				tcell.ColorFuchsia,
				true,
				0,
			)
		} else {
			setTableCell(table, r+1, columnIdx, "-", tcell.ColorDarkGray, true, 0)
			columnIdx++
			setTableCell(table, r+1, columnIdx, "-", tcell.ColorDarkGray, true, 0)
		}

		// COMMAND
		columnIdx++
		setTableCell(table, r+1, columnIdx, p.Command, tcell.ColorWhite, true, 1)
	}

	for row := table.GetRowCount() - 1; row > len(state.dynamic.Processes); row-- {
		table.RemoveRow(row)
	}
}

func setTableCell(
	table *tview.Table,
	row int,
	column int,
	text string,
	color tcell.Color,
	selectable bool,
	expansion int,
) {
	cell := table.GetCell(row, column)
	cell.SetText(text).
		SetTextColor(color).
		SetAlign(tview.AlignLeft).
		SetSelectable(selectable).
		SetExpansion(expansion)
	table.SetCell(row, column, cell)
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

// updateProcessList fetches the current process list and adds resource usage info when possible.
func updateProcessList(static *StaticInfo, runner CommandRunner) []ProcessInfo {
	return updateProcessListWithProvider(static, runner, OSProcessProvider{})
}

func updateProcessListWithProvider(static *StaticInfo, runner CommandRunner, provider ProcessProvider) []ProcessInfo {
	var gpuProcessMap map[int32]GPUProcessInfo
	if static.GPUCount > 0 {
		gpuProcessMap = getGPUProcessMap(runner)
	}
	procs, err := provider.Processes()
	if err != nil {
		return nil
	}
	processList := make([]ProcessInfo, 0, len(procs))
	for _, p := range procs {
		pi, ok := getProcessInfo(p, static, gpuProcessMap)
		if !ok {
			continue
		}
		processList = append(processList, pi)
	}
	sort.Slice(processList, func(i, j int) bool {
		return processList[i].rawCPU > processList[j].rawCPU
	})
	return processList
}

func getProcessInfo(
	p ProcessHandle,
	static *StaticInfo,
	gpuProcessMap map[int32]GPUProcessInfo,
) (ProcessInfo, bool) {
	cpuPercent, err := p.CPUPercent()
	if err != nil {
		return ProcessInfo{}, false
	}
	memInfo, err := p.MemoryInfo()
	if err != nil {
		return ProcessInfo{}, false
	}
	user, err := p.Username()
	if err != nil {
		user = "n/a"
	}
	cmdline, err := p.Cmdline()
	if err != nil || cmdline == "" {
		name, nameErr := p.Name()
		if nameErr != nil {
			return ProcessInfo{}, false
		}
		cmdline = "[" + name + "]"
	}

	containerCPUPercent := 0.0
	if static.ContainerCPULimit > 0 {
		containerCPUPercent = cpuPercent / static.ContainerCPULimit
	}
	containerMemPercent := 0.0
	if static.ContainerMemLimitBytes > 0 {
		containerMemPercent = (float64(memInfo.RSS) / float64(static.ContainerMemLimitBytes)) * percentMultiplier
	}

	pi := ProcessInfo{
		PID:        p.PID(),
		User:       user,
		CPUPercent: containerCPUPercent,
		MemPercent: containerMemPercent,
		Command:    cmdline,
		rawCPU:     cpuPercent,
		GPUIndex:   -1,
	}
	if gpuInfo, onGPU := gpuProcessMap[p.PID()]; onGPU {
		pi.GPUIndex = gpuInfo.GPUIndex
		pi.GPUUtil = gpuInfo.GPUUtil
		pi.GPUMemPercent = float64(gpuInfo.GPUMemUtil)
	}
	return pi, true
}
