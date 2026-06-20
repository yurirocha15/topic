package main

import (
	"sort"
	"strconv"
	"strings"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

const (
	processPIDColumn = iota
	processUserColumn
	processCPUColumn
	processMemColumn
	processGPUColumn
	processGPUMemColumn
	processCommandColumn
)

func updateProcessTable(table *tview.Table, state *State) {
	state.dynamic.mu.Lock()
	ui := state.ui
	totalProcesses := len(state.dynamic.Processes)
	processes := state.dynamic.Processes
	unlockAfterRender := true

	if needsProcessTransform(ui) {
		processes = append([]ProcessInfo(nil), state.dynamic.Processes...)
		state.dynamic.mu.Unlock()
		unlockAfterRender = false
		processes = prepareProcessRows(processes, ui)
	}
	if unlockAfterRender {
		defer state.dynamic.mu.Unlock()
	}

	// --- Create Header ---
	headers := processTableHeaders(ui)
	for i, header := range headers {
		setTableCell(table, 0, i, header, tcell.ColorYellow, false, 0)
	}

	// --- Populate Data ---
	for r, p := range processes {
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

	for row := table.GetRowCount() - 1; row > len(processes); row-- {
		table.RemoveRow(row)
	}

	table.SetTitle(processTableTitle(ui, len(processes), totalProcesses))
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

func needsProcessTransform(ui UIState) bool {
	return ui.ProcessFilter != "" || ui.ReverseSort || ui.ProcessSort != SortByCPU
}

func prepareProcessRows(processes []ProcessInfo, ui UIState) []ProcessInfo {
	if ui.ProcessFilter != "" {
		processes = filterProcesses(processes, ui.ProcessFilter)
	}
	sortProcesses(processes, ui.ProcessSort, ui.ReverseSort)
	return processes
}

func filterProcesses(processes []ProcessInfo, filter string) []ProcessInfo {
	filter = strings.ToLower(strings.TrimSpace(filter))
	if filter == "" {
		return processes
	}

	filtered := make([]ProcessInfo, 0, len(processes))
	for _, process := range processes {
		if processMatchesFilter(process, filter) {
			filtered = append(filtered, process)
		}
	}
	return filtered
}

func processMatchesFilter(process ProcessInfo, filter string) bool {
	return strings.Contains(strings.ToLower(process.Command), filter) ||
		strings.Contains(strings.ToLower(process.User), filter) ||
		strings.Contains(strconv.Itoa(int(process.PID)), filter)
}

func sortProcesses(processes []ProcessInfo, sortColumn ProcessSortColumn, reverse bool) {
	sort.SliceStable(processes, func(i, j int) bool {
		less := compareProcesses(processes[i], processes[j], sortColumn)
		if reverse {
			return !less
		}
		return less
	})
}

func compareProcesses(left ProcessInfo, right ProcessInfo, sortColumn ProcessSortColumn) bool {
	switch sortColumn {
	case SortByMemory:
		return left.MemPercent > right.MemPercent
	case SortByGPU:
		return left.GPUUtil > right.GPUUtil
	case SortByGPUMemory:
		return left.GPUMemPercent > right.GPUMemPercent
	case SortByPID:
		return left.PID < right.PID
	case SortByUser:
		return left.User < right.User
	case SortByCommand:
		return left.Command < right.Command
	case SortByCPU:
		fallthrough
	default:
		return left.CPUPercent > right.CPUPercent
	}
}

func processTableHeaders(ui UIState) []string {
	headers := []string{"PID", "USER", "%CPU", "%MEM", "%GPU", "%GPUMEM", "COMMAND"}
	sortIndex := processSortHeaderIndex(ui.ProcessSort)
	if sortIndex >= 0 && sortIndex < len(headers) {
		marker := "↓"
		if ui.ReverseSort {
			marker = "↑"
		}
		headers[sortIndex] += marker
	}
	return headers
}

func processSortHeaderIndex(sortColumn ProcessSortColumn) int {
	switch sortColumn {
	case SortByMemory:
		return processMemColumn
	case SortByGPU:
		return processGPUColumn
	case SortByGPUMemory:
		return processGPUMemColumn
	case SortByPID:
		return processPIDColumn
	case SortByUser:
		return processUserColumn
	case SortByCommand:
		return processCommandColumn
	case SortByCPU:
		fallthrough
	default:
		return processCPUColumn
	}
}

func processTableTitle(ui UIState, visibleCount int, totalCount int) string {
	parts := []string{" Processes"}
	if ui.Paused {
		parts = append(parts, "[paused]")
	}
	if ui.TreeMode {
		parts = append(parts, "[tree]")
	}
	if ui.ProcessFilter != "" {
		parts = append(parts, "[filter: "+ui.ProcessFilter+"]")
	}
	if visibleCount != totalCount {
		parts = append(parts, "["+strconv.Itoa(visibleCount)+"/"+strconv.Itoa(totalCount)+"]")
	}
	if ui.SearchMode {
		parts = append(parts, "[typing]")
	}
	return strings.Join(parts, " ") + " "
}

func nextProcessSortColumn(current ProcessSortColumn) ProcessSortColumn {
	switch current {
	case SortByCPU:
		return SortByMemory
	case SortByMemory:
		return SortByGPU
	case SortByGPU:
		return SortByGPUMemory
	case SortByGPUMemory:
		return SortByPID
	case SortByPID:
		return SortByUser
	case SortByUser:
		return SortByCommand
	case SortByCommand:
		fallthrough
	default:
		return SortByCPU
	}
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
