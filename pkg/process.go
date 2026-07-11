package main

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

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

const (
	processPIDLabel       = "PID"
	processUserLabel      = "USER"
	processCPULabel       = "%CPU"
	processMemoryLabel    = "%MEM"
	processGPULabel       = "%GPU"
	processGPUMemoryLabel = "%GPUMEM"
	processCommandLabel   = "COMMAND"
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
		color := tcell.ColorLightGray
		if i == processSortHeaderIndex(ui.ProcessSort) {
			color = tcell.ColorAqua
		}
		setTableCell(table, 0, i, header, color, false, 0)
	}

	// --- Populate Data ---
	for r, p := range processes {
		// PID
		columnIdx := 0
		setTableCell(table, r+1, columnIdx, strconv.Itoa(int(p.PID)), tcell.ColorGray, true, 0)
		// USER
		columnIdx++
		setTableCell(table, r+1, columnIdx, p.User, tcell.ColorLightGray, true, 0)
		// %CPU
		columnIdx++
		setTableCell(
			table,
			r+1,
			columnIdx,
			strconv.FormatFloat(p.CPUPercent, 'f', 1, 64),
			usageCellColor(p.CPUPercent),
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
			usageCellColor(p.MemPercent),
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
				usageCellColor(float64(p.GPUUtil)),
				true,
				0,
			)
			columnIdx++
			setTableCell(
				table,
				r+1,
				columnIdx,
				strconv.FormatFloat(p.GPUMemPercent, 'f', 1, 64),
				usageCellColor(p.GPUMemPercent),
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
	if ui.TreeMode {
		processes = buildProcessTreeRows(processes)
		return processes
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

func buildProcessTreeRows(processes []ProcessInfo) []ProcessInfo {
	if len(processes) == 0 {
		return nil
	}

	byParent := make(map[int32][]ProcessInfo, len(processes))
	byPID := make(map[int32]ProcessInfo, len(processes))
	for _, process := range processes {
		byPID[process.PID] = process
		byParent[process.ParentPID] = append(byParent[process.ParentPID], process)
	}

	for parentPID := range byParent {
		sortProcesses(byParent[parentPID], SortByCPU, false)
	}

	roots := make([]ProcessInfo, 0, len(processes))
	for _, process := range processes {
		if _, hasParent := byPID[process.ParentPID]; process.ParentPID == 0 || !hasParent {
			roots = append(roots, process)
		}
	}
	sortProcesses(roots, SortByCPU, false)

	visited := make(map[int32]bool, len(processes))
	rows := make([]ProcessInfo, 0, len(processes))
	var walk func(ProcessInfo, int)
	walk = func(process ProcessInfo, depth int) {
		if visited[process.PID] {
			return
		}
		visited[process.PID] = true
		process.Command = treeCommand(process.Command, depth)
		rows = append(rows, process)
		for _, child := range byParent[process.PID] {
			walk(child, depth+1)
		}
	}
	for _, root := range roots {
		walk(root, 0)
	}
	for _, process := range processes {
		if !visited[process.PID] {
			walk(process, 0)
		}
	}
	return rows
}

func treeCommand(command string, depth int) string {
	if depth <= 0 {
		return command
	}
	return strings.Repeat("  ", depth) + "└─ " + command
}

func processTableHeaders(ui UIState) []string {
	headers := []string{
		processPIDLabel,
		processUserLabel,
		processCPULabel,
		processMemoryLabel,
		processGPULabel,
		processGPUMemoryLabel,
		processCommandLabel,
	}
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

func selectedProcess(table *tview.Table, state *State) (ProcessInfo, bool) {
	row, _ := table.GetSelection()
	if row <= 0 {
		return ProcessInfo{}, false
	}

	state.dynamic.mu.Lock()
	processes := append([]ProcessInfo(nil), state.dynamic.Processes...)
	ui := state.ui
	state.dynamic.mu.Unlock()

	processes = prepareProcessRows(processes, ui)
	rowIndex := row - 1
	if rowIndex < 0 || rowIndex >= len(processes) {
		return ProcessInfo{}, false
	}
	return processes[rowIndex], true
}

func processDetailsText(process ProcessInfo) string {
	start := "unknown"
	if !process.StartTime.IsZero() {
		start = process.StartTime.Format(time.RFC3339)
	}
	gpu := "none"
	if process.GPUIndex >= 0 {
		gpu = fmt.Sprintf("GPU%d util=%d%% mem=%.1f%%", process.GPUIndex, process.GPUUtil, process.GPUMemPercent)
	}
	return fmt.Sprintf(
		"PID: %d\nParent PID: %d\nUser: %s\nCPU: %.1f%%\nMemory: %.1f%%\nGPU: %s\nStarted: %s\nThreads: %d\nOpen files: %d\n\nCommand:\n%s",
		process.PID,
		process.ParentPID,
		process.User,
		process.CPUPercent,
		process.MemPercent,
		gpu,
		start,
		process.ThreadCount,
		process.OpenFileCount,
		process.Command,
	)
}

func signalLabel(signal os.Signal) string {
	switch signal {
	case syscall.SIGKILL:
		return signalKillLabel
	case syscall.SIGINT:
		return signalIntLabel
	case syscall.SIGTERM:
		fallthrough
	default:
		return signalTermLabel
	}
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
		parts = append(parts, "[filter mode]")
	}
	if ui.SortMode {
		parts = append(parts, "[sort mode: ←/→ column ↑/↓ direction]")
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
		return SortByCommand
	case SortByPID:
		return SortByUser
	case SortByUser:
		return SortByCPU
	case SortByCommand:
		fallthrough
	default:
		return SortByPID
	}
}

func previousProcessSortColumn(current ProcessSortColumn) ProcessSortColumn {
	switch current {
	case SortByCPU:
		return SortByUser
	case SortByMemory:
		return SortByCPU
	case SortByGPU:
		return SortByMemory
	case SortByGPUMemory:
		return SortByGPU
	case SortByPID:
		return SortByCommand
	case SortByUser:
		return SortByPID
	case SortByCommand:
		fallthrough
	default:
		return SortByGPUMemory
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
		pi, processErr := getProcessInfo(p, static, gpuProcessMap)
		if processErr != nil {
			continue
		}
		processList = append(processList, pi)
	}
	sort.Slice(processList, func(i, j int) bool {
		return processList[i].rawCPU > processList[j].rawCPU
	})
	return processList
}

func resolveProcessCommand(p ProcessHandle) (string, error) {
	cmdline, err := p.Cmdline()
	if err == nil && cmdline != "" {
		return cmdline, nil
	}
	name, err := p.Name()
	if err != nil {
		return "", fmt.Errorf("process name: %w", err)
	}
	return "[" + name + "]", nil
}

func computeContainerPercents(cpuPercent float64, rssBytes uint64, static *StaticInfo) (float64, float64) {
	cpu := 0.0
	if static.ContainerCPULimit > 0 {
		cpu = cpuPercent / static.ContainerCPULimit
	}
	mem := 0.0
	if static.ContainerMemLimitBytes > 0 {
		mem = (float64(rssBytes) / float64(static.ContainerMemLimitBytes)) * percentMultiplier
	}
	return cpu, mem
}

func getProcessInfo(
	p ProcessHandle,
	static *StaticInfo,
	gpuProcessMap map[int32]GPUProcessInfo,
) (ProcessInfo, error) {
	cpuPercent, err := p.CPUPercent()
	if err != nil {
		return ProcessInfo{}, fmt.Errorf("cpu percent: %w", err)
	}
	memInfo, err := p.MemoryInfo()
	if err != nil {
		return ProcessInfo{}, fmt.Errorf("memory info: %w", err)
	}
	user, err := p.Username()
	if err != nil {
		user = "n/a"
	}
	cmdline, err := resolveProcessCommand(p)
	if err != nil {
		return ProcessInfo{}, err
	}

	containerCPUPercent, containerMemPercent := computeContainerPercents(cpuPercent, memInfo.RSS, static)

	procInfo := ProcessInfo{
		PID:           p.PID(),
		User:          user,
		CPUPercent:    containerCPUPercent,
		MemPercent:    containerMemPercent,
		Command:       cmdline,
		rawCPU:        cpuPercent,
		GPUIndex:      -1,
		ParentPID:     processParentPID(p),
		StartTime:     processStartTime(p),
		ThreadCount:   processThreadCount(p),
		OpenFileCount: processOpenFileCount(p),
	}
	if gpuInfo, onGPU := gpuProcessMap[p.PID()]; onGPU {
		procInfo.GPUIndex = gpuInfo.GPUIndex
		procInfo.GPUUtil = gpuInfo.GPUUtil
		procInfo.GPUMemPercent = float64(gpuInfo.GPUMemUtil)
	}
	return procInfo, nil
}

func processParentPID(p ProcessHandle) int32 {
	ppid, err := p.ParentPID()
	if err != nil {
		return 0
	}
	return ppid
}

func processStartTime(p ProcessHandle) time.Time {
	createTime, err := p.CreateTime()
	if err != nil || createTime <= 0 {
		return time.Time{}
	}
	return time.UnixMilli(createTime)
}

func processThreadCount(p ProcessHandle) int32 {
	threads, err := p.NumThreads()
	if err != nil {
		return 0
	}
	return threads
}

func processOpenFileCount(p ProcessHandle) int {
	files, err := p.OpenFiles()
	if err != nil {
		return 0
	}
	return len(files)
}
