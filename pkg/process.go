package main

import (
	"cmp"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode"

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
		processes = slices.Clone(state.dynamic.Processes)
		state.dynamic.mu.Unlock()
		unlockAfterRender = false
		processes = prepareProcessRows(processes, ui)
	}
	if unlockAfterRender {
		defer state.dynamic.mu.Unlock()
	}

	previousRow, _ := table.GetSelection()

	showGPU := state.static.GPUCount > 0
	columns := processColumnsFor(showGPU)

	// --- Create Header ---
	sortPosition := processColumnPosition(processSortHeaderIndex(ui.ProcessSort), showGPU)
	for position, header := range processTableHeaders(ui, showGPU) {
		color := tcell.ColorLightGray
		if position == sortPosition {
			color = tcell.ColorAqua
		}
		setTableCell(table, 0, position, header, color, false, 0)
	}

	// --- Populate Data ---
	for r, p := range processes {
		for position, column := range columns {
			text, color, expansion := processCellContent(p, column)
			setTableCell(table, r+1, position, text, color, true, expansion)
		}
		table.GetCell(r+1, processPIDColumn).SetReference(p.PID)
	}

	for row := table.GetRowCount() - 1; row > len(processes); row-- {
		table.RemoveRow(row)
	}
	if len(processes) == 0 {
		writeEmptyProcessMessage(table, columns, ui.ProcessFilter)
	}
	restoreProcessSelection(table, state, processes, previousRow)

	table.SetTitle(escapeText(processTableTitle(ui, len(processes), totalProcesses)))
}

// trackProcessSelection remembers the process the user highlights with the keyboard or mouse.
func trackProcessSelection(table *tview.Table, state *State) {
	table.SetSelectionChangedFunc(func(row, _ int) {
		if pid, ok := table.GetCell(row, processPIDColumn).GetReference().(int32); ok {
			state.followPID.Store(pid)
		}
	})
}

// restoreProcessSelection moves the highlight to the row of the process the user picked. Until the user
// picks one, or after it exits or is filtered out, the highlight stays on its row, moved up to the last
// row if the table got shorter.
func restoreProcessSelection(table *tview.Table, state *State, processes []ProcessInfo, previousRow int) {
	if len(processes) == 0 {
		return
	}
	if followPID := state.followPID.Load(); followPID != 0 {
		for index, process := range processes {
			if process.PID == followPID {
				table.Select(index+1, 0)
				return
			}
		}
	}
	if previousRow > len(processes) {
		table.Select(len(processes), 0)
	}
	// Moving the highlight here is not a choice by the user, so stop following
	state.followPID.Store(0)
}

// writeEmptyProcessMessage fills the first row with a muted, unselectable note saying why the table
// is empty, so an empty filter result does not look like a broken table.
func writeEmptyProcessMessage(table *tview.Table, columns []int, filter string) {
	message := "No processes to show"
	if filter != "" {
		message = fmt.Sprintf("No processes match %q. Press / then Ctrl+U to clear the filter.", filter)
	}
	for position := range columns {
		setTableCell(table, 1, position, "", tcell.ColorGray, false, 0)
	}
	setTableCell(table, 1, len(columns)-1, message, tcell.ColorGray, false, 1)
}

// processColumnsFor lists the table's columns in order. The GPU columns appear only with a GPU.
func processColumnsFor(showGPU bool) []int {
	if showGPU {
		return []int{
			processPIDColumn, processUserColumn, processCPUColumn, processMemColumn,
			processGPUColumn, processGPUMemColumn, processCommandColumn,
		}
	}
	return []int{processPIDColumn, processUserColumn, processCPUColumn, processMemColumn, processCommandColumn}
}

// processColumnPosition is where a column is drawn, or -1 when it is hidden.
func processColumnPosition(column int, showGPU bool) int {
	return slices.Index(processColumnsFor(showGPU), column)
}

func processCellContent(p ProcessInfo, column int) (string, tcell.Color, int) {
	switch column {
	case processPIDColumn:
		return strconv.Itoa(int(p.PID)), tcell.ColorGray, 0
	case processUserColumn:
		return p.User, tcell.ColorLightGray, 0
	case processCPUColumn:
		return strconv.FormatFloat(p.CPUPercent, 'f', 1, 64), usageCellColor(p.CPUPercent), 0
	case processMemColumn:
		return strconv.FormatFloat(p.MemPercent, 'f', 1, 64), usageCellColor(p.MemPercent), 0
	case processGPUColumn:
		if p.GPUIndex == -1 {
			return "-", tcell.ColorDarkGray, 0
		}
		return strconv.FormatUint(p.GPUUtil, 10), usageCellColor(float64(p.GPUUtil)), 0
	case processGPUMemColumn:
		if p.GPUIndex == -1 {
			return "-", tcell.ColorDarkGray, 0
		}
		return strconv.FormatFloat(p.GPUMemPercent, 'f', 1, 64), usageCellColor(p.GPUMemPercent), 0
	default:
		return p.Command, tcell.ColorWhite, 1
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
	cell.SetText(escapeText(text)).
		SetTextColor(color).
		SetAlign(tview.AlignLeft).
		SetSelectable(selectable).
		SetExpansion(expansion)
	table.SetCell(row, column, cell)
}

func needsProcessTransform(ui UIState) bool {
	return ui.TreeMode || ui.ProcessFilter != "" || ui.ReverseSort || ui.ProcessSort != SortByCPU
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
	slices.SortStableFunc(processes, func(left, right ProcessInfo) int {
		if reverse {
			return compareProcesses(right, left, sortColumn)
		}
		return compareProcesses(left, right, sortColumn)
	})
}

// compareProcesses orders by the sort column, breaking ties by ascending PID.
func compareProcesses(left ProcessInfo, right ProcessInfo, sortColumn ProcessSortColumn) int {
	return cmp.Or(comparePrimary(left, right, sortColumn), cmp.Compare(left.PID, right.PID))
}

func comparePrimary(left ProcessInfo, right ProcessInfo, sortColumn ProcessSortColumn) int {
	switch sortColumn {
	case SortByMemory:
		return cmp.Compare(right.MemPercent, left.MemPercent)
	case SortByGPU:
		return cmp.Compare(right.GPUUtil, left.GPUUtil)
	case SortByGPUMemory:
		return cmp.Compare(right.GPUMemPercent, left.GPUMemPercent)
	case SortByPID:
		return cmp.Compare(left.PID, right.PID)
	case SortByUser:
		return cmp.Compare(left.User, right.User)
	case SortByCommand:
		return cmp.Compare(left.Command, right.Command)
	case SortByCPU:
		fallthrough
	default:
		return cmp.Compare(right.CPUPercent, left.CPUPercent)
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

func processTableHeaders(ui UIState, showGPU bool) []string {
	labels := map[int]string{
		processPIDColumn:     processPIDLabel,
		processUserColumn:    processUserLabel,
		processCPUColumn:     processCPULabel,
		processMemColumn:     processMemoryLabel,
		processGPUColumn:     processGPULabel,
		processGPUMemColumn:  processGPUMemoryLabel,
		processCommandColumn: processCommandLabel,
	}
	columns := processColumnsFor(showGPU)
	headers := make([]string, 0, len(columns))
	for _, column := range columns {
		headers = append(headers, labels[column])
	}
	if position := processColumnPosition(processSortHeaderIndex(ui.ProcessSort), showGPU); position >= 0 {
		headers[position] += sortDirectionMarker(ui)
	}
	return headers
}

// naturallyDescending reports whether a column sorts from high to low before it is reversed.
// Usage columns put the busiest processes first; PID, user, and command sort A to Z.
func naturallyDescending(column ProcessSortColumn) bool {
	switch column {
	case SortByCPU, SortByMemory, SortByGPU, SortByGPUMemory:
		return true
	case SortByPID, SortByUser, SortByCommand:
		return false
	default:
		return true
	}
}

func sortsDescending(ui UIState) bool {
	return naturallyDescending(ui.ProcessSort) != ui.ReverseSort
}

// sortDirectionMarker is ↓ when the table runs from high to low and ↑ when it runs from low to high.
func sortDirectionMarker(ui UIState) string {
	if sortsDescending(ui) {
		return "↓"
	}
	return "↑"
}

func selectedProcess(table *tview.Table, state *State) (ProcessInfo, bool) {
	row, _ := table.GetSelection()
	if row <= 0 || row >= table.GetRowCount() {
		return ProcessInfo{}, false
	}
	pid, ok := table.GetCell(row, processPIDColumn).GetReference().(int32)
	if !ok {
		return ProcessInfo{}, false
	}

	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()
	for _, process := range state.dynamic.Processes {
		if process.PID == pid {
			return process, true
		}
	}
	return ProcessInfo{}, false
}

func processDetailsText(process ProcessInfo) string {
	start := "unknown"
	if !process.StartTime.IsZero() {
		start = process.StartTime.Format(time.RFC3339)
	}
	gpu := "none"
	if process.GPUIndex >= 0 {
		gpu = fmt.Sprintf(
			"GPU%d util=%d%% mem=%.1f%% (%d MiB)",
			process.GPUIndex,
			process.GPUUtil,
			process.GPUMemPercent,
			process.GPUMemMB,
		)
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

// nextProcessSortColumn moves sort mode one column right, skipping hidden GPU columns.
func nextProcessSortColumn(current ProcessSortColumn, showGPU bool) ProcessSortColumn {
	next := nextSortColumnInTable(current)
	for !showGPU && (next == SortByGPU || next == SortByGPUMemory) {
		next = nextSortColumnInTable(next)
	}
	return next
}

// previousProcessSortColumn moves sort mode one column left, skipping hidden GPU columns.
func previousProcessSortColumn(current ProcessSortColumn, showGPU bool) ProcessSortColumn {
	previous := previousSortColumnInTable(current)
	for !showGPU && (previous == SortByGPU || previous == SortByGPUMemory) {
		previous = previousSortColumnInTable(previous)
	}
	return previous
}

func nextSortColumnInTable(current ProcessSortColumn) ProcessSortColumn {
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

func previousSortColumnInTable(current ProcessSortColumn) ProcessSortColumn {
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

// processKey identifies a process across refreshes, because PIDs can be reused.
type processKey struct {
	pid        int32
	createTime int64
}

// processCPUTracker turns cumulative process CPU time into usage since the previous refresh.
type processCPUTracker struct {
	samples   map[processKey]float64
	sampledAt time.Time
}

// percent records a process's CPU time in next and returns its usage, where 100 is one full core.
// It returns 0 until the process has a previous sample to compare against.
func (t *processCPUTracker) percent(p ProcessHandle, next map[processKey]float64, now time.Time) (float64, error) {
	times, err := p.Times()
	if err != nil {
		return 0, fmt.Errorf("cpu times: %w", err)
	}
	createTime, err := p.CreateTime()
	if err != nil {
		return 0, fmt.Errorf("create time: %w", err)
	}

	key := processKey{pid: p.PID(), createTime: createTime}
	seconds := times.User + times.System
	next[key] = seconds

	previous, ok := t.samples[key]
	elapsed := now.Sub(t.sampledAt).Seconds()
	if !ok || elapsed <= 0 || seconds <= previous {
		return 0, nil
	}
	return (seconds - previous) / elapsed * percentMultiplier, nil
}

// updateProcessListWithProvider fetches the process list and adds resource usage info when possible.
func updateProcessListWithProvider(
	static *StaticInfo,
	runner CommandRunner,
	provider ProcessProvider,
	tracker *processCPUTracker,
	now time.Time,
) []ProcessInfo {
	var gpuProcessMap map[int32]GPUProcessInfo
	if static.GPUCount > 0 {
		gpuProcessMap = getGPUProcessMap(runner)
	}
	procs, err := provider.Processes()
	if err != nil {
		return nil
	}
	processList := make([]ProcessInfo, 0, len(procs))
	samples := make(map[processKey]float64, len(procs))
	for _, p := range procs {
		cpuPercent, cpuErr := tracker.percent(p, samples, now)
		if cpuErr != nil {
			continue
		}
		pi, processErr := getProcessInfo(p, static, gpuProcessMap, cpuPercent)
		if processErr != nil {
			continue
		}
		processList = append(processList, pi)
	}
	tracker.samples = samples
	tracker.sampledAt = now

	slices.SortFunc(processList, func(left, right ProcessInfo) int {
		return cmp.Or(cmp.Compare(right.rawCPU, left.rawCPU), cmp.Compare(left.PID, right.PID))
	})
	return processList
}

func resolveProcessCommand(p ProcessHandle) (string, error) {
	cmdline, err := p.Cmdline()
	if err == nil && cmdline != "" {
		return stripControlCharacters(cmdline), nil
	}
	name, err := p.Name()
	if err != nil {
		return "", fmt.Errorf("process name: %w", err)
	}
	return "[" + stripControlCharacters(name) + "]", nil
}

// stripControlCharacters keeps a process from sending terminal escape sequences or using
// bidirectional and other format characters to disguise what it displays.
func stripControlCharacters(text string) string {
	return strings.Map(func(character rune) rune {
		if unicode.IsControl(character) || unicode.In(character, unicode.Cf, unicode.Zl, unicode.Zp) {
			return ' '
		}
		return character
	}, text)
}

func computeContainerPercents(cpuPercent float64, rssBytes uint64, static *StaticInfo) (float64, float64) {
	cpu := 0.0
	if static.ContainerCPULimit > 0 {
		cpu = cpuPercent / static.ContainerCPULimit
	}
	memTotal := float64(static.ContainerMemLimitBytes)
	if memTotal <= 0 {
		memTotal = static.HostMemTotalGB * bytesPerGB
	}
	mem := 0.0
	if memTotal > 0 {
		mem = (float64(rssBytes) / memTotal) * percentMultiplier
	}
	return cpu, mem
}

func getProcessInfo(
	p ProcessHandle,
	static *StaticInfo,
	gpuProcessMap map[int32]GPUProcessInfo,
	cpuPercent float64,
) (ProcessInfo, error) {
	memInfo, err := p.MemoryInfo()
	if err != nil {
		return ProcessInfo{}, fmt.Errorf("memory info: %w", err)
	}
	user, err := p.Username()
	if err != nil {
		user = "n/a"
	}
	user = stripControlCharacters(user)
	cmdline, err := resolveProcessCommand(p)
	if err != nil {
		return ProcessInfo{}, err
	}

	containerCPUPercent, containerMemPercent := computeContainerPercents(cpuPercent, memInfo.RSS, static)

	procInfo := ProcessInfo{
		PID:         p.PID(),
		User:        user,
		CPUPercent:  containerCPUPercent,
		MemPercent:  containerMemPercent,
		Command:     cmdline,
		rawCPU:      cpuPercent,
		GPUIndex:    -1,
		ParentPID:   processParentPID(p),
		StartTime:   processStartTime(p),
		ThreadCount: processThreadCount(p),
	}
	if gpuInfo, onGPU := gpuProcessMap[p.PID()]; onGPU {
		procInfo.GPUIndex = gpuInfo.GPUIndex
		procInfo.GPUUtil = gpuInfo.GPUUtil
		procInfo.GPUMemMB = gpuInfo.UsedMemMB
		procInfo.GPUMemPercent = gpuMemoryPercent(gpuInfo, static)
	}
	return procInfo, nil
}

// gpuMemoryPercent is the process's share of its GPU's total memory, or 0 when the total is unknown.
func gpuMemoryPercent(gpuInfo GPUProcessInfo, static *StaticInfo) float64 {
	if static == nil || gpuInfo.GPUIndex < 0 || gpuInfo.GPUIndex >= len(static.GPUTotalGB) {
		return 0
	}
	totalMB := static.GPUTotalGB[gpuInfo.GPUIndex] * bytesPerKB
	if totalMB <= 0 {
		return 0
	}
	return float64(gpuInfo.UsedMemMB) / totalMB * percentMultiplier
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
