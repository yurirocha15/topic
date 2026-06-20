package main

import (
	"sort"
	"strconv"

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
