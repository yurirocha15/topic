// main.go
package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/process"
)

// CgroupVersion denotes the cgroup version (1 or 2)
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
	GPUCount               int
	GPUTotalGB             []float64
}

// GPUUsage holds live usage data for a single GPU.
type GPUUsage struct {
	Index       int
	Utilization int
	MemUsedGB   float64
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
	User          string
	CPUPercent    float64
	MemPercent    float64
	Command       string
	rawCPU        float64
	GPUIndex      int
	GPUUtil       uint64
	GPUMemPercent float64
}

// DynamicInfo holds all the data that is refreshed on each tick.
type DynamicInfo struct {
	mu                 sync.Mutex
	ContainerCPUUsage  float64
	ContainerMemUsedGB float64
	LiveGPUUsage       []GPUUsage
	Processes          []ProcessInfo
}

// State holds the application's entire state.
type State struct {
	static       StaticInfo
	dynamic      DynamicInfo
	prevCPUUsage uint64
	prevCPUTime  time.Time
}

// --- Main Application ---

func main() {
	staticInfo, err := getStaticInfo()
	if err != nil {
		log.Printf("Could not get all static info: %v. Continuing...", err)
	}

	state := &State{
		static:      staticInfo,
		dynamic:     DynamicInfo{},
		prevCPUTime: time.Now(),
	}

	app := tview.NewApplication()

	// --- Create UI Components ---

	// Left panel for resources
	resourceView := tview.NewTextView().
		SetDynamicColors(true).
		SetWrap(false)

	// Right panel for title and help
	infoView := tview.NewTextView().
		SetDynamicColors(true).
		SetTextAlign(tview.AlignLeft)

	// Bottom panel for processes
	processTable := tview.NewTable().
		SetSelectable(true, false).
		SetFixed(1, 0).
		SetSeparator(tview.Borders.Vertical)
	processTable.SetBorder(true).SetTitle(" Processes ")

	// --- Create the Layout ---

	// Top panel combines resources (left) and info (right)
	topPanel := tview.NewFlex().
		SetDirection(tview.FlexColumn).
		AddItem(resourceView, 0, 1, false). // Resources take up remaining width
		AddItem(infoView, 55, 0, false)     // Info panel has a fixed width

	topPanel.SetBorder(true).SetTitle(" System Information ")

	// Main layout combines top panel and process table
	mainLayout := tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(topPanel, 10, 0, false).  // Top panel has a placeholder height, will be resized
		AddItem(processTable, 0, 1, true) // Process table takes all remaining vertical space

	// --- Goroutine for periodic updates ---
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			<-ticker.C
			updateAll(state)
			app.QueueUpdateDraw(func() {
				// Update all components and dynamically resize the top panel
				leftHeight := updateResourceView(resourceView, state)
				rightHeight := updateInfoView(infoView)
				topPanelHeight := int(math.Max(float64(leftHeight), float64(rightHeight))) + 2 // +2 for border

				mainLayout.ResizeItem(topPanel, topPanelHeight, 0)
				updateProcessTable(processTable, state)
			})
		}
	}()

	app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Rune() == 'q' || event.Key() == tcell.KeyCtrlC {
			app.Stop()
		}
		return event
	})

	// Initial data load and draw
	updateAll(state)
	leftHeight := updateResourceView(resourceView, state)
	rightHeight := updateInfoView(infoView)
	topPanelHeight := int(math.Max(float64(leftHeight), float64(rightHeight))) + 2
	mainLayout.ResizeItem(topPanel, topPanelHeight, 0)
	updateProcessTable(processTable, state)

	if err := app.SetRoot(mainLayout, true).Run(); err != nil {
		panic(err)
	}
}

// --- UI Update Functions ---

// updateInfoView updates the info view with ASCII art and navigation guide.
func updateInfoView(view *tview.TextView) int {
	asciiArt := `
[yellow] ░██████████  ░██████   ░█████████  ░██████  ░██████  
[yellow]     ░██     ░██   ░██  ░██     ░██   ░██   ░██   ░██ 
[yellow]     ░██    ░██     ░██ ░██     ░██   ░██  ░██        
[yellow]     ░██    ░██     ░██ ░█████████    ░██  ░██        
[yellow]     ░██    ░██     ░██ ░██           ░██  ░██        
[yellow]     ░██     ░██   ░██  ░██           ░██   ░██   ░██ 
[yellow]     ░██      ░██████   ░██         ░██████  ░██████  
`
	subTitle := "\n       [::b]top inside a container"
	guide := `

[darkgrey]      Quit: q, Ctrl+C
[darkgrey]  Navigate: ←↑→↓ / Mouse
[darkgrey]                        `

	fullText := asciiArt + subTitle + guide
	view.SetText(fullText)
	return strings.Count(fullText, "\n")
}

// updateResourceView updates the current usage statistics panel.
func updateResourceView(view *tview.TextView, state *State) int {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()

	var builder strings.Builder

	// --- CPU ---
	cpuLimitStr := fmt.Sprintf("(limit: %.2f CPUs)", state.static.ContainerCPULimit)
	if state.static.ContainerCPULimit == float64(state.static.HostCores) {
		cpuLimitStr = fmt.Sprintf("(no cgroup limit, %d host cores)", state.static.HostCores)
	}
	builder.WriteString(fmt.Sprintf("CPU: [yellow]%-6.1f%%[white] %s [darkcyan]%s[white]\n",
		state.dynamic.ContainerCPUUsage,
		makeBar(state.dynamic.ContainerCPUUsage),
		cpuLimitStr))

	// --- Memory ---
	memPercent := 0.0
	if state.static.ContainerMemLimitBytes > 0 {
		memPercent = (state.dynamic.ContainerMemUsedGB * 1024 * 1024 * 1024 / float64(state.static.ContainerMemLimitBytes)) * 100
	}
	memUsageStr := fmt.Sprintf("%.3f GB / %.3f GB", state.dynamic.ContainerMemUsedGB, state.static.ContainerMemLimitGB)
	if state.static.ContainerMemLimitGB == 0 {
		memUsageStr = fmt.Sprintf("%.3f GB / N/A", state.dynamic.ContainerMemUsedGB)
	}
	builder.WriteString(fmt.Sprintf("Mem: [yellow]%-6.1f%%[white] %s [darkcyan]%s[white]\n",
		memPercent,
		makeBar(memPercent),
		memUsageStr))

	// --- GPUs ---
	if state.static.GPUCount > 0 {
		builder.WriteString("\n")
		for i, gpu := range state.dynamic.LiveGPUUsage {
			gpuMemPercent := 0.0
			if state.static.GPUTotalGB[i] > 0 {
				gpuMemPercent = (gpu.MemUsedGB / state.static.GPUTotalGB[i]) * 100
			}
			gpuMemUsageStr := fmt.Sprintf("%.2f GB / %.2f GB", gpu.MemUsedGB, state.static.GPUTotalGB[i])

			builder.WriteString(fmt.Sprintf("GPU%d Util: [yellow]%-3d%%[white] %s\n",
				i, gpu.Utilization, makeBar(float64(gpu.Utilization))))
			builder.WriteString(fmt.Sprintf("GPU%d Mem:  [yellow]%-3.0f%%[white] %s [darkcyan]%s[white]\n",
				i, gpuMemPercent, makeBar(gpuMemPercent), gpuMemUsageStr))
		}
	}

	finalText := builder.String()
	view.SetText(finalText)

	return strings.Count(finalText, "\n")
}

func updateProcessTable(table *tview.Table, state *State) {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()

	table.Clear()

	// --- Create Header ---
	headers := []string{"PID", "USER", "%CPU", "%MEM", "%GPU", "%GPUMEM", "COMMAND"}
	for i, header := range headers {
		cell := tview.NewTableCell(header).
			SetTextColor(tcell.ColorYellow).
			SetAlign(tview.AlignLeft).
			SetSelectable(false)
		table.SetCell(0, i, cell)
	}

	// --- Populate Data ---
	for r, p := range state.dynamic.Processes {
		// PID
		table.SetCell(r+1, 0, tview.NewTableCell(fmt.Sprintf("%d", p.PID)).SetTextColor(tcell.ColorWhite))
		// USER
		table.SetCell(r+1, 1, tview.NewTableCell(p.User).SetTextColor(tcell.ColorGreen))
		// %CPU
		table.SetCell(r+1, 2, tview.NewTableCell(fmt.Sprintf("%.1f", p.CPUPercent)).SetTextColor(tcell.ColorAqua))
		// %MEM
		table.SetCell(r+1, 3, tview.NewTableCell(fmt.Sprintf("%.1f", p.MemPercent)).SetTextColor(tcell.ColorAqua))

		// %GPU and %GPUMEM
		if p.GPUIndex != -1 {
			table.SetCell(r+1, 4, tview.NewTableCell(fmt.Sprintf("%d", p.GPUUtil)).SetTextColor(tcell.ColorFuchsia))
			table.SetCell(r+1, 5, tview.NewTableCell(fmt.Sprintf("%.1f", p.GPUMemPercent)).SetTextColor(tcell.ColorFuchsia))
		} else {
			table.SetCell(r+1, 4, tview.NewTableCell("-").SetTextColor(tcell.ColorDarkGray))
			table.SetCell(r+1, 5, tview.NewTableCell("-").SetTextColor(tcell.ColorDarkGray))
		}

		// COMMAND
		cmdCell := tview.NewTableCell(p.Command).
			SetTextColor(tcell.ColorWhite).
			SetExpansion(1).
			SetMaxWidth(0) // Prevent truncation
		table.SetCell(r+1, 6, cmdCell)
	}
}

// makeBar creates a visual bar representation of a percentage.
func makeBar(percent float64) string {
	barWidth := 20
	filledWidth := int((float64(barWidth) * percent) / 100.0)
	if filledWidth > barWidth {
		filledWidth = barWidth
	}
	if filledWidth < 0 {
		filledWidth = 0
	}

	bar := strings.Repeat("█", filledWidth) + strings.Repeat("─", barWidth-filledWidth)
	return fmt.Sprintf("[green]%s[white]", bar)
}

// --- Data Fetching Functions ---

// updateAll fetches all dynamic data and updates the state.
func updateAll(state *State) {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()
	var wg sync.WaitGroup
	wg.Add(4)
	go func() { defer wg.Done(); state.dynamic.ContainerCPUUsage = updateContainerCPUUsage(state) }()
	go func() {
		defer wg.Done()
		state.dynamic.ContainerMemUsedGB = updateContainerMemUsage(state.static.CgroupVersion)
	}()
	go func() { defer wg.Done(); state.dynamic.LiveGPUUsage = updateLiveGPUUsage(state.static.GPUCount) }()
	go func() { defer wg.Done(); state.dynamic.Processes = updateProcessList(&state.static) }()
	wg.Wait()
}

// getStaticInfo fetches static information about the container and host.
func getStaticInfo() (StaticInfo, error) {
	var info StaticInfo
	var err error
	if _, err := os.Stat("/sys/fs/cgroup/cpu.max"); os.IsNotExist(err) {
		info.CgroupVersion = CgroupV1
	} else {
		info.CgroupVersion = CgroupV2
	}
	info.HostCores, err = cpu.Counts(false)
	if err != nil {
		info.HostCores = 1
	}
	info.ContainerCPULimit = getContainerCPULimit(info.CgroupVersion, info.HostCores)
	info.ContainerMemLimitBytes, info.ContainerMemLimitGB = getContainerMemLimit(info.CgroupVersion)
	info.GPUCount, info.GPUTotalGB = getStaticGPUInfo()
	return info, nil
}

// readUintFromFile reads a uint64 value from a file, trimming whitespace.
func readUintFromFile(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	val, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, err
	}
	return val, nil
}

// readStringFromFile reads a string from a file, trimming whitespace.
func readStringFromFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

// getContainerCPULimit reads the CPU limit from cgroup files and returns it as a percentage of host cores.
func getContainerCPULimit(cgroupVersion CgroupVersion, hostCores int) float64 {
	var quota, period int64
	if cgroupVersion == CgroupV2 {
		cpuMaxStr, _ := readStringFromFile("/sys/fs/cgroup/cpu.max")
		parts := strings.Fields(cpuMaxStr)
		if len(parts) == 2 {
			if parts[0] == "max" {
				return float64(hostCores)
			}
			quota, _ = strconv.ParseInt(parts[0], 10, 64)
			period, _ = strconv.ParseInt(parts[1], 10, 64)
		}
	} else { // CgroupV1
		q, err1 := readUintFromFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
		p, err2 := readUintFromFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
		if err1 == nil && err2 == nil {
			quota, period = int64(q), int64(p)
		} else {
			return float64(hostCores)
		}
	}
	if quota > 0 && period > 0 {
		return float64(quota) / float64(period)
	}
	return float64(hostCores)
}

// getContainerMemLimit reads the memory limit from cgroup files and returns it in bytes and GB.
func getContainerMemLimit(cgroupVersion CgroupVersion) (int64, float64) {
	var path string
	if cgroupVersion == CgroupV2 {
		path = "/sys/fs/cgroup/memory.max"
	} else {
		path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	}

	limitStr, err := readStringFromFile(path)
	if err != nil || limitStr == "max" {
		return 0, 0
	}

	limitBytes, err := strconv.ParseInt(limitStr, 10, 64)
	if err != nil || limitBytes <= 0 {
		return 0, 0
	}

	return limitBytes, float64(limitBytes) / 1024 / 1024 / 1024
}

// updateContainerCPUUsage reads the CPU usage from the cgroup filesystem.
func updateContainerCPUUsage(state *State) float64 {
	var currentUsage uint64

	if state.static.CgroupVersion == CgroupV2 {
		statStr, _ := readStringFromFile("/sys/fs/cgroup/cpu.stat")
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "usage_usec" {
				currentUsage, _ = strconv.ParseUint(parts[1], 10, 64) // value is in microseconds
				break
			}
		}
	} else { // CgroupV1
		// value is in nanoseconds, convert to microseconds
		ns, err := readUintFromFile("/sys/fs/cgroup/cpuacct/cpuacct.usage")
		if err == nil {
			currentUsage = ns / 1000
		}
	}

	now := time.Now()
	timeDelta := now.Sub(state.prevCPUTime).Seconds()

	var cpuPercent float64
	if timeDelta > 0 && currentUsage > state.prevCPUUsage {
		usageDelta := float64(currentUsage - state.prevCPUUsage) // in microseconds
		// usageDelta is how many microseconds of CPU time were used.
		// timeDelta * 1e6 is how many microseconds passed in wall-clock time.
		// Normalize by the number of cores to get a percentage of the container's total capacity.
		cpuPercent = (usageDelta / (timeDelta * 1e6)) * 100 / state.static.ContainerCPULimit
	}

	state.prevCPUUsage = currentUsage
	state.prevCPUTime = now

	if cpuPercent > 100.0 {
		return 100.0
	}
	return cpuPercent
}

// updateContainerMemUsage reads the memory usage from the cgroup filesystem.
func updateContainerMemUsage(cgroupVersion CgroupVersion) float64 {
	if cgroupVersion == CgroupV2 {
		// 'anon' memory is a good proxy for non-cache process memory
		statStr, _ := readStringFromFile("/sys/fs/cgroup/memory.stat")
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "anon" {
				bytes, _ := strconv.ParseUint(parts[1], 10, 64)
				return float64(bytes) / 1024 / 1024 / 1024
			}
		}
	} else { // CgroupV1
		// memory.usage_in_bytes includes file cache, but is the standard.
		path := "/sys/fs/cgroup/memory/memory.usage_in_bytes"
		bytes, err := readUintFromFile(path)
		if err == nil {
			return float64(bytes) / 1024 / 1024 / 1024
		}
	}
	return 0
}

// getStaticGPUInfo fetches total memory for each GPU.
func getStaticGPUInfo() (int, []float64) {
	cmdMem := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
	outMem, err := cmdMem.Output()
	if err != nil {
		return 0, nil
	}
	linesMem := strings.Split(strings.TrimSpace(string(outMem)), "\n")
	if len(linesMem) == 0 || linesMem[0] == "" {
		return 0, nil
	}

	totals := make([]float64, len(linesMem))
	for i, line := range linesMem {
		mb, _ := strconv.ParseFloat(strings.TrimSpace(line), 64)
		totals[i] = mb / 1024
	}
	return len(linesMem), totals
}

// updateLiveGPUUsage fetches current GPU utilization and memory usage.
func updateLiveGPUUsage(gpuCount int) []GPUUsage {
	if gpuCount == 0 {
		return nil
	}
	cmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	usage := make([]GPUUsage, 0, len(lines))
	for i, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			continue
		}
		util, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		memUsedMB, _ := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		usage = append(usage, GPUUsage{
			Index: i, Utilization: util, MemUsedGB: memUsedMB / 1024,
		})
	}
	return usage
}

// getGPUProcessMap queries nvidia-smi pmon for apps running on the GPU and maps their PID to usage.
func getGPUProcessMap() map[int32]GPUProcessInfo {
	// Use `pmon` with `-c 1` to get a single snapshot, and `-s um` for utilization and memory.
	cmd := exec.Command("nvidia-smi", "pmon", "-c", "1", "-s", "um")
	out, err := cmd.Output()
	if err != nil {
		return nil
	}

	processMap := make(map[int32]GPUProcessInfo)
	lines := strings.Split(string(out), "\n")

	for _, line := range lines {
		// skip headers
		if strings.HasPrefix(line, "#") {
			continue
		}

		fields := strings.Fields(line)
		// Expecting 8 columns: gpu, pid, type, sm, mem, enc, dec, command
		if len(fields) < 8 {
			continue
		}

		// Fields: 0=gpu_idx, 1=pid, 3=sm_util, 4=mem_util
		pid, err := strconv.ParseInt(fields[1], 10, 32)
		if err != nil {
			continue
		}

		gpuIndex, _ := strconv.Atoi(fields[0])
		gpuUtil, _ := strconv.ParseUint(fields[3], 10, 64)
		memUtil, _ := strconv.ParseUint(fields[4], 10, 64)

		processMap[int32(pid)] = GPUProcessInfo{
			GPUIndex:   gpuIndex,
			GPUUtil:    gpuUtil,
			GPUMemUtil: memUtil,
		}
	}
	return processMap
}

// updateProcessList fetches the current process list and adds resource usage info when possible.
func updateProcessList(static *StaticInfo) []ProcessInfo {
	gpuProcessMap := getGPUProcessMap()
	procs, err := process.Processes()
	if err != nil {
		return nil
	}
	var processList []ProcessInfo
	for _, p := range procs {
		cpuPercent, err := p.CPUPercent()
		if err != nil {
			continue
		}
		memInfo, err := p.MemoryInfo()
		if err != nil {
			continue
		}
		user, err := p.Username()
		if err != nil {
			user = "n/a"
		}
		cmdline, err := p.Cmdline()
		if err != nil || cmdline == "" {
			name, err := p.Name()
			if err != nil {
				continue
			}
			cmdline = "[" + name + "]"
		}
		containerCPUPercent := 0.0
		if static.ContainerCPULimit > 0 {
			containerCPUPercent = cpuPercent / static.ContainerCPULimit
		}
		containerMemPercent := 0.0
		if static.ContainerMemLimitBytes > 0 {
			containerMemPercent = (float64(memInfo.RSS) / float64(static.ContainerMemLimitBytes)) * 100
		}
		gpuInfo, onGPU := gpuProcessMap[p.Pid]

		pi := ProcessInfo{
			PID:        p.Pid,
			User:       user,
			CPUPercent: containerCPUPercent,
			MemPercent: containerMemPercent,
			Command:    cmdline,
			rawCPU:     cpuPercent,
			GPUIndex:   -1, // Default to no GPU
		}

		if onGPU {
			pi.GPUIndex = gpuInfo.GPUIndex
			pi.GPUUtil = gpuInfo.GPUUtil
			pi.GPUMemPercent = float64(gpuInfo.GPUMemUtil)
		}

		processList = append(processList, pi)
	}
	sort.Slice(processList, func(i, j int) bool {
		return processList[i].rawCPU > processList[j].rawCPU
	})
	return processList
}
