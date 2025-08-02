// main.go
package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"unicode"

	"github.com/gdamore/tcell/v2"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/process"
	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
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
	GPUUUIDMap             map[string]int
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
	CPUPercent    float64 // Container-relative
	MemPercent    float64 // Container-relative
	Command       string
	rawCPU        float64 // Host-relative, for sorting
	GPUIndex      int     // -1 if not on GPU
	GPUUtil       uint64
	GPUMemPercent float64
}

// GPUUsage holds live usage data for a single GPU.
type GPUUsage struct {
	Index       int
	Utilization int
	MemUsedGB   float64
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
	// Initialize Tcell screen
	screen, err := tcell.NewScreen()
	if err != nil {
		log.Fatalf("Failed to create screen: %v", err)
	}
	if err := screen.Init(); err != nil {
		log.Fatalf("Failed to initialize screen: %v", err)
	}
	defer screen.Fini() // Ensure terminal is restored on exit

	// Fetch static info once
	staticInfo, err := getStaticInfo()
	if err != nil {
		// Log the error but continue; the UI will show "N/A"
		log.Printf("Could not get all static info: %v. Continuing...", err)
	}

	state := &State{
		static:      staticInfo,
		dynamic:     DynamicInfo{},
		prevCPUTime: time.Now(),
	}

	// Channels for managing updates and quitting
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	quit := make(chan struct{})

	// Goroutine to handle keyboard input
	go func() {
		for {
			ev := screen.PollEvent()
			switch ev := ev.(type) {
			case *tcell.EventKey:
				if ev.Rune() == 'q' || ev.Rune() == 'Q' || ev.Key() == tcell.KeyCtrlC {
					close(quit)
					return
				}
			case *tcell.EventResize:
				screen.Sync() // On resize, sync the screen
			}
		}
	}()

	// Initial draw
	updateAll(state)
	draw(screen, state)

	// Main loop
	for {
		select {
		case <-ticker.C:
			updateAll(state)
			draw(screen, state)
		case <-quit:
			return
		}
	}
}

// --- Data Fetching & Updating ---

func getStaticInfo() (StaticInfo, error) {
	var info StaticInfo
	var err error

	// Detect Cgroup version
	if _, err := os.Stat("/sys/fs/cgroup/cpu.max"); os.IsNotExist(err) {
		info.CgroupVersion = CgroupV1
	} else {
		info.CgroupVersion = CgroupV2
	}

	// Get Host Cores as a fallback
	info.HostCores, err = cpu.Counts(false)
	if err != nil {
		info.HostCores = 1 // Safe default
	}

	// Get Container CPU and Memory limits
	info.ContainerCPULimit = getContainerCPULimit(info.CgroupVersion, info.HostCores)
	info.ContainerMemLimitBytes, info.ContainerMemLimitGB = getContainerMemLimit(info.CgroupVersion)

	// Get initial GPU info
	info.GPUCount, info.GPUTotalGB, info.GPUUUIDMap = getStaticGPUInfo()

	return info, nil
}

// getGPUProcessMap queries nvidia-smi for apps running on the GPU and maps their PID to usage.
func getGPUProcessMap(uuidMap map[string]int) map[int32]GPUProcessInfo {
	if len(uuidMap) == 0 {
		return nil
	}
	// Query for pid, uuid, memory usage, gpu utilization, and memory utilization
	cmd := exec.Command("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return nil
	}

	processMap := make(map[int32]GPUProcessInfo)
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	for _, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) != 5 {
			continue
		}

		pid, err := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 32)
		if err != nil {
			continue
		}

		uuid := strings.TrimSpace(parts[1])
		mem, _ := strconv.ParseUint(strings.TrimSpace(parts[2]), 10, 64)
		gpuUtil, _ := strconv.ParseUint(strings.TrimSpace(parts[3]), 10, 64)
		memUtil, _ := strconv.ParseUint(strings.TrimSpace(parts[4]), 10, 64)

		if gpuIndex, ok := uuidMap[uuid]; ok {
			processMap[int32(pid)] = GPUProcessInfo{
				GPUIndex:   gpuIndex,
				UsedMemMB:  mem,
				GPUUtil:    gpuUtil,
				GPUMemUtil: memUtil,
			}
		}
	}
	return processMap
}

func updateAll(state *State) {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()

	// Use a WaitGroup to run updates in parallel
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

// --- Drawing ---

func draw(screen tcell.Screen, state *State) {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()

	screen.Clear()
	width, height := screen.Size()

	y := 0

	// Title
	title := "--- topic: top inside containers (press 'q' or Ctrl+C to quit) ---" // UPDATED
	drawString(screen, 0, y, tcell.StyleDefault.Bold(true), truncateString(title, width))
	y++

	// --- Resource Panel ---
	y = drawResourcePanels(screen, y, width, height, state) // UPDATED

	// --- Process List ---
	if y < height {
		drawProcessList(screen, y, width, height, &state.dynamic, &state.static)
	}

	screen.Show()
}

func drawResourcePanels(s tcell.Screen, y, width, height int, state *State) int {
	barWidth := 30 // Fixed width for the usage bars

	// --- CPU ---
	cpuLimitStr := fmt.Sprintf("(limit: %.2f CPUs)", state.static.ContainerCPULimit)
	if state.static.ContainerCPULimit == float64(state.static.HostCores) {
		cpuLimitStr = fmt.Sprintf("(no cgroup limit, %d host cores)", state.static.HostCores)
	}
	cpuText := fmt.Sprintf("CPU: %5.1f%% ", state.dynamic.ContainerCPUUsage)
	drawString(s, 0, y, tcell.StyleDefault, cpuText)
	drawBar(s, len(cpuText), y, barWidth, state.dynamic.ContainerCPUUsage, tcell.StyleDefault.Foreground(tcell.ColorGreen))
	drawString(s, len(cpuText)+barWidth+1, y, tcell.StyleDefault.Dim(true), cpuLimitStr)
	y++

	// --- Memory ---
	memLimitStr := "N/A"
	memPercent := 0.0
	if state.static.ContainerMemLimitBytes > 0 {
		memLimitStr = fmt.Sprintf("%.3f GB", state.static.ContainerMemLimitGB)
		memPercent = (state.dynamic.ContainerMemUsedGB * 1024 * 1024 * 1024 / float64(state.static.ContainerMemLimitBytes)) * 100
	}
	memText := fmt.Sprintf("Mem: %5.1f%% ", memPercent)
	memUsageStr := fmt.Sprintf("%.3f GB / %s", state.dynamic.ContainerMemUsedGB, memLimitStr)
	drawString(s, 0, y, tcell.StyleDefault, memText)
	drawBar(s, len(memText), y, barWidth, memPercent, tcell.StyleDefault.Foreground(tcell.ColorBlue))
	drawString(s, len(memText)+barWidth+1, y, tcell.StyleDefault.Dim(true), memUsageStr)
	y++

	// --- GPUs ---
	if state.static.GPUCount > 0 {
		for i, gpu := range state.dynamic.LiveGPUUsage {
			if y >= height {
				break
			}

			// GPU Utilization
			gpuUtilText := fmt.Sprintf("GPU%d Util: %3d%% ", i, gpu.Utilization)
			drawString(s, 0, y, tcell.StyleDefault, gpuUtilText)
			drawBar(s, len(gpuUtilText), y, barWidth, float64(gpu.Utilization), tcell.StyleDefault.Foreground(tcell.ColorPurple))
			y++

			// GPU Memory
			gpuMemPercent := 0.0
			if state.static.GPUTotalGB[i] > 0 {
				gpuMemPercent = (gpu.MemUsedGB / state.static.GPUTotalGB[i]) * 100
			}
			gpuMemText := fmt.Sprintf("GPU%d Mem:  %3.0f%% ", i, gpuMemPercent)
			gpuMemUsageStr := fmt.Sprintf("%.2f GB / %.2f GB", gpu.MemUsedGB, state.static.GPUTotalGB[i])
			drawString(s, 0, y, tcell.StyleDefault, gpuMemText)
			drawBar(s, len(gpuMemText), y, barWidth, gpuMemPercent, tcell.StyleDefault.Foreground(tcell.ColorAqua))
			drawString(s, len(gpuMemText)+barWidth+1, y, tcell.StyleDefault.Dim(true), gpuMemUsageStr)
			y++
		}
	}
	return y
}

// drawBar draws a horizontal usage bar.
func drawBar(s tcell.Screen, x, y, width int, percent float64, style tcell.Style) {
	if width < 2 { // Not enough space for a bar
		return
	}
	barWidth := width - 2 // for [ and ]

	filledWidth := int((float64(barWidth) * percent) / 100.0)
	if filledWidth > barWidth {
		filledWidth = barWidth
	}

	s.SetContent(x, y, '[', nil, style)
	for i := 0; i < barWidth; i++ {
		if i < filledWidth {
			s.SetContent(x+1+i, y, '█', nil, style)
		} else {
			s.SetContent(x+1+i, y, '─', nil, style)
		}
	}
	s.SetContent(x+1+barWidth, y, ']', nil, style)
}

// drawProcessList draws the list of processes in a table format.
func drawProcessList(s tcell.Screen, y, width, height int, dynamic *DynamicInfo, static *StaticInfo) {
	// Define column widths
	pidW, userW, cpuW, memW, gpuW, gpuMemW := 7, 10, 6, 6, 6, 8

	// --- Draw Header ---
	header := fmt.Sprintf("%-*s %-*s %*s %*s %*s %*s %s",
		pidW, "PID",
		userW, "USER",
		cpuW, "%CPU",
		memW, "%MEM",
		gpuW, "%GPU",
		gpuMemW, "%GPUMEM",
		"COMMAND")
	drawString(s, 0, y, tcell.StyleDefault.Reverse(true), truncateString(header, width))
	y++

	// --- Draw Process Rows ---
	for _, p := range dynamic.Processes {
		if y >= height {
			break
		}

		userStr := truncateString(p.User, userW-1)

		// Prepare GPU strings
		gpuUtilStr := "  -"
		gpuMemStr := "  -"
		if p.GPUIndex != -1 {
			gpuUtilStr = fmt.Sprintf("%2d", p.GPUUtil)
			gpuMemStr = fmt.Sprintf("%5.1f", p.GPUMemPercent)
		}

		// Calculate available width for command string
		cmdW := width - pidW - userW - cpuW - memW - gpuW - gpuMemW - 7 // 7 for spaces
		if cmdW < 1 {
			cmdW = 1
		}
		cmdStr := truncateString(p.Command, cmdW)

		// Format the final line string for printing
		line := fmt.Sprintf("%-*d %-*s %*.1f %*.1f %*s %*s %s",
			pidW, p.PID,
			userW, userStr,
			cpuW, p.CPUPercent,
			memW, p.MemPercent,
			gpuW, gpuUtilStr,
			gpuMemW, gpuMemStr,
			cmdStr)
		drawString(s, 0, y, tcell.StyleDefault, line)
		y++
	}
}

// --- Helper Functions ---

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

func readStringFromFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func drawString(s tcell.Screen, x, y int, style tcell.Style, text string) {
	for i, r := range text {
		s.SetContent(x+i, y, r, nil, style)
	}
}

func truncateString(s string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	// This transformer removes combining characters, which can mess with width calculations
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	s, _, _ = transform.String(t, s)

	if len(s) > maxLen {
		if maxLen > 3 {
			return s[:maxLen-3] + "..."
		}
		return s[:maxLen]
	}
	return s
}

// --- Cgroup and Resource Logic ---

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

	// Cap percentage at 100% of the limit
	if cpuPercent > 100.0 {
		return 100.0
	}
	return cpuPercent
}

func updateContainerMemUsage(cgroupVersion CgroupVersion) float64 {
	var path string
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
		path = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
		bytes, err := readUintFromFile(path)
		if err == nil {
			return float64(bytes) / 1024 / 1024 / 1024
		}
	}
	return 0
}

// getStaticGPUInfo fetches total memory for each GPU and a UUID-to-index map.
func getStaticGPUInfo() (int, []float64, map[string]int) {
	// Get total memory
	cmdMem := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
	outMem, err := cmdMem.Output()
	if err != nil {
		return 0, nil, nil
	}
	linesMem := strings.Split(strings.TrimSpace(string(outMem)), "\n")
	if len(linesMem) == 0 || linesMem[0] == "" {
		return 0, nil, nil
	}

	totals := make([]float64, len(linesMem))
	for i, line := range linesMem {
		mb, _ := strconv.ParseFloat(strings.TrimSpace(line), 64)
		totals[i] = mb / 1024
	}

	// Get UUID map
	cmdUUID := exec.Command("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits")
	outUUID, err := cmdUUID.Output()
	if err != nil {
		return len(linesMem), totals, nil // Return what we have
	}
	linesUUID := strings.Split(strings.TrimSpace(string(outUUID)), "\n")
	uuidMap := make(map[string]int)
	for _, line := range linesUUID {
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			continue
		}
		index, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		uuid := strings.TrimSpace(parts[1])
		uuidMap[uuid] = index
	}

	return len(linesMem), totals, uuidMap
}

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
			Index:       i,
			Utilization: util,
			MemUsedGB:   memUsedMB / 1024,
		})
	}
	return usage
}

func updateProcessList(static *StaticInfo) []ProcessInfo {
	gpuProcessMap := getGPUProcessMap(static.GPUUUIDMap)
	procs, err := process.Processes()
	if err != nil {
		return nil
	}

	var processList []ProcessInfo
	for _, p := range procs {
		// Get host-relative CPU% first.
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

		// Calculate container-relative CPU %
		containerCPUPercent := 0.0
		if static.ContainerCPULimit > 0 {
			containerCPUPercent = cpuPercent / static.ContainerCPULimit
		}

		// Calculate container-relative Memory %
		containerMemPercent := 0.0
		if static.ContainerMemLimitBytes > 0 {
			containerMemPercent = (float64(memInfo.RSS) / float64(static.ContainerMemLimitBytes)) * 100
		}

		// Check if this process is on a GPU
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
			if static.GPUTotalGB[gpuInfo.GPUIndex] > 0 {
				// Calculate %GPUMEM based on total memory of the specific GPU
				pi.GPUMemPercent = (float64(gpuInfo.UsedMemMB) / (static.GPUTotalGB[gpuInfo.GPUIndex] * 1024)) * 100
			}
		}

		processList = append(processList, pi)
	}

	// Sort by host-relative CPU usage descending
	sort.Slice(processList, func(i, j int) bool {
		return processList[i].rawCPU > processList[j].rawCPU
	})

	return processList
}
