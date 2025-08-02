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
}

// GPUUsage holds live usage data for a single GPU.
type GPUUsage struct {
	Index       int
	Utilization int
	MemUsedGB   float64
}

// ProcessInfo holds information for a single monitored process.
type ProcessInfo struct {
	PID        int32
	User       string
	CPUPercent float64 // Container-relative
	MemPercent float64 // Container-relative
	Command    string
	rawCPU     float64 // Host-relative, for sorting
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
	info.GPUCount, info.GPUTotalGB = getInitialGPUInfo()

	return info, nil
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
	title := "---📊 Pod Resource & Process Monitor (press 'q' or Ctrl+C to quit)---"
	drawString(screen, 0, y, tcell.StyleDefault.Bold(true), truncateString(title, width))
	y++

	// --- Resource Panels ---
	y = drawResourcePanels(screen, y, width, state)

	// --- GPU Panel ---
	y = drawGPUPanel(screen, y, width, height, &state.static, &state.dynamic)

	// --- Process List ---
	if y < height {
		drawProcessList(screen, y, width, height, &state.dynamic)
	}

	screen.Show()
}

func drawResourcePanels(s tcell.Screen, y, width int, state *State) int {
	// CPU
	cpuLimitStr := fmt.Sprintf("(limit: %.2f CPUs)", state.static.ContainerCPULimit)
	if state.static.ContainerCPULimit == float64(state.static.HostCores) {
		cpuLimitStr = fmt.Sprintf("(no cgroup limit, %d host cores)", state.static.HostCores)
	}
	cpuStr := fmt.Sprintf("CPU Usage    : %.2f%% %s", state.dynamic.ContainerCPUUsage, cpuLimitStr)
	drawString(s, 0, y, tcell.StyleDefault, truncateString(cpuStr, width))
	y++

	// Memory
	memLimitStr := "N/A"
	if state.static.ContainerMemLimitGB > 0 {
		memLimitStr = fmt.Sprintf("%.3f GB", state.static.ContainerMemLimitGB)
	}
	memStr := fmt.Sprintf("Memory Usage : %.3f GB / %s", state.dynamic.ContainerMemUsedGB, memLimitStr)
	drawString(s, 0, y, tcell.StyleDefault, truncateString(memStr, width))
	y++

	return y
}

func drawGPUPanel(s tcell.Screen, y, width, height int, static *StaticInfo, dynamic *DynamicInfo) int {
	maxGPUSlots := 4

	header := strings.Repeat("-", (width-6)/2) + "GPUs" + strings.Repeat("-", (width-6)/2)
	drawString(s, 0, y, tcell.StyleDefault, truncateString(header, width))
	y++

	gpusDisplayed := 0
	if static.GPUCount > 0 {
		for i, gpu := range dynamic.LiveGPUUsage {
			if y >= height || gpusDisplayed >= maxGPUSlots {
				break
			}
			gpuStr := fmt.Sprintf("GPU #%d: Usage: %d%%   VRAM: %.2f GB / %.2f GB",
				gpu.Index, gpu.Utilization, gpu.MemUsedGB, static.GPUTotalGB[i])
			drawString(s, 0, y, tcell.StyleDefault, truncateString(gpuStr, width))
			y++
			gpusDisplayed++
		}
	} else if maxGPUSlots > 0 {
		if y < height {
			drawString(s, 0, y, tcell.StyleDefault, "No NVIDIA GPU detected")
			y++
			gpusDisplayed++
		}
	}

	// Pad with blank lines for consistent layout
	for i := gpusDisplayed; i < maxGPUSlots; i++ {
		if y >= height {
			break
		}
		y++
	}

	return y
}

func drawProcessList(s tcell.Screen, y, width, height int, dynamic *DynamicInfo) {
	// Column widths
	pidW, userW, cpuW, memW := 7, 12, 6, 6

	header := fmt.Sprintf("%-*s %-*s %*s %*s %s",
		pidW, "PID", userW, "USER", cpuW, "%CPU", memW, "%MEM", "COMMAND")
	drawString(s, 0, y, tcell.StyleDefault.Reverse(true), truncateString(header, width))
	y++

	for _, p := range dynamic.Processes {
		if y >= height {
			break
		}

		userStr := truncateString(p.User, userW-1)

		// Calculate available width for command
		cmdW := width - pidW - userW - cpuW - memW - 5 // 5 for spaces
		if cmdW < 1 {
			cmdW = 1
		}
		cmdStr := truncateString(p.Command, cmdW)

		line := fmt.Sprintf("%-*d %-*s %*.1f %*.1f %s",
			pidW, p.PID, userW, userStr, cpuW, p.CPUPercent, memW, p.MemPercent, cmdStr)
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

func getInitialGPUInfo() (int, []float64) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return 0, nil
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return 0, nil
	}

	totals := make([]float64, len(lines))
	for i, line := range lines {
		mb, _ := strconv.ParseFloat(strings.TrimSpace(line), 64)
		totals[i] = mb / 1024
	}
	return len(lines), totals
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

		processList = append(processList, ProcessInfo{
			PID:        p.Pid,
			User:       user,
			CPUPercent: containerCPUPercent,
			MemPercent: containerMemPercent,
			Command:    cmdline,
			rawCPU:     cpuPercent,
		})
	}

	// Sort by host-relative CPU usage descending
	sort.Slice(processList, func(i, j int) bool {
		return processList[i].rawCPU > processList[j].rawCPU
	})

	return processList
}
