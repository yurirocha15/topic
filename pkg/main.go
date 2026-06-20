// main.go
package main

import (
	"context"
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
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

// --- Constants ---

const (
	infoPanelWidth            = 55                // Fixed size of the information panel
	placeholderHeight         = 10                // Initial height of the top panel
	borderHeight              = 2                 // Border height for boxes
	minCPUInfoCount           = 2                 // For cgroup CPU quota/period
	minGPUUsageCount          = 2                 // For nvidia-smi GPU usage
	minGPUInfoCount           = 5                 // For nvidia-smi pmon
	barsPerGPU                = 2                 // GPU utilization and memory bars
	dynamicCollectorCount     = 7                 // Number of dynamic collectors in updateAll
	percentMultiplier         = 100.0             // Multiplier for percentage calculations
	bytesPerKB                = 1024              // Bytes per kilobyte
	bytesPerMB                = bytesPerKB * 1024 // Bytes per megabyte
	bytesPerGB                = bytesPerMB * 1024 // Bytes per gigabyte
	nanosecondsToMicroseconds = 1000              // Nanoseconds to microseconds
	secondToMicroseconds      = 1e6               // Seconds to microseconds
	minBarWidth               = 20                // Minimum width for a progress bar
	columnSpacing             = 5                 // Spacing between columns
	maxDisplayPathLength      = 15                // Maximum length for mount path display
	maxReadableColumns        = 4                 // Practical limit for readability
	maxGPUColumns             = 2                 // Maximum columns for GPU display
	maxDiskColumns            = 2                 // Maximum columns for Disk display
	spacesAroundBar           = 2                 // Number of spaces around bar (before and after)
	estimatedInfoWidth        = 25                // Estimated width for typical info text
	minLabelWidth             = 15                // Minimum width for labels (e.g., "DISK /:        ")
	percentageWidth           = 5                 // Width for percentage display (e.g., " 26.3%")
	minColumnWidth            = 70                // Minimum width required per column
	minTotalWidthForMultiCol  = 150               // Minimum total width to use multi-column
	commandTimeout            = 2 * time.Second   // Maximum time allowed for external probes
	cgroupMaxToken            = "max"             // Cgroup token for unlimited resources
	cgroupCPUMaxPath          = "/sys/fs/cgroup/cpu.max"
	cgroupCPUStatPath         = "/sys/fs/cgroup/cpu.stat"
	cgroupV1CPUQuotaPath      = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	cgroupV1CPUPeriodPath     = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
	cgroupV1CPUUsagePath      = "/sys/fs/cgroup/cpuacct/cpuacct.usage"
	cgroupMemoryMaxPath       = "/sys/fs/cgroup/memory.max"
	cgroupMemoryStatPath      = "/sys/fs/cgroup/memory.stat"
	cgroupV1MemoryLimitPath   = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	cgroupV1MemoryUsagePath   = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
)

// --- Interfaces ---

// FileReader defines an interface for reading files.
type FileReader interface {
	ReadFile(path string) ([]byte, error)
}

// OSFileReader is the default implementation of FileReader using the os package.
type OSFileReader struct{}

func (fs OSFileReader) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

// CommandRunner defines an interface for running external commands, allowing for mocking.
type CommandRunner interface {
	Output(name string, arg ...string) ([]byte, error)
}

// OSCommandRunner is the default implementation using the os/exec package.
type OSCommandRunner struct {
	Timeout time.Duration
}

func (ocr OSCommandRunner) Output(name string, arg ...string) ([]byte, error) {
	timeout := ocr.Timeout
	if timeout <= 0 {
		timeout = commandTimeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return exec.CommandContext(ctx, name, arg...).Output()
}

// Stater defines an interface for checking file status.
type Stater interface {
	Stat(path string) (os.FileInfo, error)
}

// OSStater is the default implementation of Stater using the os package.
type OSStater struct{}

func (s OSStater) Stat(path string) (os.FileInfo, error) {
	return os.Stat(path)
}

// --- Data Structures ---

// CgroupVersion denotes the cgroup version (1 or 2).
type CgroupVersion int

const (
	CgroupV1 CgroupVersion = 1
	CgroupV2 CgroupVersion = 2
)

// StaticInfo holds information that is fetched once at startup.
type StaticInfo struct {
	CgroupVersion          CgroupVersion  // Cgroup version (1 or 2)
	ContainerCPULimit      float64        // The container's CPU limit
	ContainerMemLimitBytes int64          // The container's Memory limit in bytes
	ContainerMemLimitGB    float64        // The container's Memory limit in gigabytes
	HostCores              int            // Number of cores in the host
	HostMemTotalGB         float64        // Host's total memory in gigabytes
	GPUCount               int            // Number of GPUs available
	GPUTotalGB             []float64      // Total memory available on each GPU in gigabytes
	StorageMounts          []StorageMount // Information about mounted filesystems
}

// GPUUsage holds live usage data for a single GPU.
type GPUUsage struct {
	Index       int     // GPU index
	Utilization int     // GPU utilization percentage
	MemUsedGB   float64 // Current memory usage in gigabytes
}

// StorageMount holds information about a mounted filesystem.
type StorageMount struct {
	Path    string  // Mount path (e.g., "/", "/home")
	TotalGB float64 // Total capacity in GB
	Fstype  string  // Filesystem type
}

// StorageUsage holds current usage data for a mounted filesystem.
type StorageUsage struct {
	Path        string  // Mount path
	UsedGB      float64 // Used space in GB
	FreeGB      float64 // Free space in GB
	UsedPercent float64 // Usage percentage
}

// BarLayout holds information about how bars should be arranged.
type BarLayout struct {
	Columns       int // Number of columns
	BarWidth      int // Width of each bar
	TotalWidth    int // Total width used
	MaxLabelWidth int // Maximum label width for alignment
	MaxInfoWidth  int // Maximum info width for alignment
}

// BarData holds information for rendering a single bar.
type BarData struct {
	Label      string  // Left side label
	LabelWidth int     // Visible width of the label
	Percent    float64 // Percentage for the bar
	Info       string  // Right side info text
	InfoWidth  int     // Visible width of the info text
}

// GPUProcessInfo holds usage data for a process on a GPU.
type GPUProcessInfo struct {
	GPUIndex   int    // GPU index
	UsedMemMB  uint64 // Current memory used by the process in megabytes
	GPUUtil    uint64 // GPU utilization percentage
	GPUMemUtil uint64 // GPU memory utilization percentage
}

// ProcessInfo holds information for a single monitored process.
type ProcessInfo struct {
	PID           int32   // Process ID
	User          string  // User who owns the process
	CPUPercent    float64 // CPU usage percentage
	MemPercent    float64 // Memory usage percentage
	Command       string  // Command that started the process
	rawCPU        float64 // Raw CPU usage
	GPUIndex      int     // GPU index
	GPUUtil       uint64  // GPU utilization percentage
	GPUMemPercent float64 // GPU memory usage percentage
}

// DynamicInfo holds all the data that is refreshed on each tick.
type DynamicInfo struct {
	mu                 sync.Mutex     // Mutex to protect concurrent access
	ContainerCPUUsage  float64        // Current CPU usage in the container
	ContainerMemUsedGB float64        // Current memory usage in the container
	HostCPUUsage       float64        // Current CPU usage on the host
	HostMemUsedGB      float64        // Current memory usage on the host
	LiveGPUUsage       []GPUUsage     // Current GPU usage
	StorageUsage       []StorageUsage // Current storage usage
	Processes          []ProcessInfo  // Current processes
}

// State holds the application's entire state.
type State struct {
	static       StaticInfo  // Static info retrieved only once
	dynamic      DynamicInfo // Dynamic info refreshed on each tick
	prevCPUUsage uint64      // CPU usage in the last tick
	prevCPUTime  time.Time   // Time of the last CPU usage measurement
}

// --- Main Application ---

func main() {
	fileReader := OSFileReader{}
	cmdRunner := OSCommandRunner{Timeout: commandTimeout}
	stater := OSStater{}

	staticInfo, err := getStaticInfo(fileReader, cmdRunner, stater)
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
		AddItem(resourceView, 0, 1, false).         // Resources take up remaining width
		AddItem(infoView, infoPanelWidth, 0, false) // Info panel has a fixed width

	topPanel.SetBorder(true).SetTitle(" System Information ")

	// Main layout combines top panel and process table
	mainLayout := tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(topPanel, placeholderHeight, 0, false). // Top panel has a placeholder height, will be resized
		AddItem(processTable, 0, 1, true)               // Process table takes all remaining vertical space

	// --- Goroutine for periodic updates ---
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			<-ticker.C
			updateAll(state, fileReader, cmdRunner)
			app.QueueUpdateDraw(func() {
				// Update all components and dynamically resize the top panel
				leftHeight := updateResourceView(resourceView, state)
				rightHeight := updateInfoView(infoView)
				topPanelHeight := int(math.Max(float64(leftHeight), float64(rightHeight))) + borderHeight
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
	updateAll(state, fileReader, cmdRunner)
	leftHeight := updateResourceView(resourceView, state)
	rightHeight := updateInfoView(infoView)
	topPanelHeight := int(math.Max(float64(leftHeight), float64(rightHeight))) + borderHeight
	mainLayout.ResizeItem(topPanel, topPanelHeight, 0)
	updateProcessTable(processTable, state)

	if err = app.SetRoot(mainLayout, true).Run(); err != nil {
		log.Fatalf("Could not start application: %v", err)
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
		columnIdx := 0
		table.SetCell(r+1, columnIdx, tview.NewTableCell(strconv.Itoa(int(p.PID))).SetTextColor(tcell.ColorWhite))
		// USER
		columnIdx++
		table.SetCell(r+1, columnIdx, tview.NewTableCell(p.User).SetTextColor(tcell.ColorGreen))
		// %CPU
		columnIdx++
		cpuCell := tview.NewTableCell(strconv.FormatFloat(p.CPUPercent, 'f', 1, 64)).
			SetTextColor(tcell.ColorAqua)
		table.SetCell(r+1, columnIdx, cpuCell)
		// %MEM
		columnIdx++
		memCell := tview.NewTableCell(strconv.FormatFloat(p.MemPercent, 'f', 1, 64)).
			SetTextColor(tcell.ColorAqua)
		table.SetCell(r+1, columnIdx, memCell)

		// %GPU and %GPUMEM
		columnIdx++
		if p.GPUIndex != -1 {
			gpuUtilCell := tview.NewTableCell(strconv.FormatUint(p.GPUUtil, 10)).
				SetTextColor(tcell.ColorFuchsia)
			table.SetCell(r+1, columnIdx, gpuUtilCell)
			columnIdx++
			gpuMemCell := tview.NewTableCell(strconv.FormatFloat(p.GPUMemPercent, 'f', 1, 64)).
				SetTextColor(tcell.ColorFuchsia)
			table.SetCell(r+1, columnIdx, gpuMemCell)
		} else {
			table.SetCell(r+1, columnIdx, tview.NewTableCell("-").SetTextColor(tcell.ColorDarkGray))
			columnIdx++
			table.SetCell(r+1, columnIdx, tview.NewTableCell("-").SetTextColor(tcell.ColorDarkGray))
		}

		// COMMAND
		cmdCell := tview.NewTableCell(p.Command).
			SetTextColor(tcell.ColorWhite).
			SetExpansion(1).
			SetMaxWidth(0) // Prevent truncation
		columnIdx++
		table.SetCell(r+1, columnIdx, cmdCell)
	}
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

// --- Host System Functions ---

// updateHostCPUUsage fetches current host CPU usage percentage.
func updateHostCPUUsage() float64 {
	if percentages, err := cpu.Percent(0, false); err == nil && len(percentages) > 0 {
		return percentages[0]
	}
	return 0
}

// updateHostMemUsage fetches current host memory usage in GB.
func updateHostMemUsage() float64 {
	if memInfo, err := mem.VirtualMemory(); err == nil {
		return float64(memInfo.Used) / float64(bytesPerGB)
	}
	return 0
}

// --- Data Fetching Functions ---

// updateAll fetches all dynamic data and updates the state.
func updateAll(state *State, fs FileReader, runner CommandRunner) {
	staticInfo := state.static
	var dynamic DynamicInfo
	var wg sync.WaitGroup
	wg.Add(dynamicCollectorCount)
	go func() { defer wg.Done(); dynamic.ContainerCPUUsage = updateContainerCPUUsage(state, fs) }()
	go func() {
		defer wg.Done()
		dynamic.ContainerMemUsedGB = updateContainerMemUsage(staticInfo.CgroupVersion, fs)
	}()
	go func() {
		defer wg.Done()
		dynamic.LiveGPUUsage = updateLiveGPUUsage(staticInfo.GPUCount, runner)
	}()
	go func() {
		defer wg.Done()
		dynamic.StorageUsage = updateStorageUsage(staticInfo.StorageMounts)
	}()
	go func() { defer wg.Done(); dynamic.Processes = updateProcessList(&staticInfo, runner) }()
	go func() { defer wg.Done(); dynamic.HostCPUUsage = updateHostCPUUsage() }()
	go func() { defer wg.Done(); dynamic.HostMemUsedGB = updateHostMemUsage() }()
	wg.Wait()

	state.dynamic.mu.Lock()
	state.dynamic.ContainerCPUUsage = dynamic.ContainerCPUUsage
	state.dynamic.ContainerMemUsedGB = dynamic.ContainerMemUsedGB
	state.dynamic.LiveGPUUsage = dynamic.LiveGPUUsage
	state.dynamic.StorageUsage = dynamic.StorageUsage
	state.dynamic.Processes = dynamic.Processes
	state.dynamic.HostCPUUsage = dynamic.HostCPUUsage
	state.dynamic.HostMemUsedGB = dynamic.HostMemUsedGB
	state.dynamic.mu.Unlock()
}

// getStaticInfo fetches static information about the container and host.
func getStaticInfo(fs FileReader, runner CommandRunner, stater Stater) (StaticInfo, error) {
	var info StaticInfo
	var err error
	if _, err = stater.Stat(cgroupCPUMaxPath); os.IsNotExist(err) {
		info.CgroupVersion = CgroupV1
	} else {
		info.CgroupVersion = CgroupV2
	}
	info.HostCores, err = cpu.Counts(false)
	if err != nil {
		info.HostCores = 1
	}
	info.ContainerCPULimit = getContainerCPULimit(info.CgroupVersion, info.HostCores, fs)
	info.ContainerMemLimitBytes, info.ContainerMemLimitGB = getContainerMemLimit(info.CgroupVersion, fs)
	hostMem, hostMemErr := mem.VirtualMemory()
	if hostMemErr == nil {
		info.HostMemTotalGB = float64(hostMem.Total) / float64(bytesPerGB)
	} else {
		info.HostMemTotalGB = 0
	}
	info.GPUCount, info.GPUTotalGB = getStaticGPUInfo(runner)
	info.StorageMounts = getStaticStorageInfo()
	return info, err
}

// readUintFromFile reads a uint64 value from a file, trimming whitespace.
func readUintFromFile(path string, fs FileReader) (uint64, error) {
	data, err := fs.ReadFile(path)
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
func readStringFromFile(path string, fs FileReader) (string, error) {
	data, err := fs.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

// getContainerCPULimit reads the CPU limit from cgroup files and returns it as a percentage of host cores.
func getContainerCPULimit(cgroupVersion CgroupVersion, hostCores int, fs FileReader) float64 {
	var quota, period uint64
	if cgroupVersion == CgroupV2 {
		cpuMaxStr, err := readStringFromFile(cgroupCPUMaxPath, fs)
		if err != nil {
			return float64(hostCores)
		}
		parts := strings.Fields(cpuMaxStr)
		if len(parts) == minCPUInfoCount {
			if parts[0] == cgroupMaxToken {
				return float64(hostCores)
			}
			quota, err = strconv.ParseUint(parts[0], 10, 64)
			if err != nil {
				return float64(hostCores)
			}
			period, err = strconv.ParseUint(parts[1], 10, 64)
			if err != nil {
				return float64(hostCores)
			}
		}
	} else { // CgroupV1
		q, err1 := readUintFromFile(cgroupV1CPUQuotaPath, fs)
		p, err2 := readUintFromFile(cgroupV1CPUPeriodPath, fs)
		if err1 == nil && err2 == nil {
			quota, period = q, p
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
func getContainerMemLimit(cgroupVersion CgroupVersion, fs FileReader) (int64, float64) {
	var path string
	if cgroupVersion == CgroupV2 {
		path = cgroupMemoryMaxPath
	} else {
		path = cgroupV1MemoryLimitPath
	}
	limitStr, err := readStringFromFile(path, fs)
	if err != nil || limitStr == cgroupMaxToken {
		return 0, 0
	}
	limitBytes, err := strconv.ParseInt(limitStr, 10, 64)
	if err != nil || limitBytes <= 0 {
		return 0, 0
	}
	return limitBytes, float64(limitBytes) / float64(bytesPerGB)
}

// updateContainerCPUUsage reads the CPU usage from the cgroup filesystem.
func updateContainerCPUUsage(state *State, fs FileReader) float64 {
	var currentUsage uint64

	if state.static.CgroupVersion == CgroupV2 {
		statStr, err := readStringFromFile(cgroupCPUStatPath, fs)
		if err != nil {
			return 0
		}
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "usage_usec" {
				currentUsage, err = strconv.ParseUint(parts[1], 10, 64)
				if err != nil {
					return 0
				}
				break
			}
		}
	} else { // CgroupV1
		// value is in nanoseconds, convert to microseconds
		ns, err := readUintFromFile(cgroupV1CPUUsagePath, fs)
		if err == nil {
			currentUsage = ns / nanosecondsToMicroseconds
		}
	}

	now := time.Now()
	timeDelta := now.Sub(state.prevCPUTime).Seconds()

	var cpuPercent float64
	if timeDelta > 0 && currentUsage > state.prevCPUUsage && state.static.ContainerCPULimit > 0 {
		usageDelta := float64(currentUsage - state.prevCPUUsage) // in microseconds
		// usageDelta is how many microseconds of CPU time were used.
		// timeDelta * 1e6 is how many microseconds passed in wall-clock time.
		// Normalize by the number of cores to get a percentage of the container's total capacity.
		cpuPercent = (usageDelta / (timeDelta * secondToMicroseconds)) * percentMultiplier / state.static.ContainerCPULimit
	}

	state.prevCPUUsage = currentUsage
	state.prevCPUTime = now

	if cpuPercent > percentMultiplier {
		return percentMultiplier
	}
	return cpuPercent
}

// updateContainerMemUsage reads the memory usage from the cgroup filesystem.
func updateContainerMemUsage(cgroupVersion CgroupVersion, fs FileReader) float64 {
	if cgroupVersion == CgroupV2 {
		// 'anon' memory is a good proxy for non-cache process memory
		statStr, err := readStringFromFile(cgroupMemoryStatPath, fs)
		if err != nil {
			return 0
		}
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "anon" {
				bytes, parseErr := strconv.ParseUint(parts[1], 10, 64)
				if parseErr != nil {
					return 0
				}
				return float64(bytes) / float64(bytesPerGB)
			}
		}
	} else { // CgroupV1
		// memory.usage_in_bytes includes file cache, but is the standard.
		bytes, err := readUintFromFile(cgroupV1MemoryUsagePath, fs)
		if err == nil {
			return float64(bytes) / float64(bytesPerGB)
		}
	}
	return 0
}

// getStaticStorageInfo fetches mounted filesystem information.
func getStaticStorageInfo() []StorageMount {
	partitions, err := disk.Partitions(false)
	if err != nil {
		return nil
	}

	mounts := make([]StorageMount, 0, len(partitions))
	for _, partition := range partitions {
		// Skip virtual filesystems and temporary mounts
		if shouldSkipFilesystem(partition.Fstype, partition.Mountpoint) {
			continue
		}

		usage, usageErr := disk.Usage(partition.Mountpoint)
		if usageErr != nil {
			continue
		}

		mount := StorageMount{
			Path:    partition.Mountpoint,
			TotalGB: float64(usage.Total) / float64(bytesPerGB),
			Fstype:  partition.Fstype,
		}
		mounts = append(mounts, mount)
	}

	return mounts
}

// shouldSkipFilesystem determines if a filesystem should be skipped from monitoring.
func shouldSkipFilesystem(fstype, mountpoint string) bool {
	// Skip virtual/temporary filesystems
	skipFsTypes := []string{
		"tmpfs", "devtmpfs", "sysfs", "proc", "devpts", "cgroup", "cgroup2",
		"pstore", "bpf", "debugfs", "tracefs", "securityfs", "fusectl",
		"configfs", "selinuxfs", "mqueue", "hugetlbfs", "autofs", "rpc_pipefs",
		"squashfs", "overlayfs",
	}

	for _, skip := range skipFsTypes {
		if fstype == skip {
			return true
		}
	}

	// Skip system mount points
	skipPaths := []string{
		"/dev", "/sys", "/proc", "/run", "/tmp", "/var/run", "/var/lock",
	}

	for _, skip := range skipPaths {
		if strings.HasPrefix(mountpoint, skip) {
			return true
		}
	}

	// Skip loop devices (often used by snap packages and other virtual filesystems)
	if strings.Contains(mountpoint, "/loop") || strings.HasPrefix(mountpoint, "/snap/") {
		return true
	}

	return false
}

// updateStorageUsage fetches current storage usage for all monitored mounts.
func updateStorageUsage(storageMounts []StorageMount) []StorageUsage {
	if len(storageMounts) == 0 {
		return nil
	}

	usage := make([]StorageUsage, 0, len(storageMounts))
	for _, mount := range storageMounts {
		stat, err := disk.Usage(mount.Path)
		if err != nil {
			continue
		}

		usage = append(usage, StorageUsage{
			Path:        mount.Path,
			UsedGB:      float64(stat.Used) / float64(bytesPerGB),
			FreeGB:      float64(stat.Free) / float64(bytesPerGB),
			UsedPercent: stat.UsedPercent,
		})
	}

	return usage
}

// getStaticGPUInfo fetches total memory for each GPU.
func getStaticGPUInfo(runner CommandRunner) (int, []float64) {
	outMem, err := runner.Output("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
	if err != nil {
		return 0, nil
	}
	linesMem := strings.Split(strings.TrimSpace(string(outMem)), "\n")
	if len(linesMem) == 0 || linesMem[0] == "" {
		return 0, nil
	}
	totals := make([]float64, 0, len(linesMem))
	for _, line := range linesMem {
		mb, parseErr := strconv.ParseFloat(strings.TrimSpace(line), 64)
		if parseErr != nil {
			continue
		}
		totals = append(totals, mb/bytesPerKB) // Convert MB to GB
	}
	return len(totals), totals
}

// updateLiveGPUUsage fetches current GPU utilization and memory usage.
func updateLiveGPUUsage(gpuCount int, runner CommandRunner) []GPUUsage {
	if gpuCount == 0 {
		return nil
	}
	out, err := runner.Output("nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits")
	if err != nil {
		return nil
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	usage := make([]GPUUsage, 0, len(lines))
	for i, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) != minGPUUsageCount {
			continue
		}
		util, parseErr := strconv.Atoi(strings.TrimSpace(parts[0]))
		if parseErr != nil {
			continue
		}
		memUsedMB, parseErr := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if parseErr != nil {
			continue
		}
		usage = append(usage, GPUUsage{Index: i, Utilization: util, MemUsedGB: memUsedMB / bytesPerKB})
	}
	return usage
}

// getGPUProcessMap queries nvidia-smi pmon for apps running on the GPU and maps their PID to usage.
func getGPUProcessMap(runner CommandRunner) map[int32]GPUProcessInfo {
	// Use `pmon` with `-c 1` to get a single snapshot, and `-s um` for utilization and memory.
	out, err := runner.Output("nvidia-smi", "pmon", "-c", "1", "-s", "um")
	if err != nil {
		return nil
	}
	lines := strings.Split(string(out), "\n")
	processMap := make(map[int32]GPUProcessInfo, len(lines))
	for _, line := range lines {
		if strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < minGPUInfoCount {
			continue
		}
		var pid int64
		pid, err = strconv.ParseInt(fields[1], 10, 32)
		if err != nil {
			continue
		}
		gpuIndex, parseErr := strconv.Atoi(fields[0])
		if parseErr != nil {
			continue
		}
		gpuUtil, parseErr := parseGPUPercentField(fields[3])
		if parseErr != nil {
			continue
		}
		memUtil, parseErr := parseGPUPercentField(fields[4])
		if parseErr != nil {
			continue
		}
		processMap[int32(pid)] = GPUProcessInfo{GPUIndex: gpuIndex, GPUUtil: gpuUtil, GPUMemUtil: memUtil}
	}
	return processMap
}

func parseGPUPercentField(value string) (uint64, error) {
	if value == "-" {
		return 0, nil
	}
	return strconv.ParseUint(value, 10, 64)
}

// updateProcessList fetches the current process list and adds resource usage info when possible.
func updateProcessList(static *StaticInfo, runner CommandRunner) []ProcessInfo {
	var gpuProcessMap map[int32]GPUProcessInfo
	if static.GPUCount > 0 {
		gpuProcessMap = getGPUProcessMap(runner)
	}
	procs, err := process.Processes()
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
	p *process.Process,
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
		PID:        p.Pid,
		User:       user,
		CPUPercent: containerCPUPercent,
		MemPercent: containerMemPercent,
		Command:    cmdline,
		rawCPU:     cpuPercent,
		GPUIndex:   -1,
	}
	if gpuInfo, onGPU := gpuProcessMap[p.Pid]; onGPU {
		pi.GPUIndex = gpuInfo.GPUIndex
		pi.GPUUtil = gpuInfo.GPUUtil
		pi.GPUMemPercent = float64(gpuInfo.GPUMemUtil)
	}
	return pi, true
}
