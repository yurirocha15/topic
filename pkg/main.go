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
	infoPanelWidth            = 55
	placeholderHeight         = 10
	borderHeight              = 2
	minCPUInfoCount           = 2 // For cgroup CPU quota/period
	minGPUUsageCount          = 2 // For nvidia-smi GPU usage
	minGPUInfoCount           = 5 // For nvidia-smi pmon
	waitDelta                 = 7
	percentMultiplier         = 100.0
	bytesPerKB                = 1024
	bytesPerMB                = bytesPerKB * 1024
	bytesPerGB                = bytesPerMB * 1024
	nanosecondsToMicroseconds = 1000
	secondToMicroseconds      = 1e6
	minBarWidth               = 20 // Minimum width for a progress bar
	columnSpacing             = 5  // Spacing between columns
	maxDisplayPathLength      = 15 // Maximum length for mount path display
	maxReadableColumns        = 4  // Practical limit for readability
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
type OSCommandRunner struct{}

func (ocr OSCommandRunner) Output(name string, arg ...string) ([]byte, error) {
	return exec.CommandContext(context.Background(), name, arg...).Output()
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
	CgroupVersion          CgroupVersion
	ContainerCPULimit      float64
	ContainerMemLimitBytes int64
	ContainerMemLimitGB    float64
	HostCores              int
	HostMemTotalGB         float64
	GPUCount               int
	GPUTotalGB             []float64
	StorageMounts          []StorageMount
}

// GPUUsage holds live usage data for a single GPU.
type GPUUsage struct {
	Index       int
	Utilization int
	MemUsedGB   float64
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
	Columns    int // Number of columns
	BarWidth   int // Width of each bar
	TotalWidth int // Total width used
}

// BarData holds information for rendering a single bar.
type BarData struct {
	Label   string  // Left side label
	Percent float64 // Percentage for the bar
	Info    string  // Right side info text
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
	HostCPUUsage       float64
	HostMemUsedGB      float64
	LiveGPUUsage       []GPUUsage
	StorageUsage       []StorageUsage
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
	fileReader := OSFileReader{}
	cmdRunner := OSCommandRunner{}
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

	// Get the available width inside the view, minus padding.
	_, _, availableWidth, _ := view.GetInnerRect()
	availableWidth -= borderHeight // Account for horizontal padding within the box

	var builder strings.Builder

	// Add CPU section
	builder.WriteString(buildCPUSection(state, availableWidth))

	// Add Memory section
	builder.WriteString(buildMemorySection(state, availableWidth))

	// Add Storage section
	if len(state.dynamic.StorageUsage) > 0 {
		builder.WriteString(buildStorageSection(state, availableWidth))
	}

	// Add GPU section
	if state.static.GPUCount > 0 {
		builder.WriteString(buildGPUSection(state, availableWidth))
	}

	finalText := builder.String()
	view.SetText(finalText)
	return strings.Count(finalText, "\n")
}

// buildCPUSection creates the CPU usage display section.
func buildCPUSection(state *State, availableWidth int) string {
	var cpuUsage float64
	var cpuLabel, cpuInfo string

	if state.static.ContainerCPULimit == float64(state.static.HostCores) {
		// Running outside container or no cgroup limit - use host metrics
		cpuUsage = state.dynamic.HostCPUUsage
		cpuLabel = fmt.Sprintf("CPU: [yellow]%-6.1f%%[white] ", cpuUsage)
		cpuInfo = fmt.Sprintf(" [darkcyan](no cgroup limit, %d host cores)[white]", state.static.HostCores)
	} else {
		// Running inside container with limits
		cpuUsage = state.dynamic.ContainerCPUUsage
		cpuLabel = fmt.Sprintf("CPU: [yellow]%-6.1f%%[white] ", cpuUsage)
		cpuInfo = fmt.Sprintf(" [darkcyan](limit: %.2f CPUs)[white]", state.static.ContainerCPULimit)
	}

	// Calculate the width for the bar by subtracting the label and info text lengths.
	cpuBarWidth := availableWidth - tview.TaggedStringWidth(cpuLabel) - tview.TaggedStringWidth(cpuInfo)
	return cpuLabel + makeBar(cpuUsage, cpuBarWidth) + cpuInfo + "\n"
}

// buildMemorySection creates the memory usage display section.
func buildMemorySection(state *State, availableWidth int) string {
	var memPercent float64
	var memLabel, memInfo string

	if state.static.ContainerMemLimitGB == 0 {
		// Running outside container or no memory limit - use host metrics
		if state.static.HostMemTotalGB > 0 {
			memPercent = (state.dynamic.HostMemUsedGB / state.static.HostMemTotalGB) * percentMultiplier
		}
		memLabel = fmt.Sprintf("MEM: [yellow]%-6.1f%%[white] ", memPercent)
		memInfo = fmt.Sprintf(
			" [darkcyan]%.3f GB / %.3f GB (no cgroup limit)[white]",
			state.dynamic.HostMemUsedGB,
			state.static.HostMemTotalGB,
		)
	} else {
		// Running inside container with limits
		memPercent = (state.dynamic.ContainerMemUsedGB * bytesPerGB / float64(state.static.ContainerMemLimitBytes)) * percentMultiplier
		memLabel = fmt.Sprintf("MEM: [yellow]%-6.1f%%[white] ", memPercent)
		memInfo = fmt.Sprintf(" [darkcyan]%.3f GB / %.3f GB[white]", state.dynamic.ContainerMemUsedGB, state.static.ContainerMemLimitGB)
	}
	memBarWidth := availableWidth - tview.TaggedStringWidth(memLabel) - tview.TaggedStringWidth(memInfo)
	return memLabel + makeBar(memPercent, memBarWidth) + memInfo + "\n"
}

// buildStorageSection creates the storage usage display section.
func buildStorageSection(state *State, availableWidth int) string {
	var builder strings.Builder
	builder.WriteString("\n")

	// Collect storage bars data
	var storageBars []BarData
	for _, storage := range state.dynamic.StorageUsage {
		// Shorten long mount paths for display
		displayPath := storage.Path
		if len(displayPath) > maxDisplayPathLength {
			displayPath = "..." + displayPath[len(displayPath)-12:]
		}

		storageBars = append(storageBars, BarData{
			Label:   fmt.Sprintf("DISK %s: [yellow]%-6.1f%%[white]", displayPath, storage.UsedPercent),
			Percent: storage.UsedPercent,
			Info:    fmt.Sprintf("[darkcyan]%.2f GB / %.2f GB[white]", storage.UsedGB, storage.UsedGB+storage.FreeGB),
		})
	}

	// Calculate layout for storage bars
	storageLayout := calculateBarLayout(availableWidth, len(storageBars))

	// Render storage bars using multi-column layout
	if storageLayout.Columns == 1 {
		// Single column - render with labels and info
		for _, bar := range storageBars {
			labelInfoWidth := tview.TaggedStringWidth(bar.Label) + tview.TaggedStringWidth(" "+bar.Info)
			barWidth := availableWidth - labelInfoWidth
			if barWidth < minBarWidth {
				barWidth = minBarWidth
			}
			builder.WriteString(bar.Label + " " + makeBar(bar.Percent, barWidth) + " " + bar.Info + "\n")
		}
	} else {
		// Multi-column layout - render complete bars side by side with proper alignment
		barRows := makeAlignedMultiColumnBars(storageBars, storageLayout)
		for _, row := range barRows {
			builder.WriteString(row + "\n")
		}
	}

	return builder.String()
}

// buildGPUSection creates the GPU usage display section.
func buildGPUSection(state *State, availableWidth int) string {
	var builder strings.Builder
	builder.WriteString("\n")

	// Collect GPU bars data
	var gpuBars []BarData
	for i, gpu := range state.dynamic.LiveGPUUsage {
		// GPU Utilization
		gpuBars = append(gpuBars, BarData{
			Label:   fmt.Sprintf("GPU%d Util: [yellow]%-3d%%[white]", i, gpu.Utilization),
			Percent: float64(gpu.Utilization),
			Info:    "",
		})

		// GPU Memory
		gpuMemPercent := 0.0
		if state.static.GPUTotalGB[i] > 0 {
			gpuMemPercent = (gpu.MemUsedGB / state.static.GPUTotalGB[i]) * percentMultiplier
		}
		gpuBars = append(gpuBars, BarData{
			Label:   fmt.Sprintf("GPU%d Mem:  [yellow]%-3.0f%%[white]", i, gpuMemPercent),
			Percent: gpuMemPercent,
			Info:    fmt.Sprintf("[darkcyan]%.2f GB / %.2f GB[white]", gpu.MemUsedGB, state.static.GPUTotalGB[i]),
		})
	}

	// Calculate layout for GPU bars (limited to 1-2 columns)
	gpuLayout := calculateGPUBarLayout(availableWidth, len(gpuBars))

	// Render GPU bars using multi-column layout
	if gpuLayout.Columns == 1 {
		// Single column - render with labels and info
		for _, bar := range gpuBars {
			labelInfoWidth := tview.TaggedStringWidth(bar.Label) + tview.TaggedStringWidth(" "+bar.Info)
			barWidth := availableWidth - labelInfoWidth
			if barWidth < minBarWidth {
				barWidth = minBarWidth
			}
			builder.WriteString(bar.Label + " " + makeBar(bar.Percent, barWidth) + " " + bar.Info + "\n")
		}
	} else {
		// Multi-column layout - render complete bars side by side with proper alignment
		barRows := makeAlignedMultiColumnBars(gpuBars, gpuLayout)
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

// calculateBarLayout determines the optimal column arrangement for bars.
func calculateBarLayout(availableWidth, numBars int) BarLayout {
	if availableWidth < minBarWidth || numBars <= 0 {
		return BarLayout{Columns: 1, BarWidth: minBarWidth, TotalWidth: minBarWidth}
	}

	// Estimate space needed for text per bar
	// Labels are typically around 20-25 chars, info text around 20-25 chars
	estimatedTextWidth := 50 // Conservative estimate for label + info + spacing

	// Try different numbers of columns, starting from the maximum possible
	maxColumns := numBars
	if maxColumns > maxReadableColumns { // Practical limit for readability
		maxColumns = maxReadableColumns
	}

	for columns := maxColumns; columns >= 1; columns-- {
		// Calculate space needed for spacing between columns
		spacingWidth := (columns - 1) * columnSpacing

		// Calculate available width per column
		availableWidthPerColumn := (availableWidth - spacingWidth) / columns

		// Each column needs space for text + bar
		barWidth := availableWidthPerColumn - estimatedTextWidth

		if barWidth >= minBarWidth {
			totalUsedWidth := columns*(barWidth+estimatedTextWidth) + spacingWidth
			return BarLayout{
				Columns:    columns,
				BarWidth:   barWidth,
				TotalWidth: totalUsedWidth,
			}
		}
	}

	// Fallback to single column with minimum width
	return BarLayout{Columns: 1, BarWidth: minBarWidth, TotalWidth: minBarWidth}
}

// calculateGPUBarLayout determines the optimal column arrangement for GPU bars (max 2 columns).
func calculateGPUBarLayout(availableWidth, numBars int) BarLayout {
	if availableWidth < minBarWidth || numBars <= 0 {
		return BarLayout{Columns: 1, BarWidth: minBarWidth, TotalWidth: minBarWidth}
	}

	// Estimate space needed for text per bar (GPU bars have shorter text)
	estimatedTextWidth := 40 // GPU labels are shorter

	// GPU bars are limited to 1 or 2 columns for better readability
	maxColumns := 2
	if numBars == 1 {
		maxColumns = 1
	}

	for columns := maxColumns; columns >= 1; columns-- {
		// Calculate space needed for spacing between columns
		spacingWidth := (columns - 1) * columnSpacing

		// Calculate available width per column
		availableWidthPerColumn := (availableWidth - spacingWidth) / columns

		// Each column needs space for text + bar
		barWidth := availableWidthPerColumn - estimatedTextWidth

		if barWidth >= minBarWidth {
			totalUsedWidth := columns*(barWidth+estimatedTextWidth) + spacingWidth
			return BarLayout{
				Columns:    columns,
				BarWidth:   barWidth,
				TotalWidth: totalUsedWidth,
			}
		}
	}

	// Fallback to single column with minimum width
	return BarLayout{Columns: 1, BarWidth: minBarWidth, TotalWidth: minBarWidth}
}

// makeAlignedMultiColumnBars creates properly aligned bars in columns with consistent spacing.
func makeAlignedMultiColumnBars(bars []BarData, layout BarLayout) []string {
	if len(bars) == 0 {
		return nil
	}

	numRows := (len(bars) + layout.Columns - 1) / layout.Columns
	maxLabelWidth, maxInfoWidth := calculateMaxWidths(bars)

	var result []string
	for row := range numRows {
		rowContent := buildAlignedRow(bars, layout, row, maxLabelWidth, maxInfoWidth)
		result = append(result, rowContent)
	}

	return result
}

// calculateMaxWidths finds the maximum label and info widths across all bars.
func calculateMaxWidths(bars []BarData) (int, int) {
	maxLabelWidth := 0
	maxInfoWidth := 0

	for _, bar := range bars {
		labelWidth := tview.TaggedStringWidth(bar.Label)
		if labelWidth > maxLabelWidth {
			maxLabelWidth = labelWidth
		}

		infoWidth := tview.TaggedStringWidth(bar.Info)
		if infoWidth > maxInfoWidth {
			maxInfoWidth = infoWidth
		}
	}

	return maxLabelWidth, maxInfoWidth
}

// buildAlignedRow creates a single row of aligned bars.
func buildAlignedRow(bars []BarData, layout BarLayout, row, maxLabelWidth, maxInfoWidth int) string {
	var completeBars []string

	for col := range layout.Columns {
		barIndex := row*layout.Columns + col
		if barIndex >= len(bars) {
			// Fill empty columns with spaces
			emptyWidth := maxLabelWidth + 1 + layout.BarWidth + 1 + maxInfoWidth
			completeBars = append(completeBars, strings.Repeat(" ", emptyWidth))
			continue
		}

		barContent := formatAlignedBar(bars[barIndex], layout.BarWidth, maxLabelWidth, maxInfoWidth)
		completeBars = append(completeBars, barContent)
	}

	// Join columns with spacing
	return strings.Join(completeBars, strings.Repeat(" ", columnSpacing))
}

// formatAlignedBar formats a single bar with consistent alignment.
func formatAlignedBar(bar BarData, barWidth, maxLabelWidth, maxInfoWidth int) string {
	// Pad label to consistent width
	paddedLabel := bar.Label
	labelPadding := maxLabelWidth - tview.TaggedStringWidth(bar.Label)
	if labelPadding > 0 {
		paddedLabel += strings.Repeat(" ", labelPadding)
	}

	// Create the bar
	barContent := makeBar(bar.Percent, barWidth)

	// Pad info to consistent width (if info exists)
	if bar.Info != "" {
		paddedInfo := bar.Info
		infoPadding := maxInfoWidth - tview.TaggedStringWidth(bar.Info)
		if infoPadding > 0 {
			paddedInfo += strings.Repeat(" ", infoPadding)
		}
		// Complete bar: label + space + bar + space + info
		return paddedLabel + " " + barContent + " " + paddedInfo
	}

	// No info, just label + bar
	return paddedLabel + " " + barContent
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
	bar := strings.Repeat("▓", filledWidth) + strings.Repeat("░", barWidth-filledWidth)
	return fmt.Sprintf("[green]%s[white]", bar)
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
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()
	var wg sync.WaitGroup
	wg.Add(waitDelta)
	go func() { defer wg.Done(); state.dynamic.ContainerCPUUsage = updateContainerCPUUsage(state, fs) }()
	go func() {
		defer wg.Done()
		state.dynamic.ContainerMemUsedGB = updateContainerMemUsage(state.static.CgroupVersion, fs)
	}()
	go func() {
		defer wg.Done()
		state.dynamic.LiveGPUUsage = updateLiveGPUUsage(state.static.GPUCount, runner)
	}()
	go func() {
		defer wg.Done()
		state.dynamic.StorageUsage = updateStorageUsage(state.static.StorageMounts)
	}()
	go func() { defer wg.Done(); state.dynamic.Processes = updateProcessList(&state.static, runner) }()
	go func() { defer wg.Done(); state.dynamic.HostCPUUsage = updateHostCPUUsage() }()
	go func() { defer wg.Done(); state.dynamic.HostMemUsedGB = updateHostMemUsage() }()
	wg.Wait()
}

// getStaticInfo fetches static information about the container and host.
func getStaticInfo(fs FileReader, runner CommandRunner, stater Stater) (StaticInfo, error) {
	var info StaticInfo
	var err error
	if _, err = stater.Stat("/sys/fs/cgroup/cpu.max"); os.IsNotExist(err) {
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
		cpuMaxStr, _ := readStringFromFile("/sys/fs/cgroup/cpu.max", fs)
		parts := strings.Fields(cpuMaxStr)
		if len(parts) == minCPUInfoCount {
			if parts[0] == "max" {
				return float64(hostCores)
			}
			quota, _ = strconv.ParseUint(parts[0], 10, 64)
			period, _ = strconv.ParseUint(parts[1], 10, 64)
		}
	} else { // CgroupV1
		q, err1 := readUintFromFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", fs)
		p, err2 := readUintFromFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us", fs)
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
		path = "/sys/fs/cgroup/memory.max"
	} else {
		path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	}
	limitStr, err := readStringFromFile(path, fs)
	if err != nil || limitStr == "max" {
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
		statStr, _ := readStringFromFile("/sys/fs/cgroup/cpu.stat", fs)
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "usage_usec" {
				currentUsage, _ = strconv.ParseUint(parts[1], 10, 64)
				break
			}
		}
	} else { // CgroupV1
		// value is in nanoseconds, convert to microseconds
		ns, err := readUintFromFile("/sys/fs/cgroup/cpuacct/cpuacct.usage", fs)
		if err == nil {
			currentUsage = ns / nanosecondsToMicroseconds
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
		statStr, _ := readStringFromFile("/sys/fs/cgroup/memory.stat", fs)
		for _, line := range strings.Split(statStr, "\n") {
			parts := strings.Fields(line)
			if len(parts) == 2 && parts[0] == "anon" {
				bytes, _ := strconv.ParseUint(parts[1], 10, 64)
				return float64(bytes) / float64(bytesPerGB)
			}
		}
	} else { // CgroupV1
		// memory.usage_in_bytes includes file cache, but is the standard.
		path := "/sys/fs/cgroup/memory/memory.usage_in_bytes"
		bytes, err := readUintFromFile(path, fs)
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

	var mounts []StorageMount
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

	var usage []StorageUsage
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
	totals := make([]float64, len(linesMem))
	for i, line := range linesMem {
		mb, _ := strconv.ParseFloat(strings.TrimSpace(line), 64)
		totals[i] = mb / bytesPerKB // Convert MB to GB
	}
	return len(linesMem), totals
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
		util, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		memUsedMB, _ := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
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
	processMap := make(map[int32]GPUProcessInfo)
	lines := strings.Split(string(out), "\n")
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
		gpuIndex, _ := strconv.Atoi(fields[0])
		gpuUtil, _ := strconv.ParseUint(fields[3], 10, 64)
		memUtil, _ := strconv.ParseUint(fields[4], 10, 64)
		processMap[int32(pid)] = GPUProcessInfo{GPUIndex: gpuIndex, GPUUtil: gpuUtil, GPUMemUtil: memUtil}
	}
	return processMap
}

// updateProcessList fetches the current process list and adds resource usage info when possible.
func updateProcessList(static *StaticInfo, runner CommandRunner) []ProcessInfo {
	gpuProcessMap := getGPUProcessMap(runner)
	procs, err := process.Processes()
	if err != nil {
		return nil
	}
	var processList []ProcessInfo
	for _, p := range procs {
		var cpuPercent float64
		cpuPercent, err = p.CPUPercent()
		if err != nil {
			continue
		}
		var memInfo *process.MemoryInfoStat
		memInfo, err = p.MemoryInfo()
		if err != nil {
			continue
		}
		var user string
		user, err = p.Username()
		if err != nil {
			user = "n/a"
		}
		var cmdline string
		cmdline, err = p.Cmdline()
		if err != nil || cmdline == "" {
			var name string
			name, err = p.Name()
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
			containerMemPercent = (float64(memInfo.RSS) / float64(static.ContainerMemLimitBytes)) * percentMultiplier
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
