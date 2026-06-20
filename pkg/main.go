// main.go
package main

import (
	"flag"
	"log"
	"math"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
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
	fsTypeTmpfs               = "tmpfs"
	fsTypeProc                = "proc"
	mountProcPath             = "/proc"
	mountTmpPath              = "/tmp"
	nvidiaSMICommand          = "nvidia-smi"
	nvidiaSMIMemoryTotalQuery = "--query-gpu=memory.total"
	nvidiaSMIUsageQuery       = "--query-gpu=utilization.gpu,memory.used"
	nvidiaSMICSVFormat        = "--format=csv,noheader,nounits"
	nvidiaSMIPMonCommand      = "pmon"
	nvidiaSMIPMonCountFlag    = "-c"
	nvidiaSMIPMonSampleCount  = "1"
	nvidiaSMIPMonSelectFlag   = "-s"
	nvidiaSMIPMonUsageMemory  = "um"
	signalTermLabel           = "SIGTERM"
	signalKillLabel           = "SIGKILL"
	signalIntLabel            = "SIGINT"
)

// --- Main Application ---

func main() {
	hideASCII := flag.Bool("no-ascii", false, "hide the ASCII art panel")
	flag.Parse()

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
		ui:          UIState{ProcessSort: SortByCPU, HideASCIIArt: *hideASCII},
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
	pages := tview.NewPages().
		AddPage("main", mainLayout, true, true)

	// --- Goroutine for periodic updates ---
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			<-ticker.C
			if !isPaused(state) {
				updateAll(state, fileReader, cmdRunner)
			}
			app.QueueUpdateDraw(func() {
				// Update all components and dynamically resize the top panel
				leftHeight := updateResourceView(resourceView, state)
				rightHeight := updateInfoView(infoView, state)
				topPanelHeight := int(math.Max(float64(leftHeight), float64(rightHeight))) + borderHeight
				mainLayout.ResizeItem(topPanel, topPanelHeight, 0)
				updateProcessTable(processTable, state)
			})
		}
	}()

	app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		return handleInput(event, state, app, pages, processTable, OSProcessSignaler{})
	})

	// Initial data load and draw
	updateAll(state, fileReader, cmdRunner)
	leftHeight := updateResourceView(resourceView, state)
	rightHeight := updateInfoView(infoView, state)
	topPanelHeight := int(math.Max(float64(leftHeight), float64(rightHeight))) + borderHeight
	mainLayout.ResizeItem(topPanel, topPanelHeight, 0)
	updateProcessTable(processTable, state)

	if err = app.SetRoot(pages, true).Run(); err != nil {
		log.Fatalf("Could not start application: %v", err)
	}
}

// --- UI Update Functions ---

// updateInfoView updates the info view with ASCII art and navigation guide.
func updateInfoView(view *tview.TextView, state *State) int {
	state.dynamic.mu.Lock()
	ui := state.ui
	state.dynamic.mu.Unlock()

	if ui.HideASCIIArt {
		fullText := `[::b]topic
[darkgrey]top inside a container

[darkgrey]Quit: q, Ctrl+C
[darkgrey]Sort mode: s  Reverse: r
[darkgrey]Filter mode: /  Clear: Ctrl+U
[darkgrey]Pause: p  Tree: t  ASCII: a
[darkgrey]Details: Enter  Signal: k
[darkgrey]Help: ?
[darkgrey]Navigate: ←↑→↓ / Mouse
`
		view.SetText(fullText)
		return strings.Count(fullText, "\n")
	}

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
[darkgrey] Sort mode: s  Reverse: r
[darkgrey]Filter mode: /  Clear: Ctrl+U
[darkgrey]     Pause: p  Tree: t  ASCII: a
[darkgrey]   Details: Enter  Signal: k
[darkgrey]      Help: ?
[darkgrey]  Navigate: ←↑→↓ / Mouse
[darkgrey]                        `

	fullText := asciiArt + subTitle + guide
	view.SetText(fullText)
	return strings.Count(fullText, "\n")
}
