// main.go
package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

// --- Constants ---

const (
	compactInfoPanelWidth     = 48                // Fixed compact width of the information panel
	asciiInfoPanelWidth       = 55                // Fixed width needed for the ASCII art panel
	placeholderHeight         = 10                // Initial height of the top panel
	borderHeight              = 2                 // Border height for boxes
	minCPUInfoCount           = 2                 // For cgroup CPU quota/period
	minGPUUsageCount          = 2                 // For nvidia-smi GPU usage
	minGPUInfoCount           = 5                 // For nvidia-smi pmon
	barsPerGPU                = 2                 // GPU utilization and memory bars
	dynamicCollectorCount     = 12                // Number of dynamic collectors in refreshDynamicState
	percentMultiplier         = 100.0             // Multiplier for percentage calculations
	bytesPerKB                = 1024              // Bytes per kilobyte
	bytesPerMB                = bytesPerKB * 1024 // Bytes per megabyte
	bytesPerGB                = bytesPerMB * 1024 // Bytes per gigabyte
	nanosecondsToMicroseconds = 1000              // Nanoseconds to microseconds
	secondToMicroseconds      = 1e6               // Seconds to microseconds
	minBarWidth               = 20                // Preferred minimum width for a progress bar
	maxBarWidth               = 30                // Keep wide-terminal bars compact and scannable
	compactBarWidth           = 4                 // Smallest useful bar when the terminal is constrained
	columnSpacing             = 5                 // Width of the divider between metric columns
	maxDisplayPathLength      = 15                // Maximum length for mount path display
	metricFieldSpacing        = 2                 // Space between label, value, bar, and detail
	metricLeadingFieldGaps    = 2                 // Gaps between label, value, and bar
	metricPercentWidth        = 6                 // Visible width of a percentage value
	maxMetricLabelWidth       = 18                // Prevent long device labels from shifting the grid
	minMetricInfoWidth        = 10                // Smallest useful detail field before it is hidden
	minMetricColumnWidth      = 57                // Minimum readable width for one resource column
	activityLabelWidth        = 14                // Shared label gutter for activity and history rows
	commandTimeout            = 2 * time.Second   // Maximum time allowed for external probes
	cgroupMaxToken            = "max"             // Cgroup token for unlimited resources
	cgroupCPUMaxPath          = "/sys/fs/cgroup/cpu.max"
	cgroupCPUStatPath         = "/sys/fs/cgroup/cpu.stat"
	cgroupV1CPUQuotaPath      = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	cgroupV1CPUPeriodPath     = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
	cgroupV1CPUUsagePath      = "/sys/fs/cgroup/cpuacct/cpuacct.usage"
	cgroupMemoryMaxPath       = "/sys/fs/cgroup/memory.max"
	cgroupMemoryEventsPath    = "/sys/fs/cgroup/memory.events"
	cgroupMemoryStatPath      = "/sys/fs/cgroup/memory.stat"
	cgroupPIDsCurrentPath     = "/sys/fs/cgroup/pids.current"
	cgroupPIDsMaxPath         = "/sys/fs/cgroup/pids.max"
	cgroupCPUPressurePath     = "/sys/fs/cgroup/cpu.pressure"
	cgroupMemoryPressurePath  = "/sys/fs/cgroup/memory.pressure"
	cgroupIOPressurePath      = "/sys/fs/cgroup/io.pressure"
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
	metricCPULabel            = "CPU"
	metricMemoryLabel         = "MEMORY"
)

// --- Main Application ---

func main() {
	config := parseConfig()

	fileReader := OSFileReader{}
	cmdRunner := OSCommandRunner{Timeout: commandTimeout}
	stater := OSStater{}

	staticInfo, err := getStaticInfo(fileReader, cmdRunner, stater)
	if err != nil {
		log.Printf("Could not get all static info: %v. Continuing...", err)
	}
	if config.DisableGPU {
		staticInfo.GPUCount = 0
		staticInfo.GPUTotalGB = nil
	}
	staticInfo.Metadata, staticInfo.Integrations = discoverIntegrations(fileReader, config)

	state := &State{
		static:      staticInfo,
		dynamic:     DynamicInfo{},
		ui:          UIState{ProcessSort: config.InitialSort, HideASCIIArt: config.HideASCIIArt},
		prevCPUTime: time.Now(),
	}

	if config.Once {
		refreshDynamicState(state, fileReader, cmdRunner)
		if config.JSONOutput {
			if err = writeJSONSnapshot(os.Stdout, state); err != nil {
				log.Fatalf("Could not write JSON snapshot: %v", err)
			}
			return
		}
		if _, err = os.Stdout.WriteString(onceText(state)); err != nil {
			log.Fatalf("Could not write snapshot: %v", err)
		}
		return
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
	processTable := configureProcessTable(tview.NewTable())

	// --- Create the Layout ---

	// Top panel combines resources (left) and info (right)
	topPanel := tview.NewFlex().
		SetDirection(tview.FlexColumn).
		AddItem(resourceView, 0, 1, false).                // Resources take up remaining width
		AddItem(infoView, compactInfoPanelWidth, 0, false) // Info panel has a fixed width

	topPanel.SetBorder(true).SetTitle(" System ")

	// Main layout combines top panel and process table
	mainLayout := tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(topPanel, placeholderHeight, 0, false). // Top panel has a placeholder height, will be resized
		AddItem(processTable, 0, 1, true)               // Process table takes all remaining vertical space
	pages := tview.NewPages().
		AddPage("main", mainLayout, true, true)
	render := func() {
		leftHeight := updateResourceView(resourceView, state)
		rightHeight := updateInfoView(infoView, state)
		topPanel.ResizeItem(infoView, currentInfoPanelWidth(state), 0)
		topPanelHeight := int(math.Max(float64(leftHeight), float64(rightHeight))) + borderHeight
		mainLayout.ResizeItem(topPanel, topPanelHeight, 0)
		updateProcessTable(processTable, state)
	}

	// --- Goroutine for periodic updates ---
	go func() {
		ticker := time.NewTicker(config.RefreshInterval)
		defer ticker.Stop()
		for {
			<-ticker.C
			if !isPaused(state) {
				refreshDynamicState(state, fileReader, cmdRunner)
			}
			app.QueueUpdateDraw(render)
		}
	}()

	setAppInputCapture(app, state, pages, processTable, OSProcessSignaler{}, render)

	// Initial data load and draw
	refreshDynamicState(state, fileReader, cmdRunner)
	render()

	if err = app.SetRoot(pages, true).Run(); err != nil {
		log.Fatalf("Could not start application: %v", err)
	}
}

func configureProcessTable(table *tview.Table) *tview.Table {
	table.SetSelectable(true, false).
		SetFixed(1, 0).
		SetSeparator(tview.Borders.Vertical).
		SetSelectedStyle(tcell.StyleDefault.Foreground(tcell.ColorWhite).Background(tcell.ColorDarkCyan).Bold(true))
	table.SetBorder(true).SetTitle(" Processes ")
	return table
}

// --- UI Update Functions ---

// updateInfoView updates the info view with ASCII art and navigation guide.
func updateInfoView(view *tview.TextView, state *State) int {
	state.dynamic.mu.Lock()
	ui := state.ui
	state.dynamic.mu.Unlock()

	if ui.HideASCIIArt {
		fullText := compactInfoText(ui, state.static.Integrations)
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
	integrations := integrationStatusText(state.static.Integrations)
	guide := `
` + integrations + `

[darkgrey]      Quit: q, Ctrl+C
[darkgrey] Sort mode: s  Reverse: r
[darkgrey]Filter mode: /  Clear: Ctrl+U
[darkgrey]     Pause: p  Tree: t  Logo: a
[darkgrey]   Details: Enter  Signal: k
[darkgrey]      Help: ?
[darkgrey]  Navigate: ←↑→↓ / Mouse
[darkgrey]                        `

	fullText := asciiArt + subTitle + guide
	view.SetText(fullText)
	return strings.Count(fullText, "\n")
}

func currentInfoPanelWidth(state *State) int {
	state.dynamic.mu.Lock()
	hideASCII := state.ui.HideASCIIArt
	state.dynamic.mu.Unlock()
	if hideASCII {
		return compactInfoPanelWidth
	}
	return asciiInfoPanelWidth
}

func compactInfoText(ui UIState, statuses []IntegrationStatus) string {
	integrations := strings.TrimSpace(integrationStatusText(statuses))
	if integrations == "" {
		integrations = themeMutedTag + "Integrations: none" + themeResetTag
	}
	return fmt.Sprintf(
		themeStrongTag+"topic"+themeStyleResetTag+" "+themeMutedTag+"top inside a container"+themeResetTag+"\n"+
			themeMutedTag+"Mode "+themeResetTag+"%s  "+themeMutedTag+"Sort "+themeResetTag+"%s  "+
			themeMutedTag+"Filter "+themeResetTag+"%s\n"+
			"%s\n"+
			themeMutedTag+"Keys  "+themeResetTag+"q quit  p pause  ? help\n"+
			themeMutedTag+"Modes "+themeResetTag+"/ filter  s sort  t tree\n"+
			themeMutedTag+"View  "+themeResetTag+"Enter details  a logo\n",
		infoModeText(ui),
		sortStatusText(ui),
		filterStatusText(ui),
		integrations,
	)
}

func infoModeText(ui UIState) string {
	switch {
	case ui.SearchMode:
		return themeAttentionTag + "filter" + themeResetTag
	case ui.SortMode:
		return themeAttentionTag + "sort" + themeResetTag
	case ui.Paused:
		return themeAttentionTag + "paused" + themeResetTag
	default:
		return "[green]live" + themeResetTag
	}
}

func filterStatusText(ui UIState) string {
	if ui.ProcessFilter == "" {
		return themeMutedTag + "-" + themeResetTag
	}
	filter := ui.ProcessFilter
	const maxFilterStatusWidth = 13
	if len([]rune(filter)) > maxFilterStatusWidth {
		filter = string([]rune(filter)[:maxFilterStatusWidth-1]) + "…"
	}
	return themeAttentionTag + filter + themeResetTag
}

func sortStatusText(ui UIState) string {
	direction := "↓"
	if ui.ReverseSort {
		direction = "↑"
	}
	return themeAccentTag + processSortColumnLabel(ui.ProcessSort) + direction + themeResetTag
}

func processSortColumnLabel(column ProcessSortColumn) string {
	switch column {
	case SortByMemory:
		return processMemoryLabel
	case SortByGPU:
		return processGPULabel
	case SortByGPUMemory:
		return processGPUMemoryLabel
	case SortByPID:
		return processPIDLabel
	case SortByUser:
		return processUserLabel
	case SortByCommand:
		return processCommandLabel
	case SortByCPU:
		fallthrough
	default:
		return processCPULabel
	}
}

func integrationStatusText(statuses []IntegrationStatus) string {
	if len(statuses) == 0 {
		return ""
	}
	var builder strings.Builder
	builder.WriteByte('\n')
	builder.WriteString(themeMutedTag)
	builder.WriteString("Integrations:")
	builder.WriteString(themeResetTag)
	for _, status := range statuses {
		marker := "-"
		markerTag := themeMutedTag
		if status.Available {
			marker = "+"
			markerTag = "[green]"
		}
		builder.WriteString(themeMutedTag)
		builder.WriteString(" ")
		builder.WriteString(status.Name)
		builder.WriteString("=")
		builder.WriteString(themeResetTag)
		builder.WriteString(markerTag)
		builder.WriteString(marker)
		builder.WriteString(themeResetTag)
	}
	return builder.String()
}
