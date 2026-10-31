// main.go
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

// --- Constants ---

const (
	exitOK      = 0 // Successful run
	exitFailure = 1 // Runtime failure
	exitUsage   = 2 // Invalid command-line arguments, matching the flag package

	defaultVersion     = "dev"     // Version of a binary built without release ldflags
	develModuleVersion = "(devel)" // Build info version of a local build

	onceSampleWindow = 250 * time.Millisecond // Time between the readings of a one-shot snapshot

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
	displayPathPrefix         = "..."             // Marks a shortened mount path
	metricFieldSpacing        = 2                 // Space between label, value, bar, and detail
	metricLeadingFieldGaps    = 2                 // Gaps between label, value, and bar
	metricPercentWidth        = 6                 // Visible width of a percentage value
	maxMetricLabelWidth       = 18                // Prevent long device labels from shifting the grid
	minMetricInfoWidth        = 10                // Smallest useful detail field before it is hidden
	minMetricColumnWidth      = 57                // Minimum readable width for one resource column
	activityLabelWidth        = 14                // Shared label gutter for activity and history rows
	commandTimeout            = 2 * time.Second   // Maximum time allowed for external probes
	cgroupMaxToken            = "max"             // Cgroup token for unlimited resources
	cgroupV1UnlimitedMemory   = 1 << 62           // Cgroup v1 memory limits at or above this are unlimited
	cgroupCPUMaxPath          = "/sys/fs/cgroup/cpu.max"
	cgroupCPUStatPath         = "/sys/fs/cgroup/cpu.stat"
	cgroupV1CPUQuotaPath      = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	cgroupV1CPUPeriodPath     = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
	cgroupV1CPUUsagePath      = "/sys/fs/cgroup/cpuacct/cpuacct.usage"
	cgroupMemoryMaxPath       = "/sys/fs/cgroup/memory.max"
	cgroupMemoryEventsPath    = "/sys/fs/cgroup/memory.events"
	cgroupMemoryCurrentPath   = "/sys/fs/cgroup/memory.current"
	cgroupMemoryStatPath      = "/sys/fs/cgroup/memory.stat"
	cgroupPIDsCurrentPath     = "/sys/fs/cgroup/pids.current"
	cgroupPIDsMaxPath         = "/sys/fs/cgroup/pids.max"
	cgroupCPUPressurePath     = "/sys/fs/cgroup/cpu.pressure"
	cgroupMemoryPressurePath  = "/sys/fs/cgroup/memory.pressure"
	cgroupIOPressurePath      = "/sys/fs/cgroup/io.pressure"
	cgroupV1MemoryLimitPath   = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	cgroupV1MemoryUsagePath   = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
	cgroupV1MemoryStatPath    = "/sys/fs/cgroup/memory/memory.stat"
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

// Version is set at build time with -ldflags "-X main.Version=vX.Y.Z".
var Version = defaultVersion //nolint:gochecknoglobals // -ldflags -X needs a package variable

// --- Main Application ---

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// run executes the program and returns its exit code.
func run(args []string, stdout, stderr io.Writer) int {
	config, err := parseConfigArgs(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return exitOK
	}
	if err != nil {
		return exitUsage
	}
	if config.ShowVersion {
		fmt.Fprintf(stdout, "topic %s\n", versionString())
		return exitOK
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
	defer stop()
	useStableProcessStartTimes()

	fileReader := OSFileReader{}
	cmdRunner := OSCommandRunner{Context: ctx, Timeout: commandTimeout}
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
	refresh := func() {
		refreshDynamicState(state, fileReader, cmdRunner)
	}

	if config.Once {
		sampleOnce(refresh, time.Sleep)
		if err = writeOnceSnapshot(stdout, config, state); err != nil {
			fmt.Fprintf(stderr, "Could not write snapshot: %v\n", err)
			return exitFailure
		}
		return exitOK
	}

	if err = runTUI(ctx, tview.NewApplication(), config, state, refresh); err != nil {
		fmt.Fprintf(stderr, "Could not start application: %v\n", err)
		return exitFailure
	}
	return exitOK
}

// versionString returns the injected version, or the module version for `go install` builds.
func versionString() string {
	if Version != defaultVersion {
		return Version
	}
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != develModuleVersion {
		return info.Main.Version
	}
	return Version
}

// sampleOnce refreshes twice because CPU, network, and disk rates need two readings.
func sampleOnce(refresh func(), wait func(time.Duration)) {
	refresh()
	wait(onceSampleWindow)
	refresh()
}

func writeOnceSnapshot(stdout io.Writer, config AppConfig, state *State) error {
	if config.JSONOutput {
		return writeJSONSnapshot(stdout, state)
	}
	_, err := io.WriteString(stdout, onceText(state))
	return err
}

// runTUI shows the dashboard until the user quits or ctx is cancelled.
func runTUI(ctx context.Context, app *tview.Application, config AppConfig, state *State, refresh func()) error {
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
		topPanelHeight := max(leftHeight, rightHeight) + borderHeight
		mainLayout.ResizeItem(topPanel, topPanelHeight, 0)
		updateProcessTable(processTable, state)
	}

	setAppInputCapture(app, state, pages, processTable, OSProcessSignaler{}, render)

	// Initial data load and draw, finished before the ticker can start another refresh
	refresh()
	render()

	// QueueUpdate waits for the event loop, so a signal during startup still stops the app
	stopWatching := context.AfterFunc(ctx, func() {
		app.QueueUpdate(app.Stop)
	})
	defer stopWatching()

	tickerCtx, stopTicker := context.WithCancel(ctx)
	defer stopTicker()

	// --- Goroutine for periodic updates ---
	go func() {
		ticker := time.NewTicker(config.RefreshInterval)
		defer ticker.Stop()
		for {
			select {
			case <-tickerCtx.Done():
				return
			case <-ticker.C:
			}
			if !isPaused(state) {
				refresh()
			}
			app.QueueUpdateDraw(render)
		}
	}()

	return app.SetRoot(pages, true).Run()
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
	return themeAttentionTag + escapeText(filter) + themeResetTag
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
