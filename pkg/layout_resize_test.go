package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

func hostDashboardState() *State {
	return &State{
		ui: UIState{ProcessSort: SortByCPU, HideASCIIArt: true},
		static: StaticInfo{
			HostCores:         12,
			ContainerCPULimit: 12,
			HostMemTotalGB:    32,
		},
		dynamic: DynamicInfo{
			HostCPUUsage:  42,
			HostMemUsedGB: 8,
			Processes:     []ProcessInfo{{PID: 1, Command: "init", GPUIndex: -1}},
		},
	}
}

func screenLineWith(screen tcell.SimulationScreen, text string) string {
	for line := range strings.SplitSeq(simulationScreenText(screen), "\n") {
		if strings.Contains(line, text) {
			return line
		}
	}
	return ""
}

func TestRunTUILaysOutForTheRealWidthOnStartAndResize(t *testing.T) {
	// The simulation screen starts at 80x25 when the application initializes it.
	screen := tcell.NewSimulationScreen("UTF-8")
	app := tview.NewApplication().SetScreen(screen)
	// No refresh tick arrives during the test, so only re-layout on draw can update the rows.
	config := AppConfig{RefreshInterval: time.Hour}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- runTUI(ctx, app, config, hostDashboardState(), func() {}) }()
	defer func() {
		cancel()
		waitForRunTUI(t, done)
	}()

	// Laid out at width 0, the first frame has no labels; laid out again, the CPU row has its label.
	waitForTUICondition(t, app, func() bool {
		return strings.Contains(screenLineWith(screen, "42.0%"), "CPU")
	}, "the first frame to show the CPU label")

	screen.SetSize(200, 50)
	if err := screen.PostEvent(tcell.NewEventResize(200, 50)); err != nil {
		t.Fatal(err)
	}
	waitForTUICondition(t, app, func() bool {
		line := screenLineWith(screen, "CPU")
		return strings.Contains(line, "12 cores") && strings.Contains(line, "│") && strings.Contains(line, "MEMORY")
	}, "the wider terminal to show CPU and memory side by side with details")
}
