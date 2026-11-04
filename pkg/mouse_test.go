package main

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

// mouseTrackingScreen records whether the application turned mouse reporting on.
type mouseTrackingScreen struct {
	tcell.SimulationScreen

	mouseEnabled atomic.Bool
}

func (s *mouseTrackingScreen) EnableMouse(flags ...tcell.MouseFlags) {
	s.mouseEnabled.Store(true)
	s.SimulationScreen.EnableMouse(flags...)
}

func TestRunTUISelectsProcessWithMouseClick(t *testing.T) {
	screen := &mouseTrackingScreen{SimulationScreen: tcell.NewSimulationScreen("UTF-8")}
	screen.SetSize(100, 32)
	app := tview.NewApplication().SetScreen(screen)
	config := AppConfig{RefreshInterval: time.Hour}
	state := &State{ui: UIState{ProcessSort: SortByCPU, HideASCIIArt: true}}
	state.dynamic.Processes = []ProcessInfo{
		{PID: 10, CPUPercent: 50, Command: "proc-alpha", GPUIndex: -1},
		{PID: 20, CPUPercent: 30, Command: "proc-bravo", GPUIndex: -1},
		{PID: 30, CPUPercent: 10, Command: "proc-charlie", GPUIndex: -1},
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- runTUI(ctx, app, config, state, func() {}) }()
	defer func() {
		cancel()
		waitForRunTUI(t, done)
	}()

	charlieRow := -1
	waitForTUICondition(t, app, func() bool {
		for y, line := range strings.Split(simulationScreenText(screen), "\n") {
			if strings.Contains(line, "proc-charlie") {
				charlieRow = y
				return true
			}
		}
		return false
	}, "process rows to draw")

	if !screen.mouseEnabled.Load() {
		t.Fatal("expected the TUI to enable mouse input")
	}
	screen.InjectMouse(3, charlieRow, tcell.Button1, tcell.ModNone)
	screen.InjectMouse(3, charlieRow, tcell.ButtonNone, tcell.ModNone)
	screen.InjectKey(tcell.KeyEnter, 0, tcell.ModNone)
	waitForTUICondition(t, app, func() bool {
		return strings.Contains(simulationScreenText(screen), "PID: 30")
	}, "details for the clicked process")
}
