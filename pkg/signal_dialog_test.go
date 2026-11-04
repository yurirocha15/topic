package main

import (
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

type signalDialogTUI struct {
	app      *tview.Application
	screen   tcell.SimulationScreen
	pages    *tview.Pages
	signaler *MockProcessSignaler
	done     chan error
}

func startSignalDialogTUI(t *testing.T) *signalDialogTUI {
	t.Helper()
	state := &State{
		ui: UIState{ProcessSort: SortByCPU},
		dynamic: DynamicInfo{Processes: []ProcessInfo{
			{PID: 123, User: "root", CPUPercent: 12.3, Command: "sleep 600", GPUIndex: -1},
		}},
	}
	tui := &signalDialogTUI{
		app:      tview.NewApplication(),
		screen:   tcell.NewSimulationScreen("UTF-8"),
		signaler: &MockProcessSignaler{},
		done:     make(chan error, 1),
	}
	table := configureProcessTable(tview.NewTable())
	tui.pages = tview.NewPages().AddPage("main", table, true, true)
	render := func() { updateProcessTable(table, state) }
	render()
	table.Select(1, 0)
	setAppInputCapture(tui.app, state, tui.pages, table, tui.signaler, render)
	tui.app.SetScreen(tui.screen).SetRoot(tui.pages, true)
	tui.screen.SetSize(100, 32)
	go func() { tui.done <- tui.app.Run() }()
	t.Cleanup(func() {
		go tui.app.Stop()
		select {
		case <-tui.done:
		case <-time.After(time.Second):
		}
	})
	waitForTUICondition(t, tui.app, table.HasFocus, "process table focus")
	return tui
}

func (tui *signalDialogTUI) press(t *testing.T, keys ...tcell.Key) {
	t.Helper()
	for _, key := range keys {
		tui.screen.InjectKey(key, 0, tcell.ModNone)
	}
}

func (tui *signalDialogTUI) openSignalDialog(t *testing.T) {
	t.Helper()
	tui.screen.InjectKey(tcell.KeyRune, 'k', tcell.ModNone)
	waitForTUICondition(t, tui.app, func() bool { return tui.pages.HasPage("signal") }, "signal dialog to open")
}

func TestSignalDialogEnterCancelsByDefault(t *testing.T) {
	tui := startSignalDialogTUI(t)
	tui.openSignalDialog(t)

	tui.press(t, tcell.KeyEnter)
	waitForTUICondition(t, tui.app, func() bool { return !tui.pages.HasPage("signal") }, "signal dialog to close")
	if len(tui.signaler.calls) != 0 {
		t.Fatalf("Enter in a fresh signal dialog sent %+v", tui.signaler.calls)
	}
}

func TestSignalDialogSendsSIGTERMWhenChosen(t *testing.T) {
	tui := startSignalDialogTUI(t)
	tui.openSignalDialog(t)

	tui.press(t, tcell.KeyLeft, tcell.KeyLeft, tcell.KeyEnter)
	waitForTUICondition(t, tui.app, func() bool { return len(tui.signaler.calls) == 1 }, "SIGTERM to be sent")
	if call := tui.signaler.calls[0]; call.pid != 123 || call.signal != syscall.SIGTERM {
		t.Fatalf("expected SIGTERM to PID 123, got %+v", call)
	}
}

func TestSIGKILLConfirmationCancelsByDefaultAndNamesCommand(t *testing.T) {
	tui := startSignalDialogTUI(t)
	tui.openSignalDialog(t)

	tui.press(t, tcell.KeyLeft, tcell.KeyEnter)
	waitForTUICondition(t, tui.app, func() bool { return tui.pages.HasPage("signal-confirm") }, "SIGKILL confirmation")
	var screenText string
	waitForTUICondition(t, tui.app, func() bool {
		screenText = simulationScreenText(tui.screen)
		return strings.Contains(screenText, "SIGKILL")
	}, "SIGKILL confirmation to draw")
	if !strings.Contains(screenText, "sleep 600") {
		t.Fatalf("expected the SIGKILL confirmation to name the command, got:\n%s", screenText)
	}

	tui.press(t, tcell.KeyEnter)
	waitForTUICondition(t, tui.app, func() bool {
		return !tui.pages.HasPage("signal-confirm")
	}, "confirmation to close")
	if len(tui.signaler.calls) != 0 {
		t.Fatalf("Enter in the SIGKILL confirmation sent %+v", tui.signaler.calls)
	}
}

func simulationScreenText(screen tcell.SimulationScreen) string {
	cells, width, height := screen.GetContents()
	var builder strings.Builder
	for y := range height {
		for x := range width {
			cell := cells[y*width+x]
			if len(cell.Runes) == 0 {
				builder.WriteByte(' ')
				continue
			}
			builder.WriteString(string(cell.Runes))
		}
		builder.WriteByte('\n')
	}
	return builder.String()
}
