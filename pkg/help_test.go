package main

import (
	"strings"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

type helpTUI struct {
	app    *tview.Application
	screen tcell.SimulationScreen
	pages  *tview.Pages
}

func startHelpTUI(t *testing.T, width, height int) *helpTUI {
	t.Helper()
	state := &State{ui: UIState{ProcessSort: SortByCPU}}
	tui := &helpTUI{app: tview.NewApplication(), screen: tcell.NewSimulationScreen("UTF-8")}
	table := configureProcessTable(tview.NewTable())
	tui.pages = tview.NewPages().AddPage("main", table, true, true)
	setAppInputCapture(tui.app, state, tui.pages, table, &MockProcessSignaler{}, func() {})
	tui.app.SetScreen(tui.screen).SetRoot(tui.pages, true)
	done := make(chan error, 1)
	go func() { done <- tui.app.Run() }()
	t.Cleanup(func() {
		go tui.app.Stop()
		select {
		case <-done:
		case <-time.After(time.Second):
		}
	})
	waitForTUICondition(t, tui.app, table.HasFocus, "process table focus")
	tui.screen.SetSize(width, height)
	if err := tui.screen.PostEvent(tcell.NewEventResize(width, height)); err != nil {
		t.Fatal(err)
	}
	tui.screen.InjectKey(tcell.KeyRune, '?', tcell.ModNone)
	waitForTUICondition(t, tui.app, func() bool { return tui.pages.HasPage("help") }, "help to open")
	return tui
}

func (tui *helpTUI) text(t *testing.T) string {
	t.Helper()
	var text string
	waitForTUICondition(t, tui.app, func() bool {
		text = simulationScreenText(tui.screen)
		return strings.Contains(text, "Help")
	}, "help to draw")
	return text
}

func TestHelpShowsKeysLeftAlignedOnOneLineEach(t *testing.T) {
	tui := startHelpTUI(t, 80, 24)
	text := tui.text(t)
	// The first lines of help fit on an 80x24 screen
	for _, want := range strings.Split(helpText(), "\n")[:10] {
		if !strings.Contains(text, want) {
			t.Errorf("expected help line %q on one line:\n%s", want, text)
		}
	}
}

func TestHelpScrollsOnShortTerminals(t *testing.T) {
	tui := startHelpTUI(t, 60, 12)
	if strings.Contains(tui.text(t), "close the dialog") {
		t.Fatal("expected the last help line to be below the fold on a 12-row terminal")
	}
	for range strings.Count(helpText(), "\n") {
		tui.screen.InjectKey(tcell.KeyPgDn, 0, tcell.ModNone)
	}
	waitForTUICondition(t, tui.app, func() bool {
		return strings.Contains(simulationScreenText(tui.screen), "close the dialog")
	}, "Page Down to reach the end of help")
}

func TestHelpClosesWithEscQOrQuestionMark(t *testing.T) {
	for _, key := range []struct {
		name string
		key  tcell.Key
		r    rune
	}{{"Esc", tcell.KeyEsc, 0}, {"q", tcell.KeyRune, 'q'}, {"?", tcell.KeyRune, '?'}} {
		t.Run(key.name, func(t *testing.T) {
			tui := startHelpTUI(t, 80, 24)
			tui.screen.InjectKey(key.key, key.r, tcell.ModNone)
			waitForTUICondition(t, tui.app, func() bool { return !tui.pages.HasPage("help") }, "help to close")
		})
	}
}
