package main

import (
	"fmt"
	"os"
	"syscall"
	"unicode/utf8"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

func handleInput(
	event *tcell.EventKey,
	state *State,
	app *tview.Application,
	pages *tview.Pages,
	processTable *tview.Table,
	signaler ProcessSignaler,
) *tcell.EventKey {
	if handled, nextEvent := handleModalInput(event, pages, app); handled {
		return nextEvent
	}

	if event.Key() == tcell.KeyCtrlC {
		app.Stop()
		return nil
	}

	state.dynamic.mu.Lock()
	searchMode := state.ui.SearchMode
	sortMode := state.ui.SortMode
	state.dynamic.mu.Unlock()

	if handled, nextEvent := handleActiveModeInput(event, state, app, searchMode, sortMode); handled {
		return nextEvent
	}

	switch {
	case event.Rune() == 'q':
		app.Stop()
		return nil
	case event.Rune() == '/':
		updateUIState(state, func(ui *UIState) {
			ui.SearchMode = true
			ui.SortMode = false
		})
		return nil
	case event.Rune() == 's':
		updateUIState(state, func(ui *UIState) {
			ui.SortMode = true
			ui.SearchMode = false
		})
		return nil
	case event.Rune() == 'r':
		updateUIState(state, func(ui *UIState) {
			ui.ReverseSort = !ui.ReverseSort
		})
		return nil
	case event.Rune() == 'p':
		updateUIState(state, func(ui *UIState) {
			ui.Paused = !ui.Paused
		})
		return nil
	case event.Rune() == 't':
		updateUIState(state, func(ui *UIState) {
			ui.TreeMode = !ui.TreeMode
		})
		return nil
	case event.Rune() == '?':
		showHelp(pages)
		return nil
	case event.Key() == tcell.KeyEnter:
		showProcessDetails(pages, state, processTable)
		return nil
	case event.Rune() == 'k':
		showSignalDialog(pages, state, processTable, signaler)
		return nil
	case event.Rune() == 'a':
		updateUIState(state, func(ui *UIState) {
			ui.HideASCIIArt = !ui.HideASCIIArt
		})
		return nil
	case event.Key() == tcell.KeyEsc:
		updateUIState(state, func(ui *UIState) {
			ui.SearchMode = false
			ui.SortMode = false
		})
		return nil
	default:
		return event
	}
}

func handleActiveModeInput(
	event *tcell.EventKey,
	state *State,
	app *tview.Application,
	searchMode bool,
	sortMode bool,
) (bool, *tcell.EventKey) {
	if searchMode {
		handleSearchInput(event, state)
		return true, nil
	}
	if sortMode {
		if event.Rune() == 'q' {
			app.Stop()
			return true, nil
		}
		handleSortInput(event, state)
		return true, nil
	}
	return false, event
}

func handleModalInput(
	event *tcell.EventKey,
	pages *tview.Pages,
	app *tview.Application,
) (bool, *tcell.EventKey) {
	if pages == nil {
		return false, event
	}
	name, _ := pages.GetFrontPage()
	if name == "" || name == "main" {
		return false, event
	}
	if event.Key() == tcell.KeyCtrlC {
		app.Stop()
		return true, nil
	}
	if event.Key() == tcell.KeyEsc || event.Rune() == 'q' {
		pages.RemovePage(name)
		return true, nil
	}
	return true, event
}

func handleSearchInput(event *tcell.EventKey, state *State) {
	//nolint:exhaustive // Search mode handles text input through the default branch.
	switch event.Key() {
	case tcell.KeyEsc:
		updateUIState(state, func(ui *UIState) {
			ui.SearchMode = false
		})
	case tcell.KeyCtrlU:
		updateUIState(state, func(ui *UIState) {
			ui.ProcessFilter = ""
		})
	case tcell.KeyEnter:
		updateUIState(state, func(ui *UIState) {
			ui.SearchMode = false
		})
	case tcell.KeyBackspace, tcell.KeyBackspace2:
		updateUIState(state, func(ui *UIState) {
			if ui.ProcessFilter == "" {
				return
			}
			_, size := utf8.DecodeLastRuneInString(ui.ProcessFilter)
			ui.ProcessFilter = ui.ProcessFilter[:len(ui.ProcessFilter)-size]
		})
	default:
		if event.Rune() != 0 {
			updateUIState(state, func(ui *UIState) {
				ui.ProcessFilter += string(event.Rune())
			})
		}
	}
}

func handleSortInput(event *tcell.EventKey, state *State) {
	//nolint:exhaustive // Sort mode only handles navigation/confirmation keys.
	switch event.Key() {
	case tcell.KeyEsc, tcell.KeyEnter:
		updateUIState(state, func(ui *UIState) {
			ui.SortMode = false
		})
	case tcell.KeyLeft:
		updateUIState(state, func(ui *UIState) {
			ui.ProcessSort = previousProcessSortColumn(ui.ProcessSort)
		})
	case tcell.KeyRight:
		updateUIState(state, func(ui *UIState) {
			ui.ProcessSort = nextProcessSortColumn(ui.ProcessSort)
		})
	case tcell.KeyUp:
		updateUIState(state, func(ui *UIState) {
			ui.ReverseSort = true
		})
	case tcell.KeyDown:
		updateUIState(state, func(ui *UIState) {
			ui.ReverseSort = false
		})
	default:
		return
	}
}

func updateUIState(state *State, update func(*UIState)) {
	state.dynamic.mu.Lock()
	update(&state.ui)
	state.dynamic.mu.Unlock()
}

func isPaused(state *State) bool {
	state.dynamic.mu.Lock()
	defer state.dynamic.mu.Unlock()
	return state.ui.Paused
}

func showHelp(pages *tview.Pages) {
	text := `topic controls

q / Ctrl+C    quit
↑↓←→ / mouse  navigate process table
/             filter processes
s             sort mode
←/→           choose sort column in sort mode
↑/↓           choose sort direction in sort mode
r             reverse sort outside sort mode
Esc           leave sort/filter mode
Ctrl+U        clear filter in filter mode
p             pause refresh
t             toggle process tree
a             toggle ASCII art
Enter         process details
k             signal selected process
?             this help`
	showModal(pages, "help", text, []string{"OK"}, nil)
}

func showProcessDetails(pages *tview.Pages, state *State, table *tview.Table) {
	process, ok := selectedProcess(table, state)
	if !ok {
		showMessage(pages, "No process selected")
		return
	}
	updateUIState(state, func(ui *UIState) {
		ui.SelectedPID = process.PID
	})
	showModal(pages, "process-details", processDetailsText(process), []string{"OK"}, nil)
}

func showSignalDialog(pages *tview.Pages, state *State, table *tview.Table, signaler ProcessSignaler) {
	process, ok := selectedProcess(table, state)
	if !ok {
		showMessage(pages, "No process selected")
		return
	}
	updateUIState(state, func(ui *UIState) {
		ui.SelectedPID = process.PID
	})
	text := fmt.Sprintf("Send a signal to PID %d?\n\n%s", process.PID, process.Command)
	showModal(pages, "signal", text, []string{signalTermLabel, signalKillLabel, "Cancel"}, func(_ int, label string) {
		switch label {
		case signalTermLabel:
			sendProcessSignal(pages, signaler, process.PID, syscall.SIGTERM)
		case signalKillLabel:
			confirmSIGKILL(pages, signaler, process.PID)
		}
	})
}

func confirmSIGKILL(pages *tview.Pages, signaler ProcessSignaler, pid int32) {
	text := fmt.Sprintf("Really send SIGKILL to PID %d?\n\nThis cannot be handled by the process.", pid)
	showModal(pages, "signal-confirm", text, []string{"Kill", "Cancel"}, func(_ int, label string) {
		if label == "Kill" {
			sendProcessSignal(pages, signaler, pid, syscall.SIGKILL)
		}
	})
}

func sendProcessSignal(pages *tview.Pages, signaler ProcessSignaler, pid int32, signal os.Signal) {
	if signaler == nil {
		showMessage(pages, "Process signaling is unavailable")
		return
	}
	if err := signaler.Signal(pid, signal); err != nil {
		showMessage(pages, fmt.Sprintf("Failed to send %s to PID %d: %v", signalLabel(signal), pid, err))
		return
	}
	showMessage(pages, fmt.Sprintf("Sent %s to PID %d", signalLabel(signal), pid))
}

func showMessage(pages *tview.Pages, text string) {
	showModal(pages, "message", text, []string{"OK"}, nil)
}

func showModal(
	pages *tview.Pages,
	name string,
	text string,
	buttons []string,
	done func(buttonIndex int, buttonLabel string),
) {
	modal := tview.NewModal().
		SetText(text).
		AddButtons(buttons).
		SetDoneFunc(func(buttonIndex int, buttonLabel string) {
			pages.RemovePage(name)
			if done != nil {
				done(buttonIndex, buttonLabel)
			}
		})
	pages.AddPage(name, modal, true, true)
}
