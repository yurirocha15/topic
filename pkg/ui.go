package main

import (
	"unicode/utf8"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

func handleInput(event *tcell.EventKey, state *State, app *tview.Application) *tcell.EventKey {
	state.dynamic.mu.Lock()
	searchMode := state.ui.SearchMode
	state.dynamic.mu.Unlock()

	if searchMode {
		return handleSearchInput(event, state)
	}

	switch {
	case event.Rune() == 'q' || event.Key() == tcell.KeyCtrlC:
		app.Stop()
		return nil
	case event.Rune() == '/':
		updateUIState(state, func(ui *UIState) {
			ui.SearchMode = true
			ui.ProcessFilter = ""
		})
		return nil
	case event.Rune() == 's':
		updateUIState(state, func(ui *UIState) {
			ui.ProcessSort = nextProcessSortColumn(ui.ProcessSort)
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
	case event.Rune() == 'a':
		updateUIState(state, func(ui *UIState) {
			ui.HideASCIIArt = !ui.HideASCIIArt
		})
		return nil
	case event.Key() == tcell.KeyEsc:
		updateUIState(state, func(ui *UIState) {
			ui.ProcessFilter = ""
			ui.SearchMode = false
		})
		return nil
	default:
		return event
	}
}

func handleSearchInput(event *tcell.EventKey, state *State) *tcell.EventKey {
	//nolint:exhaustive // Search mode handles text input through the default branch.
	switch event.Key() {
	case tcell.KeyEsc:
		updateUIState(state, func(ui *UIState) {
			ui.ProcessFilter = ""
			ui.SearchMode = false
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
	return nil
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
