package main

import (
	"strings"
	"testing"

	"github.com/rivo/tview"
)

func TestProcessTableExplainsAFilterWithNoMatches(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	updateProcessTable(table, state)
	updateUIState(state, func(ui *UIState) { ui.ProcessFilter = "[red]zzz" })
	updateProcessTable(table, state)

	if table.GetRowCount() != 2 {
		t.Fatalf("expected the header and one message row, got %d rows", table.GetRowCount())
	}
	message := table.GetCell(1, processColumnPosition(processCommandColumn, false))
	if got := plainText(message.Text); !strings.Contains(got, `No processes match "[red]zzz"`) {
		t.Fatalf("expected a no-match message showing the filter literally, got %q", got)
	}
	if message.NotSelectable == false {
		t.Fatal("expected the message row not to be selectable")
	}
	if _, ok := selectedProcess(table, state); ok {
		t.Fatal("expected no process to be selected on the message row")
	}

	updateUIState(state, func(ui *UIState) { ui.ProcessFilter = "" })
	updateProcessTable(table, state)
	if got := table.GetCell(1, processPIDColumn).Text; got != "10" {
		t.Fatalf("expected processes back after clearing the filter, got %q", got)
	}
	if cell := table.GetCell(1, processColumnPosition(processCommandColumn, false)); cell.NotSelectable {
		t.Fatal("expected process rows to be selectable again")
	}
}

func TestProcessTableExplainsAnEmptyList(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	updateProcessTable(table, &State{ui: UIState{ProcessSort: SortByCPU}})
	message := table.GetCell(1, processColumnPosition(processCommandColumn, false))
	if got := plainText(message.Text); got != "No processes to show" {
		t.Fatalf("expected an empty-list message, got %q", got)
	}
}
