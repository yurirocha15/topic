package main

import (
	"testing"

	"github.com/rivo/tview"
)

func selectedPID(t *testing.T, table *tview.Table) int32 {
	t.Helper()
	row, _ := table.GetSelection()
	pid, ok := table.GetCell(row, processPIDColumn).GetReference().(int32)
	if !ok {
		t.Fatalf("selected row %d has no PID", row)
	}
	return pid
}

func TestProcessTableSelectionFollowsPIDWhenRowsReorder(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	trackProcessSelection(table, state)
	updateProcessTable(table, state)
	table.Select(3, 0) // PID 30, the least busy process

	// The collector publishes processes already sorted by CPU
	replaceProcesses(state, []ProcessInfo{
		{PID: 30, CPUPercent: 90, Command: "c", GPUIndex: -1},
		{PID: 20, CPUPercent: 30, Command: "b", GPUIndex: -1},
		{PID: 10, CPUPercent: 5, Command: "a", GPUIndex: -1},
	})
	updateProcessTable(table, state)

	if got := selectedPID(t, table); got != 30 {
		t.Fatalf("expected the highlight to stay on PID 30, got PID %d", got)
	}
	if row, _ := table.GetSelection(); row != 1 {
		t.Fatalf("expected PID 30 on row 1 after it became the busiest, got row %d", row)
	}
}

func TestProcessTableHighlightStaysOnRowUntilUserPicksProcess(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	trackProcessSelection(table, state)
	updateProcessTable(table, state)
	// tview moves the highlight onto the first row by itself while drawing; that is not a user choice.
	table.SetSelectionChangedFunc(nil)
	table.Select(1, 0)
	trackProcessSelection(table, state)

	replaceProcesses(state, []ProcessInfo{
		{PID: 30, CPUPercent: 90, Command: "c", GPUIndex: -1},
		{PID: 20, CPUPercent: 30, Command: "b", GPUIndex: -1},
		{PID: 10, CPUPercent: 5, Command: "a", GPUIndex: -1},
	})
	updateProcessTable(table, state)

	if row, _ := table.GetSelection(); row != 1 {
		t.Fatalf("expected the highlight to stay on row 1, got row %d", row)
	}
	if got := selectedPID(t, table); got != 30 {
		t.Fatalf("expected row 1 to show the busiest process, PID 30, got PID %d", got)
	}
}

func TestProcessTableStopsFollowingProcessThatExits(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	trackProcessSelection(table, state)
	updateProcessTable(table, state)
	table.Select(2, 0) // the user picks PID 20

	replaceProcesses(state, []ProcessInfo{
		{PID: 10, CPUPercent: 50, Command: "a", GPUIndex: -1},
		{PID: 30, CPUPercent: 10, Command: "c", GPUIndex: -1},
	})
	updateProcessTable(table, state)
	if state.followPID.Load() != 0 {
		t.Fatalf("expected to stop following the exited PID, still following %d", state.followPID.Load())
	}

	replaceProcesses(state, []ProcessInfo{
		{PID: 30, CPUPercent: 80, Command: "c", GPUIndex: -1},
		{PID: 10, CPUPercent: 50, Command: "a", GPUIndex: -1},
	})
	updateProcessTable(table, state)
	if row, _ := table.GetSelection(); row != 2 {
		t.Fatalf("expected the highlight to stay on row 2, got row %d", row)
	}
}

func TestProcessTableSelectionStaysOnVisibleRowWhenTableShrinks(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	trackProcessSelection(table, state)
	updateProcessTable(table, state)
	table.Select(3, 0)

	updateUIState(state, func(ui *UIState) { ui.ProcessFilter = "a" })
	updateProcessTable(table, state)

	process, ok := selectedProcess(table, state)
	if !ok || process.PID != 10 {
		t.Fatalf("expected the only visible process to be selected, got %+v ok=%v", process, ok)
	}
}

func TestProcessTableSelectionKeepsRowWhenSelectedProcessExits(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	trackProcessSelection(table, state)
	updateProcessTable(table, state)
	table.Select(2, 0) // PID 20

	replaceProcesses(state, []ProcessInfo{
		{PID: 10, CPUPercent: 50, Command: "a", GPUIndex: -1},
		{PID: 30, CPUPercent: 10, Command: "c", GPUIndex: -1},
	})
	updateProcessTable(table, state)

	if row, _ := table.GetSelection(); row != 2 {
		t.Fatalf("expected the selection to stay on row 2, got row %d", row)
	}
	if got := selectedPID(t, table); got != 30 {
		t.Fatalf("expected row 2 to show PID 30, got PID %d", got)
	}
}
