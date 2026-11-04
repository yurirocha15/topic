package main

import (
	"strings"
	"testing"

	"github.com/gdamore/tcell/v2"
)

func TestProcessTableSortMarkerShowsActualDirection(t *testing.T) {
	tests := []struct {
		sort    ProcessSortColumn
		reverse bool
		header  string
	}{
		{SortByCPU, false, "%CPU↓"},
		{SortByCPU, true, "%CPU↑"},
		{SortByPID, false, "PID↑"},
		{SortByPID, true, "PID↓"},
		{SortByUser, false, "USER↑"},
		{SortByCommand, true, "COMMAND↓"},
	}
	for _, tt := range tests {
		ui := UIState{ProcessSort: tt.sort, ReverseSort: tt.reverse}
		headers := processTableHeaders(ui)
		if !strings.Contains(strings.Join(headers, " "), tt.header) {
			t.Errorf("sort %v reverse=%v: expected header %q, got %v", tt.sort, tt.reverse, tt.header, headers)
		}
		if status := plainText(sortStatusText(ui)); status != tt.header {
			t.Errorf("sort %v reverse=%v: expected info panel %q, got %q", tt.sort, tt.reverse, tt.header, status)
		}
	}
}

func TestSortModeDownArrowSortsDescendingForEveryColumn(t *testing.T) {
	processes := []ProcessInfo{
		{PID: 10, User: "alice", CPUPercent: 10, Command: "a", GPUIndex: -1},
		{PID: 20, User: "bob", CPUPercent: 20, Command: "b", GPUIndex: -1},
	}
	for _, column := range []ProcessSortColumn{SortByCPU, SortByPID, SortByUser, SortByCommand} {
		state := &State{ui: UIState{ProcessSort: column, SortMode: true}}
		handleSortInput(tcell.NewEventKey(tcell.KeyDown, 0, tcell.ModNone), state)
		rows := prepareProcessRows(append([]ProcessInfo(nil), processes...), state.ui)
		if rows[0].PID != 20 {
			t.Errorf("column %v: Down should sort descending, got first PID %d", column, rows[0].PID)
		}

		handleSortInput(tcell.NewEventKey(tcell.KeyUp, 0, tcell.ModNone), state)
		rows = prepareProcessRows(append([]ProcessInfo(nil), processes...), state.ui)
		if rows[0].PID != 10 {
			t.Errorf("column %v: Up should sort ascending, got first PID %d", column, rows[0].PID)
		}
	}
}
