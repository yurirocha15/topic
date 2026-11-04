package main

import (
	"strings"
	"testing"

	"github.com/rivo/tview"
)

func headerTexts(table *tview.Table) []string {
	var headers []string
	for column := range table.GetColumnCount() {
		headers = append(headers, table.GetCell(0, column).Text)
	}
	return headers
}

func TestProcessTableHidesGPUColumnsWithoutGPU(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	updateProcessTable(table, state)

	headers := strings.Join(headerTexts(table), " ")
	if strings.Contains(headers, processGPULabel) {
		t.Fatalf("expected no GPU columns without a GPU, got %q", headers)
	}
	if got := table.GetCell(1, processColumnPosition(processCommandColumn, false)).Text; got != "a" {
		t.Fatalf("expected the command in the last column, got %q", got)
	}
}

func TestProcessTableShowsGPUColumnsWithGPU(t *testing.T) {
	table := configureProcessTable(tview.NewTable())
	state := flatTestState()
	state.static.GPUCount = 1
	state.dynamic.Processes[0].GPUIndex = 0
	state.dynamic.Processes[0].GPUUtil = 55
	updateProcessTable(table, state)

	headers := headerTexts(table)
	if len(headers) != 7 || headers[processGPUColumn] != processGPULabel {
		t.Fatalf("expected the GPU columns, got %v", headers)
	}
	if got := table.GetCell(1, processGPUColumn).Text; got != "55" {
		t.Fatalf("expected GPU use 55, got %q", got)
	}
}

func TestSortModeSkipsGPUColumnsWithoutGPU(t *testing.T) {
	seen := make(map[ProcessSortColumn]bool)
	column := SortByCPU
	for range 10 {
		column = nextProcessSortColumn(column, false)
		seen[column] = true
		if previousProcessSortColumn(column, false) == SortByGPU {
			t.Fatal("moving left reached the hidden GPU column")
		}
	}
	if seen[SortByGPU] || seen[SortByGPUMemory] {
		t.Fatalf("sort mode reached a hidden GPU column: %v", seen)
	}
	if len(seen) != 5 {
		t.Fatalf("expected the five visible columns, got %v", seen)
	}
}
