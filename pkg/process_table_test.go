package main

import (
	"math"
	"math/rand"
	"reflect"
	"slices"
	"strconv"
	"testing"

	"github.com/rivo/tview"
)

func treeModeTestState() *State {
	return &State{
		ui: UIState{ProcessSort: SortByCPU, TreeMode: true},
		dynamic: DynamicInfo{Processes: []ProcessInfo{
			{PID: 30, ParentPID: 10, CPUPercent: 90, Command: "grandchild-hot", GPUIndex: -1},
			{PID: 20, ParentPID: 10, CPUPercent: 80, Command: "child", GPUIndex: -1},
			{PID: 10, ParentPID: 0, CPUPercent: 5, Command: "root", GPUIndex: -1},
			{PID: 40, ParentPID: 0, CPUPercent: 70, Command: "other-root", GPUIndex: -1},
		}},
	}
}

func displayedPIDs(t *testing.T, table *tview.Table) []string {
	t.Helper()
	var pids []string
	for row := 1; row < table.GetRowCount(); row++ {
		pids = append(pids, table.GetCell(row, processPIDColumn).Text)
	}
	return pids
}

func TestTreeModeWithDefaultSortRendersParentsBeforeChildren(t *testing.T) {
	table := tview.NewTable()
	updateProcessTable(table, treeModeTestState())

	got := displayedPIDs(t, table)
	want := []string{"40", "10", "30", "20"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tree order PIDs = %v, want %v", got, want)
	}
}

func TestTreeModeWithDefaultSortPrefixesChildCommandsWithTreeMarker(t *testing.T) {
	table := tview.NewTable()
	updateProcessTable(table, treeModeTestState())

	// Row order expected: 40 (root), 10 (root), 30 (child), 20 (child).
	for row, want := range map[int]string{
		1: "other-root",
		2: "root",
		3: treeCommand("grandchild-hot", 1),
		4: treeCommand("child", 1),
	} {
		if got := table.GetCell(row, processCommandColumn).Text; got != want {
			t.Errorf("row %d command = %q, want %q", row, got, want)
		}
	}
}

func TestSelectedProcessMatchesDisplayedPIDForEveryRowInTreeMode(t *testing.T) {
	table := tview.NewTable()
	state := treeModeTestState()
	updateProcessTable(table, state)

	for row := 1; row < table.GetRowCount(); row++ {
		table.Select(row, 0)
		shown := table.GetCell(row, processPIDColumn).Text
		p, ok := selectedProcess(table, state)
		if !ok {
			t.Fatalf("row %d: selectedProcess ok=false, displayed PID %s", row, shown)
		}
		if got := strconv.Itoa(int(p.PID)); got != shown {
			t.Errorf("row %d: displayed PID %s but selectedProcess returned %s", row, shown, got)
		}
	}
}

// assertSelectionAnchored checks every displayed row against a (possibly newer) snapshot:
// survivors must resolve to the same PID with fresh data, vanished PIDs must give ok=false.
func assertSelectionAnchored(t *testing.T, table *tview.Table, state *State, displayed []string, fresh []ProcessInfo) {
	t.Helper()
	byPID := map[string]ProcessInfo{}
	for _, p := range fresh {
		byPID[strconv.Itoa(int(p.PID))] = p
	}
	for i, shown := range displayed {
		table.Select(i+1, 0)
		p, ok := selectedProcess(table, state)
		want, survives := byPID[shown]
		if !survives {
			if ok {
				t.Errorf("row %d displays vanished PID %s but selectedProcess returned PID %d", i+1, shown, p.PID)
			}
			continue
		}
		if !ok {
			t.Errorf("row %d displays surviving PID %s but selectedProcess returned ok=false", i+1, shown)
			continue
		}
		if p.PID != want.PID {
			t.Errorf("row %d displays PID %s but selectedProcess returned PID %d", i+1, shown, p.PID)
		} else if p.CPUPercent != want.CPUPercent {
			t.Errorf(
				"row %d PID %s returned stale CPU %.1f, want fresh %.1f",
				i+1, shown, p.CPUPercent, want.CPUPercent,
			)
		}
	}
}

func replaceProcesses(state *State, ps []ProcessInfo) {
	state.dynamic.mu.Lock()
	state.dynamic.Processes = ps
	state.dynamic.mu.Unlock()
}

func flatTestState() *State {
	return &State{
		ui: UIState{ProcessSort: SortByCPU},
		dynamic: DynamicInfo{Processes: []ProcessInfo{
			{PID: 10, CPUPercent: 50, Command: "a", GPUIndex: -1},
			{PID: 20, CPUPercent: 30, Command: "b", GPUIndex: -1},
			{PID: 30, CPUPercent: 10, Command: "c", GPUIndex: -1},
		}},
	}
}

func TestSelectedProcessAnchorsToDisplayedPIDAfterSnapshotChanges(t *testing.T) {
	table := tview.NewTable()
	state := flatTestState()
	updateProcessTable(table, state)
	displayed := displayedPIDs(t, table)

	fresh := []ProcessInfo{
		{PID: 30, CPUPercent: 95, Command: "c", GPUIndex: -1},
		{PID: 20, CPUPercent: 60, Command: "b", GPUIndex: -1},
		{PID: 99, CPUPercent: 40, Command: "new", GPUIndex: -1},
	}
	replaceProcesses(state, fresh)
	assertSelectionAnchored(t, table, state, displayed, fresh)
}

func TestSelectedProcessAnchorsToDisplayedPIDAfterSnapshotChangesInTreeMode(t *testing.T) {
	table := tview.NewTable()
	state := treeModeTestState()
	updateProcessTable(table, state)
	displayed := displayedPIDs(t, table)

	fresh := []ProcessInfo{
		{PID: 20, ParentPID: 0, CPUPercent: 99, Command: "child", GPUIndex: -1},
		{PID: 10, ParentPID: 0, CPUPercent: 1, Command: "root", GPUIndex: -1},
	}
	replaceProcesses(state, fresh)
	assertSelectionAnchored(t, table, state, displayed, fresh)
}

func TestSelectedProcessAnchorsToDisplayedPIDAfterFilterChangesWithoutRerender(t *testing.T) {
	table := tview.NewTable()
	state := flatTestState()
	updateProcessTable(table, state)
	displayed := displayedPIDs(t, table)

	state.dynamic.mu.Lock()
	state.ui.ProcessFilter = "b"
	state.dynamic.mu.Unlock()
	assertSelectionAnchored(t, table, state, displayed, state.dynamic.Processes)
}

func TestSelectedProcessAnchorsToDisplayedPIDAfterReverseSortChangesWithoutRerender(t *testing.T) {
	table := tview.NewTable()
	state := flatTestState()
	updateProcessTable(table, state)
	displayed := displayedPIDs(t, table)

	state.dynamic.mu.Lock()
	state.ui.ReverseSort = true
	state.dynamic.mu.Unlock()
	assertSelectionAnchored(t, table, state, displayed, state.dynamic.Processes)
}

func TestTreeModeWithFilterBuildsTreeFromFilteredRowsOnly(t *testing.T) {
	table := tview.NewTable()
	state := treeModeTestState()
	state.ui.ProcessFilter = "child"
	updateProcessTable(table, state)

	// Filter "child" matches commands "child" (20) and "grandchild-hot" (30); parent 10 is filtered out,
	// so both become roots, CPU-descending, with no tree marker.
	if got, want := displayedPIDs(t, table), []string{"30", "20"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("PIDs = %v, want %v", got, want)
	}
	for row, want := range map[int]string{1: "grandchild-hot", 2: "child"} {
		if got := table.GetCell(row, processCommandColumn).Text; got != want {
			t.Errorf("row %d command = %q, want %q", row, got, want)
		}
	}
}

func TestSelectedProcessMatchesDisplayedPIDInTreeModeWithFilter(t *testing.T) {
	table := tview.NewTable()
	state := treeModeTestState()
	state.ui.ProcessFilter = "child"
	updateProcessTable(table, state)

	displayed := displayedPIDs(t, table)
	if len(displayed) != 2 {
		t.Fatalf("expected 2 filtered rows, got %v", displayed)
	}
	assertSelectionAnchored(t, table, state, displayed, state.dynamic.Processes)
	for i, shown := range displayed {
		table.Select(i+1, 0)
		p, ok := selectedProcess(table, state)
		if !ok || strconv.Itoa(int(p.PID)) != shown {
			t.Errorf("row %d displays PID %s, selectedProcess = (%d, %v)", i+1, shown, p.PID, ok)
		}
	}
}

func TestTreeModeIgnoresReverseSortAndSortColumn(t *testing.T) {
	table := tview.NewTable()
	state := treeModeTestState()
	state.ui.ProcessSort = SortByUser
	state.ui.ReverseSort = true
	updateProcessTable(table, state)

	if got, want := displayedPIDs(t, table), []string{"40", "10", "30", "20"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tree order PIDs = %v, want %v", got, want)
	}
	assertSelectionAnchored(t, table, state, displayedPIDs(t, table), state.dynamic.Processes)
}

func TestSelectedProcessReturnsNotOKForHeaderRow(t *testing.T) {
	table := tview.NewTable()
	state := flatTestState()
	updateProcessTable(table, state)
	table.Select(0, 0)
	if p, ok := selectedProcess(table, state); ok {
		t.Fatalf("header row returned PID %d", p.PID)
	}
}

func TestSelectedProcessReturnsNotOKForRowBeyondTable(t *testing.T) {
	table := tview.NewTable()
	state := flatTestState()
	updateProcessTable(table, state)
	table.Select(table.GetRowCount()+5, 0)
	if p, ok := selectedProcess(table, state); ok {
		t.Fatalf("out-of-range row returned PID %d", p.PID)
	}
}

func TestSelectedProcessReturnsNotOKWhenNoProcessesRendered(t *testing.T) {
	table := tview.NewTable()
	state := &State{ui: UIState{ProcessSort: SortByCPU}}
	updateProcessTable(table, state)
	table.Select(1, 0)
	if p, ok := selectedProcess(table, state); ok {
		t.Fatalf("empty table returned PID %d", p.PID)
	}
}

type tieSortCase struct {
	name   string
	column ProcessSortColumn
	build  func(pid int32, group int) ProcessInfo
}

func tieSortCases() []tieSortCase {
	return []tieSortCase{
		{"cpu", SortByCPU, func(pid int32, g int) ProcessInfo {
			return ProcessInfo{PID: pid, CPUPercent: float64(50 - 10*g), User: "u", Command: "c", GPUIndex: -1}
		}},
		{"user", SortByUser, func(pid int32, g int) ProcessInfo {
			return ProcessInfo{PID: pid, User: string(rune('a' + g)), Command: "c", GPUIndex: -1}
		}},
		{"memory", SortByMemory, func(pid int32, g int) ProcessInfo {
			return ProcessInfo{PID: pid, MemPercent: float64(50 - 10*g), GPUIndex: -1}
		}},
		{"gpu", SortByGPU, func(pid int32, g int) ProcessInfo {
			return ProcessInfo{PID: pid, GPUUtil: uint64(50 - 10*g), GPUIndex: 0}
		}},
		{"gpumem", SortByGPUMemory, func(pid int32, g int) ProcessInfo {
			return ProcessInfo{PID: pid, GPUMemPercent: float64(50 - 10*g), GPUIndex: 0}
		}},
		{"command", SortByCommand, func(pid int32, g int) ProcessInfo {
			return ProcessInfo{PID: pid, Command: string(rune('a' + g)), GPUIndex: -1}
		}},
	}
}

func tieSortInput(c tieSortCase) []ProcessInfo {
	var ps []ProcessInfo
	// 3 groups of 4 processes with identical sort keys; PIDs interleaved across groups.
	for g := range 3 {
		for i := range 4 {
			ps = append(ps, c.build(int32(100+g+3*i), g))
		}
	}
	return ps
}

func pidsOf(ps []ProcessInfo) []int32 {
	out := make([]int32, len(ps))
	for i, p := range ps {
		out[i] = p.PID
	}
	return out
}

func TestSortProcessesBreaksTiesByAscendingPIDWhenNotReversed(t *testing.T) {
	for _, c := range tieSortCases() {
		t.Run(c.name, func(t *testing.T) {
			ps := tieSortInput(c)
			rand.New(rand.NewSource(1)).Shuffle(len(ps), func(i, j int) { ps[i], ps[j] = ps[j], ps[i] })
			sortProcesses(ps, c.column, false)

			// Group order is fixed by construction (group 0 first); PIDs in a group ascend.
			want := []int32{100, 103, 106, 109, 101, 104, 107, 110, 102, 105, 108, 111}
			if got := pidsOf(ps); !reflect.DeepEqual(got, want) {
				t.Fatalf("normal order = %v, want %v", got, want)
			}
		})
	}
}

func TestSortProcessesReverseIsExactReverseOfNormalOrder(t *testing.T) {
	for _, c := range tieSortCases() {
		t.Run(c.name, func(t *testing.T) {
			normal := tieSortInput(c)
			sortProcesses(normal, c.column, false)
			want := pidsOf(normal)
			for i, j := 0, len(want)-1; i < j; i, j = i+1, j-1 {
				want[i], want[j] = want[j], want[i]
			}

			ps := tieSortInput(c)
			rand.New(rand.NewSource(2)).Shuffle(len(ps), func(i, j int) { ps[i], ps[j] = ps[j], ps[i] })
			sortProcesses(ps, c.column, true)
			if got := pidsOf(ps); !reflect.DeepEqual(got, want) {
				t.Fatalf("reverse order = %v, want exact reverse %v", got, want)
			}
		})
	}
}

func TestSortProcessesResultIsIndependentOfInputOrder(t *testing.T) {
	for _, c := range tieSortCases() {
		for _, reverse := range []bool{false, true} {
			t.Run(c.name+"/reverse="+strconv.FormatBool(reverse), func(t *testing.T) {
				base := tieSortInput(c)
				sortProcesses(base, c.column, reverse)
				want := pidsOf(base)

				for seed := range int64(20) {
					ps := tieSortInput(c)
					rand.New(rand.NewSource(seed)).Shuffle(len(ps), func(i, j int) { ps[i], ps[j] = ps[j], ps[i] })
					sortProcesses(ps, c.column, reverse)
					if got := pidsOf(ps); !reflect.DeepEqual(got, want) {
						t.Fatalf("seed %d: order %v differs from %v", seed, got, want)
					}
				}
			})
		}
	}
}

func TestSortProcessesHandlesNaN(t *testing.T) {
	nan := math.NaN()
	processes := []ProcessInfo{
		{PID: 3, CPUPercent: nan},
		{PID: 1, CPUPercent: 5},
		{PID: 2, CPUPercent: nan},
		{PID: 4, CPUPercent: 9},
	}
	sortProcesses(processes, SortByCPU, false)

	pids := make([]int32, 0, len(processes))
	for _, process := range processes {
		pids = append(pids, process.PID)
	}
	if want := []int32{4, 1, 2, 3}; !slices.Equal(pids, want) {
		t.Fatalf("sorted PIDs = %v, want %v", pids, want)
	}
}
