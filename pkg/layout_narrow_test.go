package main

import (
	"fmt"
	"strings"
	"testing"
)

func richDashboardState() *State {
	state := dashboardLayoutTestState()
	state.dynamic.Alerts = []Alert{{Level: alertWarning, Message: "container memory is high"}}
	processes := make([]ProcessInfo, 0, 40)
	for pid := range 40 {
		processes = append(processes, ProcessInfo{
			PID: int32(1000 + pid), User: "root", Command: fmt.Sprintf("worker-%d", pid), GPUIndex: -1,
		})
	}
	state.dynamic.Processes = processes
	return state
}

func TestNarrowTerminalHidesSidePanelAndKeepsResourceLabels(t *testing.T) {
	for _, size := range [][2]int{{60, 20}, {80, 24}} {
		t.Run(fmt.Sprintf("%dx%d", size[0], size[1]), func(t *testing.T) {
			_, _, processTable, screen := renderDashboardTestScreen(richDashboardState(), size[0], size[1])
			text := strings.Join(simulationScreenRows(screen), "\n")
			for _, want := range []string{"CPU", "MEMORY"} {
				if !strings.Contains(text, want) {
					t.Errorf("expected the %s label on a narrow screen:\n%s", want, text)
				}
			}
			if strings.Contains(text, "top inside a container") {
				t.Errorf("expected the side panel to be hidden:\n%s", text)
			}
			if !strings.Contains(processTable.GetTitle(), "? help") {
				t.Errorf("expected a help hint in the process table title, got %q", processTable.GetTitle())
			}
		})
	}
}

func TestShortTerminalKeepsHalfTheScreenForProcesses(t *testing.T) {
	for _, size := range [][2]int{{60, 20}, {80, 24}, {120, 30}} {
		t.Run(fmt.Sprintf("%dx%d", size[0], size[1]), func(t *testing.T) {
			_, topPanel, processTable, screen := renderDashboardTestScreen(richDashboardState(), size[0], size[1])
			_, _, _, topHeight := topPanel.GetRect()
			_, _, _, tableHeight := processTable.GetRect()
			if topHeight > size[1]/2 || tableHeight < size[1]/2 {
				t.Fatalf("expected processes to keep half of %d rows, top=%d table=%d", size[1], topHeight, tableHeight)
			}
			text := strings.Join(simulationScreenRows(screen), "\n")
			for _, want := range []string{"CPU", "MEMORY", "ALERT"} {
				if !strings.Contains(text, want) {
					t.Errorf("expected %q to stay visible:\n%s", want, text)
				}
			}
			// The table border and header take three rows of the process half.
			if rows := strings.Count(text, "worker-"); rows < size[1]/2-3 {
				t.Errorf("expected at least %d process rows, got %d:\n%s", size[1]/2-3, rows, text)
			}
		})
	}
}

func TestDropLowPriorityLinesDropsHistoryThenActivityThenNetwork(t *testing.T) {
	lines := []resourceLine{
		{text: "alert", dropRank: keepResourceLine},
		{text: "pids", dropRank: dropNetworkLine},
		{text: "net", dropRank: dropNetworkLine},
		{text: "io", dropRank: dropActivityLine},
		{text: "psi", dropRank: dropActivityLine},
		{text: "hist cpu", dropRank: dropHistoryLine},
		{text: "hist mem", dropRank: dropHistoryLine},
	}
	texts := func(lines []resourceLine) string {
		var out []string
		for _, line := range lines {
			out = append(out, line.text)
		}
		return strings.Join(out, ",")
	}
	for limit, want := range map[int]string{
		7: "alert,pids,net,io,psi,hist cpu,hist mem",
		6: "alert,pids,net,io,psi,hist cpu",
		5: "alert,pids,net,io,psi",
		4: "alert,pids,net,io",
		2: "alert,pids",
		0: "alert",
	} {
		if got := texts(dropLowPriorityLines(append([]resourceLine(nil), lines...), limit)); got != want {
			t.Errorf("limit %d: got %s, want %s", limit, got, want)
		}
	}
}

func TestWideTerminalKeepsSidePanelAndAllRows(t *testing.T) {
	_, _, processTable, screen := renderDashboardTestScreen(richDashboardState(), 160, 50)
	text := strings.Join(simulationScreenRows(screen), "\n")
	for _, want := range []string{"top inside a container", "HIST CPU", "PSI", "NET eth0", "IO sda", "PIDS"} {
		if !strings.Contains(text, want) {
			t.Errorf("expected %q on a large screen:\n%s", want, text)
		}
	}
	if strings.Contains(processTable.GetTitle(), "? help") {
		t.Errorf("expected no help hint when the side panel shows the keys, got %q", processTable.GetTitle())
	}
}
