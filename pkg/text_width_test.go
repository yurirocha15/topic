package main

import (
	"strings"
	"testing"

	"github.com/rivo/tview"
)

func TestTruncateAndPadMeasureTerminalCells(t *testing.T) {
	tests := []struct {
		value string
		width int
		want  string
	}{
		{"/mnt/data", 20, "/mnt/data"},
		{"/mnt/data/projects", 8, "/mnt/da…"},
		{"データベース", 12, "データベース"},
		{"データベース", 7, "データ…"},
		{"データベース", 6, "デー…"},
		{"été", 3, "été"},
	}
	for _, tt := range tests {
		got := truncatePlainText(tt.value, tt.width)
		if got != tt.want {
			t.Errorf("truncatePlainText(%q, %d) = %q, want %q", tt.value, tt.width, got, tt.want)
		}
		if width := tview.TaggedStringWidth(got); width > tt.width {
			t.Errorf("truncatePlainText(%q, %d) is %d cells wide", tt.value, tt.width, width)
		}
		if padded := padPlainText(got, tt.width); tview.TaggedStringWidth(padded) != tt.width &&
			tview.TaggedStringWidth(got) <= tt.width {
			t.Errorf("padPlainText(%q, %d) is %d cells wide", got, tt.width, tview.TaggedStringWidth(padded))
		}
	}
}

func TestResourceRowsStayAlignedWithWideCharacters(t *testing.T) {
	state := hostDashboardState()
	state.dynamic.StorageUsage = []StorageUsage{
		{Path: "/", UsedGB: 10, FreeGB: 90, UsedPercent: 10},
		{Path: "/mnt/データ/プロジェクト", UsedGB: 10, FreeGB: 90, UsedPercent: 10},
	}
	for _, width := range []int{60, 100, 180} {
		bars := []BarData{}
		label, info, usage := calculateCPULabelInfo(state)
		bars = append(bars, newBarData(label, usage, info))
		bars = append(bars, calculateStorageBars(state)...)
		layout := calculateResourceLayout(width, bars)
		rows := makeAlignedMultiColumnBars(bars, layout)
		for _, row := range rows {
			if got := tview.TaggedStringWidth(row); got != layout.TotalWidth {
				t.Errorf("width %d: row is %d cells, want %d: %q", width, got, layout.TotalWidth, row)
			}
		}
	}
}

func TestMetricLinesFitWithWideCharacters(t *testing.T) {
	state := hostDashboardState()
	state.dynamic.NetworkUsage = []NetworkUsage{{Name: "ネットワーク接続インターフェース"}}
	for _, width := range []int{40, 80} {
		for _, row := range activityMetricLines(state, width) {
			line := strings.TrimSuffix(row.text, "\n")
			if got := tview.TaggedStringWidth(line); got > width {
				t.Errorf("width %d: metric line is %d cells: %q", width, got, line)
			}
		}
	}
}
