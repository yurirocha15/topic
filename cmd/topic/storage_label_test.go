package main

import (
	"strings"
	"testing"

	"github.com/rivo/tview"
)

func TestLongMountPathsKeepTheirEndsWhenDrawn(t *testing.T) {
	state := hostDashboardState()
	state.dynamic.StorageUsage = []StorageUsage{
		{Path: "/home/user/models/llama", UsedGB: 1, FreeGB: 9, UsedPercent: 10},
		{Path: "/home/user/models/mistral", UsedGB: 1, FreeGB: 9, UsedPercent: 10},
	}
	bars := calculateStorageBars(state)
	layout := calculateResourceLayout(100, bars)
	rows := makeAlignedMultiColumnBars(bars, layout)
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(rows))
	}
	for index, want := range []string{"llama", "mistral"} {
		label := strings.SplitN(plainText(rows[index]), "  ", 2)[0]
		if !strings.HasPrefix(label, "DISK …") || !strings.HasSuffix(label, "/"+want) {
			t.Errorf("expected the label to keep the end of the path, got %q", label)
		}
		if width := tview.TaggedStringWidth(label); width > layout.MaxLabelWidth {
			t.Errorf("label %q is wider than %d cells", label, layout.MaxLabelWidth)
		}
	}
}

func TestShortMountPathsAreNotShortened(t *testing.T) {
	state := hostDashboardState()
	state.dynamic.StorageUsage = []StorageUsage{{Path: "/data", UsedGB: 1, FreeGB: 9, UsedPercent: 10}}
	bars := calculateStorageBars(state)
	rows := makeAlignedMultiColumnBars(bars, calculateResourceLayout(100, bars))
	if !strings.HasPrefix(plainText(rows[0]), "DISK /data ") {
		t.Fatalf("expected the full short path, got %q", plainText(rows[0]))
	}
}
