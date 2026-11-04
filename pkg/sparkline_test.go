package main

import (
	"slices"
	"strings"
	"testing"
)

func TestHistoryShowsOnlyRecordedSamples(t *testing.T) {
	var ring HistoryRing
	ring.Add(25)
	ring.Add(30)
	if got := (&ring).Ordered(); !slices.Equal(got, []float64{25, 30}) {
		t.Fatalf("expected only the two recorded samples, got %v", got)
	}
	for value := range historySize + 2 {
		ring.Add(float64(value))
	}
	got := (&ring).Ordered()
	if len(got) != historySize || got[0] != 2 || got[historySize-1] != historySize+1 {
		t.Fatalf("expected the last %d samples in order, got %v", historySize, got)
	}
}

func TestPercentSparklineUsesAFixedScale(t *testing.T) {
	var steady HistoryRing
	for range 5 {
		steady.Add(20)
	}
	if got := percentSparkline(steady); got != "▂▂▂▂▂" {
		t.Fatalf("expected a steady 20%% to stay low, got %q", got)
	}

	var spread HistoryRing
	for _, value := range []float64{-5, 0, 50, 100, 150} {
		spread.Add(value)
	}
	if got := percentSparkline(spread); got != "▁▁▄██" {
		t.Fatalf("expected 0-100 mapped to the eight blocks and clamped, got %q", got)
	}
}

func TestHistoryRowsUseTheFixedPercentScale(t *testing.T) {
	state := hostDashboardState()
	for range 5 {
		state.history.CPU.Add(20)
		state.history.Memory.Add(20)
	}
	text := plainText(buildMetricsSection(state, 80))
	if !containsLine(text, "HIST CPU", "▂▂▂▂▂") || containsLine(text, "HIST CPU", "█") {
		t.Fatalf("expected a low steady CPU history, got:\n%s", text)
	}
}

func containsLine(text, label, value string) bool {
	for line := range strings.SplitSeq(text, "\n") {
		if strings.HasPrefix(line, label) {
			return strings.Contains(line, value)
		}
	}
	return false
}
