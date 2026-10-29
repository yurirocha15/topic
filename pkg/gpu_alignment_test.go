package main

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"
)

const (
	gpuTotalKey = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
	gpuUsageKey = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
)

func TestGetStaticGPUInfoKeepsSlotForNotAvailableRow(t *testing.T) {
	runner := MockCommandRunner{outputs: map[string]string{gpuTotalKey: "16384\n[N/A]\n24576"}}

	count, totals, err := getStaticGPUInfo(runner)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 3 {
		t.Fatalf("expected GPU count 3, got %d", count)
	}
	if len(totals) != 3 || totals[0] != 16 || totals[1] != 0 || totals[2] != 24 {
		t.Fatalf("expected totals [16 0 24], got %v", totals)
	}
}

func TestCalculateGPUBarsUsesGPUIndexForLabelsAndTotals(t *testing.T) {
	state := &State{}
	state.static.GPUTotalGB = []float64{8, 8, 16}
	state.dynamic.LiveGPUUsage = []GPUUsage{{Index: 2, Utilization: 50, MemUsedGB: 4}}

	bars := calculateGPUBars(state)
	if len(bars) != 2 {
		t.Fatalf("expected 2 bars, got %d", len(bars))
	}
	if bars[0].Label != "GPU2 UTIL" || bars[1].Label != "GPU2 MEM" {
		t.Fatalf("expected labels GPU2 UTIL/GPU2 MEM, got %q/%q", bars[0].Label, bars[1].Label)
	}
	if bars[1].Info != "4.00 / 16.00 GB" {
		t.Fatalf("expected info %q, got %q", "4.00 / 16.00 GB", bars[1].Info)
	}
	if bars[1].Percent != 25 {
		t.Fatalf("expected mem percent 25, got %v", bars[1].Percent)
	}
}

func TestCalculateGPUBarsHandlesIndexOutOfRangeOfTotals(t *testing.T) {
	state := &State{}
	state.static.GPUTotalGB = []float64{8}
	state.dynamic.LiveGPUUsage = []GPUUsage{{Index: 5, Utilization: 10, MemUsedGB: 2}}

	bars := calculateGPUBars(state)
	if len(bars) != 2 {
		t.Fatalf("expected 2 bars, got %d", len(bars))
	}
	if bars[1].Percent != 0 || bars[1].Info != "" {
		t.Fatalf("expected percent 0 and empty info, got %v %q", bars[1].Percent, bars[1].Info)
	}
}

func TestGPUBarsAlignTotalsWhenStaticAndLiveRowsAreMalformed(t *testing.T) {
	runner := MockCommandRunner{outputs: map[string]string{
		gpuTotalKey: "16384\n[N/A]\n24576",
		gpuUsageKey: "10, 1024\n[N/A], [N/A]\n30, 2048",
	}}

	count, totals, err := getStaticGPUInfo(runner)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	state := &State{}
	state.static.GPUCount = count
	state.static.GPUTotalGB = totals
	state.dynamic.LiveGPUUsage = updateLiveGPUUsage(count, runner)

	bars := calculateGPUBars(state)
	if len(bars) != 4 {
		t.Fatalf("expected 4 bars (GPU0 and GPU2), got %d", len(bars))
	}
	if bars[0].Label != "GPU0 UTIL" || bars[2].Label != "GPU2 UTIL" {
		t.Fatalf("unexpected labels %q, %q", bars[0].Label, bars[2].Label)
	}
	if bars[1].Info != "1.00 / 16.00 GB" {
		t.Fatalf("GPU0 info: expected %q, got %q", "1.00 / 16.00 GB", bars[1].Info)
	}
	if bars[3].Info != "2.00 / 24.00 GB" {
		t.Fatalf("GPU2 info: expected %q, got %q", "2.00 / 24.00 GB", bars[3].Info)
	}
}

func TestCalculateGPUBarsUnknownTotalShowsUsedOnlyAndZeroPercent(t *testing.T) {
	state := &State{}
	state.static.GPUTotalGB = []float64{0}
	state.dynamic.LiveGPUUsage = []GPUUsage{{Index: 0, Utilization: 5, MemUsedGB: 1}}

	bars := calculateGPUBars(state)
	if len(bars) != 2 {
		t.Fatalf("expected 2 bars, got %d", len(bars))
	}
	if p := bars[1].Percent; p != 0 || math.IsNaN(p) || math.IsInf(p, 0) {
		t.Fatalf("expected mem percent exactly 0, got %v", p)
	}
	if bars[1].Info != "1.00 GB" {
		t.Fatalf("expected info %q, got %q", "1.00 GB", bars[1].Info)
	}
}

func TestGetStaticGPUInfoAllMalformedRowsKeepSlots(t *testing.T) {
	runner := MockCommandRunner{outputs: map[string]string{gpuTotalKey: "[N/A]\n[Not Supported]"}}

	count, totals, err := getStaticGPUInfo(runner)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 2 || len(totals) != 2 || totals[0] != 0 || totals[1] != 0 {
		t.Fatalf("expected count 2 totals [0 0], got %d %v", count, totals)
	}
}

func TestGetStaticGPUInfoHandlesWhitespaceCRLFAndBlankLines(t *testing.T) {
	runner := MockCommandRunner{outputs: map[string]string{gpuTotalKey: " 8192 \r\n[N/A]\r\n\n"}}

	count, totals, err := getStaticGPUInfo(runner)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 2 || len(totals) != 2 || totals[0] != 8 || totals[1] != 0 {
		t.Fatalf("expected count 2 totals [8 0], got %d %v", count, totals)
	}
}

func TestGPUJSONSnapshotKeepsUnknownTotalSlots(t *testing.T) {
	runner := MockCommandRunner{outputs: map[string]string{gpuTotalKey: "16384\n[N/A]\n24576"}}
	count, totals, err := getStaticGPUInfo(runner)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	state := &State{}
	state.static.GPUCount = count
	state.static.GPUTotalGB = totals

	var buf bytes.Buffer
	if err = writeJSONSnapshot(&buf, state); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var decoded struct {
		Static struct {
			GPUCount   int       `json:"gpuCount"`
			GPUTotalGB []float64 `json:"gpuTotalGb"`
		} `json:"static"`
	}
	if err = json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	got := decoded.Static
	if got.GPUCount != 3 || len(got.GPUTotalGB) != 3 ||
		got.GPUTotalGB[0] != 16 || got.GPUTotalGB[1] != 0 || got.GPUTotalGB[2] != 24 {
		t.Fatalf("expected gpuCount 3 gpuTotalGb [16 0 24], got %d %v", got.GPUCount, got.GPUTotalGB)
	}
}

func TestCalculateGPUBarsTwoLiveRowsUseTheirOwnIndexTotals(t *testing.T) {
	state := &State{}
	state.static.GPUTotalGB = []float64{8, 8, 16}
	state.dynamic.LiveGPUUsage = []GPUUsage{
		{Index: 0, Utilization: 10, MemUsedGB: 2},
		{Index: 2, Utilization: 50, MemUsedGB: 4},
	}

	bars := calculateGPUBars(state)
	if len(bars) != 4 {
		t.Fatalf("expected 4 bars, got %d", len(bars))
	}
	if bars[0].Label != "GPU0 UTIL" || bars[2].Label != "GPU2 UTIL" {
		t.Fatalf("expected GPU0/GPU2 labels, got %q/%q", bars[0].Label, bars[2].Label)
	}
	if bars[1].Info != "2.00 / 8.00 GB" || bars[3].Info != "4.00 / 16.00 GB" {
		t.Fatalf("unexpected infos %q / %q", bars[1].Info, bars[3].Info)
	}
}
