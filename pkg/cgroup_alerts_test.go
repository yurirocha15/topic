package main

import (
	"strings"
	"testing"
	"time"
)

const (
	testOOMAlert      = "cgroup reports OOM kills"
	testThrottleAlert = "cgroup reports CPU throttling"
)

func alertMessages(alerts []Alert) string {
	messages := make([]string, 0, len(alerts))
	for _, alert := range alerts {
		messages = append(messages, alert.Message)
	}
	return strings.Join(messages, ",")
}

func TestCgroupEventAlertsIgnoreCountersFromBeforeStartup(t *testing.T) {
	state := &State{}
	now := time.Now()
	events := CgroupEvents{MemoryOOMKill: 3, CPUThrottledPeriods: 50}

	for tick := range 3 {
		alerts := cgroupEventAlerts(state, events, now.Add(time.Duration(tick)*time.Second))
		if len(alerts) != 0 {
			t.Fatalf("tick %d: unchanged counters raised %s", tick, alertMessages(alerts))
		}
	}
}

func TestCgroupEventAlertsFireOnIncrease(t *testing.T) {
	tests := []struct {
		name   string
		events CgroupEvents
		level  string
		want   string
	}{
		{"oom kill", CgroupEvents{MemoryOOMKill: 2, CPUThrottledPeriods: 5}, alertCritical, testOOMAlert},
		{"throttling", CgroupEvents{MemoryOOMKill: 1, CPUThrottledPeriods: 6}, alertWarning, testThrottleAlert},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := &State{}
			now := time.Now()
			cgroupEventAlerts(state, CgroupEvents{MemoryOOMKill: 1, CPUThrottledPeriods: 5}, now)

			alerts := cgroupEventAlerts(state, test.events, now.Add(time.Second))
			if len(alerts) != 1 || alerts[0].Message != test.want || alerts[0].Level != test.level {
				t.Fatalf("alerts = %+v, want one %s %q", alerts, test.level, test.want)
			}
		})
	}
}

func TestCgroupEventAlertsClearAfterHoldTime(t *testing.T) {
	state := &State{}
	now := time.Now()
	cgroupEventAlerts(state, CgroupEvents{}, now)
	increased := CgroupEvents{MemoryOOMKill: 1}
	occurred := now.Add(time.Second)
	cgroupEventAlerts(state, increased, occurred)

	held := cgroupEventAlerts(state, increased, occurred.Add(cgroupEventAlertHold-time.Second))
	if alertMessages(held) != testOOMAlert {
		t.Errorf("alert should stay during the hold time, got %+v", held)
	}
	if cleared := cgroupEventAlerts(state, increased, occurred.Add(cgroupEventAlertHold)); len(cleared) != 0 {
		t.Errorf("alert should clear after the hold time, got %+v", cleared)
	}
}

func TestCgroupEventAlertsRebaselineWhenCountersReset(t *testing.T) {
	state := &State{}
	now := time.Now()
	cgroupEventAlerts(state, CgroupEvents{MemoryOOMKill: 5, CPUThrottledPeriods: 5}, now)

	if alerts := cgroupEventAlerts(state, CgroupEvents{}, now.Add(time.Second)); len(alerts) != 0 {
		t.Fatalf("a counter reset raised %+v", alerts)
	}
	alerts := cgroupEventAlerts(state, CgroupEvents{MemoryOOMKill: 1}, now.Add(2*time.Second))
	if alertMessages(alerts) != testOOMAlert {
		t.Fatalf("increase after a reset should alert, got %+v", alerts)
	}
}

func TestEvaluateAlertsLeavesCumulativeCountersOut(t *testing.T) {
	dynamic := DynamicSnapshot{CgroupEvents: CgroupEvents{MemoryOOMKill: 9, CPUThrottledPeriods: 9}}
	if alerts := evaluateAlerts(StaticInfo{}, dynamic); len(alerts) != 0 {
		t.Fatalf("evaluateAlerts() = %+v, want none", alerts)
	}
}

func TestSparklineHandlesNegativeValues(t *testing.T) {
	ring := HistoryRing{}
	ring.Add(10)
	ring.Add(-5)
	if blocks := []rune(sparkline(ring)); len(blocks) < 2 || blocks[0] != '█' || blocks[1] != '▁' {
		t.Fatalf("sparkline() = %q, want the negative value on the lowest block", string(blocks))
	}
}

func TestWriteAlertMetricsUsesLevelColor(t *testing.T) {
	tests := []struct {
		level string
		want  string
		avoid string
	}{
		{alertWarning, themeAttentionTag, "[red]"},
		{alertCritical, "[red]", themeAttentionTag},
	}
	for _, test := range tests {
		t.Run(test.level, func(t *testing.T) {
			var builder strings.Builder
			writeAlertMetrics(&builder, []Alert{{Level: test.level, Message: "message"}}, 80)
			text := builder.String()
			if !strings.Contains(text, test.want) || strings.Contains(text, test.avoid) {
				t.Fatalf("alert text = %q, want %s and not %s", text, test.want, test.avoid)
			}
		})
	}
}
