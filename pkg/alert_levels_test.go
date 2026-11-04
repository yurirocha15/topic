package main

import (
	"strings"
	"testing"
)

func TestAlertLevelsMatchTheUsageColors(t *testing.T) {
	tests := []struct {
		percent float64
		level   string
		message string
	}{
		{65, "", ""},
		{70, alertWarning, "container memory is above 70%"},
		{89.9, alertWarning, "container memory is above 70%"},
		{90, alertCritical, "container memory is above 90%"},
	}
	for _, tt := range tests {
		static := StaticInfo{ContainerMemLimitBytes: 100 * bytesPerGB}
		alerts := evaluateAlerts(static, DynamicSnapshot{ContainerMemUsedGB: tt.percent})
		if tt.level == "" {
			if len(alerts) != 0 {
				t.Errorf("%.1f%%: expected no alert, got %+v", tt.percent, alerts)
			}
			continue
		}
		if len(alerts) != 1 || alerts[0].Level != tt.level || alerts[0].Message != tt.message {
			t.Errorf("%.1f%%: expected %s %q, got %+v", tt.percent, tt.level, tt.message, alerts)
		}
		if wantColor := usageColorName(tt.percent); (tt.level == alertCritical) != (wantColor == "red") {
			t.Errorf("%.1f%%: alert level %s disagrees with bar color %s", tt.percent, tt.level, wantColor)
		}
	}
}

func TestPIDAlertsUseTheSameLevels(t *testing.T) {
	for current, want := range map[uint64]string{69: "", 70: alertWarning, 90: alertCritical} {
		alerts := evaluateAlerts(StaticInfo{}, DynamicSnapshot{PIDUsage: PIDUsage{Current: current, Max: 100}})
		got := ""
		if len(alerts) == 1 {
			got = alerts[0].Level
		}
		if got != want {
			t.Errorf("%d of 100 PIDs: expected level %q, got %+v", current, want, alerts)
		}
	}
}

func TestPressureRowIsHighlightedWhenItAlerts(t *testing.T) {
	calm := &State{dynamic: DynamicInfo{Pressure: []PressureStat{{Resource: "cpu", SomeAvg10: 1}}}}
	high := &State{dynamic: DynamicInfo{Pressure: []PressureStat{{Resource: "cpu", SomeAvg10: 12}}}}
	if text := buildMetricsSection(calm, 80); strings.Contains(text, themeAttentionTag) {
		t.Errorf("expected calm pressure in the normal color, got %q", text)
	}
	if text := buildMetricsSection(high, 80); !strings.Contains(text, themeAttentionTag+"CPU 12.0") {
		t.Errorf("expected high pressure in the warning color, got %q", text)
	}
}
