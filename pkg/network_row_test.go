package main

import (
	"os"
	"path/filepath"
	"testing"
)

func usageNames(usage []NetworkUsage) []string {
	names := make([]string, 0, len(usage))
	for _, network := range usage {
		names = append(names, network.Name)
	}
	return names
}

func TestShownNetworkInterfaceStaysUntilAnotherIsClearlyBusier(t *testing.T) {
	usage := func(eth, veth float64) []NetworkUsage {
		// Sorted busiest first, as the collector sorts them
		rows := []NetworkUsage{{Name: "eth0", RXBytesPerSec: eth}, {Name: "veth1", RXBytesPerSec: veth}}
		if veth > eth {
			rows[0], rows[1] = rows[1], rows[0]
		}
		return rows
	}
	tests := []struct {
		name     string
		previous string
		usage    []NetworkUsage
		want     string
	}{
		{"keeps the shown interface when another is a little busier", "eth0", usage(100, 150), "eth0"},
		{"switches when another carries more than twice as much", "eth0", usage(100, 250), "veth1"},
		{"switches when the shown interface is gone", "wlan0", usage(100, 50), "eth0"},
		{"shows the busiest at first", "", usage(10, 50), "veth1"},
		{"shows the default route when everything is idle", "", usage(0, 0), "veth1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := orderShownNetworkFirst(tt.usage, tt.previous, "veth1")
			if got[0].Name != tt.want || len(got) != 2 {
				t.Fatalf("expected %s first, got %v", tt.want, usageNames(got))
			}
		})
	}
}

func TestDefaultRouteInterfaceReadsTheRoutingTable(t *testing.T) {
	root := t.TempDir()
	route := "Iface\tDestination\tGateway\tFlags\n" +
		"docker0\t000011AC\t00000000\t0001\n" +
		"enp11s0\t00000000\t0101A8C0\t0003\n"
	if err := os.MkdirAll(filepath.Join(root, "net"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "net", "route"), []byte(route), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := defaultRouteInterface(root); got != "enp11s0" {
		t.Fatalf("expected enp11s0, got %q", got)
	}
	if got := defaultRouteInterface(t.TempDir()); got != "" {
		t.Fatalf("expected no interface without a routing table, got %q", got)
	}
}

func TestNetworkHistoryFollowsTheShownInterface(t *testing.T) {
	var history MetricsHistory
	dynamic := DynamicSnapshot{NetworkUsage: []NetworkUsage{
		{Name: "eth0", RXBytesPerSec: 2 * bytesPerSecondToMiBSecond},
		{Name: "docker0", RXBytesPerSec: 5 * bytesPerSecondToMiBSecond},
	}}
	updateMetricsHistory(&history, StaticInfo{}, dynamic)
	if got := (&history.Network).Ordered(); len(got) != 1 || got[0] != 2 {
		t.Fatalf("expected the shown interface's 2 MiB/s in history, got %v", got)
	}
}
