package main

import (
	"os"
	"strings"
	"testing"
)

func TestHelpLinesFitTheHelpBox(t *testing.T) {
	for line := range strings.SplitSeq(helpText(), "\n") {
		if width := plainCellWidth(line); width > helpMaxWidth-borderHeight {
			t.Errorf("help line is %d cells, wider than the %d-cell box: %q", width, helpMaxWidth-borderHeight, line)
		}
	}
}

func TestHelpListsEveryKeyOnce(t *testing.T) {
	text := helpText()
	for _, section := range keySections() {
		if !strings.Contains(text, section.title+"\n") {
			t.Errorf("expected section %q in help", section.title)
		}
		for _, binding := range section.bindings {
			if !strings.Contains(text, binding.keys) || !strings.Contains(text, binding.action) {
				t.Errorf("expected %q and %q in help", binding.keys, binding.action)
			}
		}
	}
}

func TestKeyHintsFitTheCompactSidePanel(t *testing.T) {
	want := map[string][]string{
		"Keys":  {"q quit", "? help", "p pause"},
		"Modes": {"/ filter", "s sort", "r reverse", "t tree"},
		"View":  {"Enter details", "k signal", "a logo"},
	}
	lines := keyHintLines()
	if len(lines) != len(want) {
		t.Fatalf("expected %d hint lines, got %q", len(want), lines)
	}
	for _, line := range lines {
		plain := plainText(line)
		if width := plainCellWidth(plain); width > compactInfoPanelWidth {
			t.Errorf("hint line is %d cells, wider than the %d-cell panel: %q", width, compactInfoPanelWidth, plain)
		}
		group := strings.Fields(plain)[0]
		for _, hint := range want[group] {
			if !strings.Contains(plain, hint) {
				t.Errorf("expected %q on the %s line, got %q", hint, group, plain)
			}
		}
	}
}

func TestReadmeControlsMatchKeyBindings(t *testing.T) {
	readme, err := os.ReadFile("../README.md")
	if err != nil {
		t.Fatal(err)
	}
	table := readmeControlsTable()
	if !strings.Contains(string(readme), table) {
		t.Fatalf("README.md Controls table is out of date; replace it with:\n\n%s", table)
	}
}
