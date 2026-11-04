package main

import (
	"strings"
	"testing"
)

func TestResourceDetailsUseTheNormalTextColor(t *testing.T) {
	row := formatAlignedBar(newBarData("MEMORY", 50, "16.10 / 30.40 GiB"), 10, 8, 20, 60)
	if !strings.Contains(row, themeLabelTag+"16.10"+themeMutedTag+" / "+themeLabelTag+"30.40 GiB") {
		t.Fatalf("expected details in the label color with a muted separator, got %q", row)
	}
	if strings.Contains(row, themeMutedTag+"16.10") {
		t.Fatalf("expected the numbers not to be muted, got %q", row)
	}
	if got, want := plainText(row), "16.10 / 30.40 GiB"; !strings.Contains(got, want) {
		t.Fatalf("expected the plain detail %q, got %q", want, got)
	}
}
