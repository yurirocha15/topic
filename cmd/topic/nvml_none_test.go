//go:build !nvml

package main

import (
	"errors"
	"testing"
)

func TestStaticBuildDoesNotLoadNVML(t *testing.T) {
	if _, err := openNVMLGPUSource(); !errors.Is(err, errNVMLNotBuilt) {
		t.Fatalf("expected the static build to report that it cannot load NVML, got %v", err)
	}
}
