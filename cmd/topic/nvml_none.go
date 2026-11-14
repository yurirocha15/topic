//go:build !nvml

package main

import "errors"

var errNVMLNotBuilt = errors.New("this build of topic cannot load NVML")

// openNVML is unavailable in the static build, which uses nvidia-smi instead.
func openNVML() (nvmlLibrary, error) {
	return nil, errNVMLNotBuilt
}
