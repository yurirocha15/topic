package main

import "testing"

// Note: the "/loop" substring rule is intentionally unchanged in this slice,
// so there is no lookalike keep test for it.
func TestShouldSkipFilesystemKeepsLookalikeMountpoints(t *testing.T) {
	paths := []string{
		"/developer", "/devdata", "/runner", "/system", "/sysroot", "/tmpdata",
		"/processing", "/var/runtime", "/var/locks", "/data/devices", "/mnt/data",
		"/etc", "/etc/hostsdata", "/etc/hostname.d", "/etc/resolv.conf.d",
		"/developer/", "/snapshots", "/snap",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			if shouldSkipFilesystem("ext4", path) {
				t.Errorf("shouldSkipFilesystem(ext4, %q) = true, expected false", path)
			}
		})
	}
}

func TestShouldSkipFilesystemSkipsSystemPathsAndSubpaths(t *testing.T) {
	paths := []string{
		"/dev", "/sys", "/sys/fs/cgroup", "/proc", "/run", "/run/user/1000",
		"/tmp", "/var/run/secrets", "/var/lock", "/dev/", "/run/",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			if !shouldSkipFilesystem("ext4", path) {
				t.Errorf("shouldSkipFilesystem(ext4, %q) = false, expected true", path)
			}
		})
	}
}

func TestShouldSkipFilesystemSkipsContainerBindMountedFiles(t *testing.T) {
	paths := []string{"/etc/hosts", "/etc/hostname", "/etc/resolv.conf"}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			if !shouldSkipFilesystem("ext4", path) {
				t.Errorf("shouldSkipFilesystem(ext4, %q) = false, expected true", path)
			}
		})
	}
}

func TestShouldSkipFilesystemKeepsOverlayRoot(t *testing.T) {
	if shouldSkipFilesystem("overlay", "/") {
		t.Error("shouldSkipFilesystem(overlay, /) = true, expected false")
	}
}

func TestShouldSkipFilesystemSkipsVirtualFstypes(t *testing.T) {
	for _, fstype := range []string{"tmpfs", "proc", "sysfs", "cgroup2"} {
		t.Run(fstype, func(t *testing.T) {
			if !shouldSkipFilesystem(fstype, "/mnt/x") {
				t.Errorf("shouldSkipFilesystem(%q, /mnt/x) = false, expected true", fstype)
			}
		})
	}
}

func TestShouldSkipFilesystemSkipsSnapMounts(t *testing.T) {
	if !shouldSkipFilesystem("ext4", "/snap/core/123") {
		t.Error("expected /snap/core/123 to be skipped")
	}
}
