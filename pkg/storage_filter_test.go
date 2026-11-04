package main

import (
	"testing"

	"github.com/shirou/gopsutil/v4/disk"
)

func TestShouldSkipFilesystemKeepsLookalikeMountpoints(t *testing.T) {
	paths := []string{
		"/developer", "/devdata", "/runner", "/system", "/sysroot", "/tmpdata",
		"/processing", "/var/runtime", "/var/locks", "/data/devices", "/mnt/data",
		"/etc", "/etc/hostsdata", "/etc/hostname.d", "/etc/resolv.conf.d",
		"/developer/", "/snapshots", "/snap",
		"/loop", "/mnt/loopback", "/srv/loop-data", "/home/user/loops",
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

func TestStorageDiscoveryHidesLoopDevicesByDevice(t *testing.T) {
	partitions := []disk.PartitionStat{
		{Device: "/dev/loop3", Mountpoint: "/mnt/image", Fstype: "ext4"},
		{Device: "/dev/loop0p1", Mountpoint: "/mnt/image-part", Fstype: "ext4"},
		{Device: "/dev/sdb1", Mountpoint: "/mnt/loopback", Fstype: "ext4"},
		{Device: "/dev/mapper/loopvg-data", Mountpoint: "/data", Fstype: "xfs"},
		{Device: "/dev/loop-control", Mountpoint: "/mnt/odd", Fstype: "ext4"},
	}
	usages := make(map[string]*disk.UsageStat, len(partitions))
	for _, partition := range partitions {
		usages[partition.Mountpoint] = &disk.UsageStat{Total: bytesPerGB}
	}

	mounts, err := getStaticStorageInfoWithProvider(
		MockStorageProvider{partitions: partitions, usages: usages},
		MockStater{},
	)
	if err != nil {
		t.Fatalf("getStaticStorageInfoWithProvider() error = %v", err)
	}

	var got []string
	for _, mount := range mounts {
		got = append(got, mount.Path)
	}
	want := []string{"/mnt/loopback", "/data", "/mnt/odd"}
	if len(got) != len(want) {
		t.Fatalf("expected mounts %v, got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected mounts %v, got %v", want, got)
		}
	}
}
