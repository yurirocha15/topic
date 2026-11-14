package main

import (
	"fmt"
	"strings"
)

func getStaticStorageInfoWithProvider(provider StorageProvider, stater Stater) ([]StorageMount, error) {
	partitions, err := provider.Partitions(false)
	if err != nil {
		return nil, fmt.Errorf("list partitions: %w", err)
	}

	mounts := make([]StorageMount, 0, len(partitions))
	for _, partition := range partitions {
		// Skip virtual filesystems, temporary mounts, and loop devices such as snap images
		if shouldSkipFilesystem(partition.Fstype, partition.Mountpoint) || isLoopDevice(partition.Device) {
			continue
		}

		// Skip files bind-mounted into the container, which report the usage of the host's disk
		if info, statErr := stater.Stat(partition.Mountpoint); statErr == nil && !info.IsDir() {
			continue
		}

		usage, usageErr := provider.Usage(partition.Mountpoint)
		if usageErr != nil {
			continue
		}

		mount := StorageMount{
			Path:    partition.Mountpoint,
			TotalGB: float64(usage.Total) / float64(bytesPerGB),
			Fstype:  partition.Fstype,
		}
		mounts = append(mounts, mount)
	}
	return mounts, nil
}

// shouldSkipFilesystem determines if a filesystem should be skipped from monitoring.
func shouldSkipFilesystem(fstype, mountpoint string) bool {
	// Skip virtual/temporary filesystems
	skipFsTypes := []string{
		fsTypeTmpfs, "devtmpfs", "sysfs", fsTypeProc, "devpts", "cgroup", "cgroup2",
		"pstore", "bpf", "debugfs", "tracefs", "securityfs", "fusectl",
		"configfs", "selinuxfs", "mqueue", "hugetlbfs", "autofs", "rpc_pipefs",
		"squashfs", "overlayfs",
	}

	for _, skip := range skipFsTypes {
		if fstype == skip {
			return true
		}
	}

	// Skip system mount points
	skipPaths := []string{
		"/dev", "/sys", mountProcPath, "/run", mountTmpPath, "/var/run", "/var/lock",
	}

	// Match on path-segment boundaries so /developer is not mistaken for /dev
	normalized := mountpoint
	if len(normalized) > 1 {
		normalized = strings.TrimRight(normalized, "/")
	}

	for _, skip := range skipPaths {
		if normalized == skip || strings.HasPrefix(normalized, skip+"/") {
			return true
		}
	}

	// Skip single files bind-mounted by Docker and Kubernetes
	switch normalized {
	case "/etc/hosts", "/etc/hostname", "/etc/resolv.conf":
		return true
	}

	// Skip snap package mounts
	return strings.HasPrefix(mountpoint, "/snap/")
}

// isLoopDevice reports whether device is a loop block device such as /dev/loop3 or /dev/loop0p1.
func isLoopDevice(device string) bool {
	number, ok := strings.CutPrefix(device, "/dev/loop")
	return ok && number != "" && number[0] >= '0' && number[0] <= '9'
}

func updateStorageUsageWithProvider(storageMounts []StorageMount, provider StorageProvider) []StorageUsage {
	if len(storageMounts) == 0 {
		return nil
	}

	usage := make([]StorageUsage, 0, len(storageMounts))
	for _, mount := range storageMounts {
		stat, err := provider.Usage(mount.Path)
		if err != nil {
			continue
		}

		usage = append(usage, StorageUsage{
			Path:        mount.Path,
			UsedGB:      float64(stat.Used) / float64(bytesPerGB),
			FreeGB:      float64(stat.Free) / float64(bytesPerGB),
			UsedPercent: stat.UsedPercent,
		})
	}

	return usage
}
