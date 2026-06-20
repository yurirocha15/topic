package main

import "strings"

func getStaticStorageInfoWithProvider(provider StorageProvider) []StorageMount {
	partitions, err := provider.Partitions(false)
	if err != nil {
		return nil
	}

	mounts := make([]StorageMount, 0, len(partitions))
	for _, partition := range partitions {
		// Skip virtual filesystems and temporary mounts
		if shouldSkipFilesystem(partition.Fstype, partition.Mountpoint) {
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

	return mounts
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

	for _, skip := range skipPaths {
		if strings.HasPrefix(mountpoint, skip) {
			return true
		}
	}

	// Skip loop devices (often used by snap packages and other virtual filesystems)
	if strings.Contains(mountpoint, "/loop") || strings.HasPrefix(mountpoint, "/snap/") {
		return true
	}

	return false
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
