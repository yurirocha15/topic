package main

import (
	"strings"
	"testing"
)

func TestContainerIDFromCgroupContract(t *testing.T) {
	id := strings.Repeat("ab12cd34", 8)
	pod := "kubepods-burstable-pod1234_5678.slice"
	cases := []struct {
		name   string
		cgroup string
		want   string
	}{
		{"v2 systemd docker scope", "0::/system.slice/docker-" + id + ".scope", id},
		{
			"v1 systemd docker multi controller",
			"12:memory:/system.slice/docker-" + id + ".scope\n11:cpu,cpuacct:/system.slice/docker-" + id + ".scope",
			id,
		},
		{
			"systemd containerd kubernetes scope",
			"0::/kubepods.slice/kubepods-burstable.slice/" + pod + "/cri-containerd-" + id + ".scope",
			id,
		},
		{"systemd cri-o scope", "0::/kubepods.slice/kubepods-burstable.slice/" + pod + "/crio-" + id + ".scope", id},
		{"legacy cgroupfs docker path", "0::/docker/" + id, id},
		{
			"legacy cgroupfs kubepods path",
			"12:memory:/kubepods/burstable/pod0a1b2c3d-4e5f-6789-abcd-ef0123456789/" + id,
			id,
		},
		{"root cgroup has no id", "0::/", ""},
		{"user session scope has no id", "0::/user.slice/user-1000.slice/session-2.scope", ""},
		{"docker service has no id", "0::/system.slice/docker.service", ""},
		{"pod uid slice is not an id", "0::/kubepods.slice/kubepods-pod0a1b2c3d_4e5f_6789_abcd_ef0123456789.slice", ""},
		{"non hex letters rejected", "0::/system.slice/docker-" + strings.Repeat("g", 64) + ".scope", ""},
		{"path traversal rejected", "0::/system.slice/docker-../../x.scope", ""},
		{"uppercase hex rejected", "0::/system.slice/docker-" + strings.ToUpper(id) + ".scope", ""},
		{"63 hex rejected", "0::/system.slice/docker-" + id[:63] + ".scope", ""},
		{"65 hex rejected", "0::/system.slice/docker-" + id + "a.scope", ""},
		{"12 hex short id rejected", "0::/system.slice/docker-" + id[:12] + ".scope", ""},
		{"legacy docker path with 65 hex rejected", "0::/docker/" + id + "a", ""},
		{"unknown prefix rejected", "0::/x/foo-" + id + ".scope", ""},
		{"bare id with scope suffix rejected", "0::/x/" + id + ".scope", ""},
		{"podman libpod scope", "0::/machine.slice/libpod-" + id + ".scope", id},
		{"nested child of systemd scope", "0::/system.slice/docker-" + id + ".scope/init", id},
		{"nested child of legacy docker path", "0::/docker/" + id + "/init", id},
		{
			"id on later line after session scope",
			"13:name=systemd:/user.slice/user-1000.slice/session-2.scope\n12:memory:/system.slice/docker-" + id + ".scope",
			id,
		},
		{"root line with trailing newline", "0::/\n", ""},
		{"empty file", "", ""},
		{
			"first valid id across lines wins",
			"1:name=systemd:/\n2:memory:/docker/" + id + "\n3:cpu:/docker/" + strings.Repeat("cd56ef78", 8),
			id,
		},
		{
			"legacy kubepods dashed pod uid only",
			"12:memory:/kubepods/burstable/pod0a1b2c3d-4e5f-6789-abcd-ef0123456789",
			"",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fs := MockFileReader{files: map[string]string{procSelfCgroupPath: tc.cgroup}}
			if got := containerIDFromCgroup(fs); got != tc.want {
				t.Fatalf("containerIDFromCgroup(%q) = %q, want %q", tc.cgroup, got, tc.want)
			}
		})
	}
}
