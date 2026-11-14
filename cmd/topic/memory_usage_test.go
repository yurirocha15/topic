package main

import (
	"math"
	"testing"
)

func TestUpdateContainerMemUsageReportsWorkingSet(t *testing.T) {
	tests := []struct {
		name    string
		version CgroupVersion
		files   map[string]string
		want    float64
	}{
		{
			name:    "v2 counts shmem and active cache",
			version: CgroupV2,
			files: map[string]string{
				cgroupMemoryCurrentPath: "8589934592",
				cgroupMemoryStatPath:    "anon 1073741824\nshmem 6442450944\ninactive_file 1073741824\n",
			},
			want: 7,
		},
		{
			name:    "v1 excludes inactive cache",
			version: CgroupV1,
			files: map[string]string{
				cgroupV1MemoryUsagePath: "8589934592",
				cgroupV1MemoryStatPath:  "cache 7516192768\ninactive_file 1\ntotal_inactive_file 6442450944\n",
			},
			want: 2,
		},
		{
			name:    "v2 without memory.stat falls back to usage",
			version: CgroupV2,
			files:   map[string]string{cgroupMemoryCurrentPath: "2147483648"},
			want:    2,
		},
		{
			name:    "v1 without memory.stat falls back to usage",
			version: CgroupV1,
			files:   map[string]string{cgroupV1MemoryUsagePath: "2147483648"},
			want:    2,
		},
		{
			name:    "malformed inactive value falls back to usage",
			version: CgroupV2,
			files: map[string]string{
				cgroupMemoryCurrentPath: "2147483648",
				cgroupMemoryStatPath:    "inactive_file not-a-number\n",
			},
			want: 2,
		},
		{
			name:    "inactive above usage stays at zero",
			version: CgroupV2,
			files: map[string]string{
				cgroupMemoryCurrentPath: "1024",
				cgroupMemoryStatPath:    "inactive_file 4096\n",
			},
			want: 0,
		},
		{
			name:    "v2 missing usage",
			version: CgroupV2,
			files:   map[string]string{cgroupMemoryStatPath: "inactive_file 1024\n"},
			want:    0,
		},
		{
			name:    "v1 malformed usage",
			version: CgroupV1,
			files:   map[string]string{cgroupV1MemoryUsagePath: "not-a-number"},
			want:    0,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := updateContainerMemUsage(test.version, MockFileReader{files: test.files})
			if math.Abs(got-test.want) > 1e-9 {
				t.Fatalf("updateContainerMemUsage() = %v, want %v", got, test.want)
			}
		})
	}
}
