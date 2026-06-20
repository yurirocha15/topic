package main

import "testing"

func BenchmarkMakeBar(b *testing.B) {
	for range b.N {
		_ = makeBar(72.5, 80)
	}
}

func BenchmarkCalculateBarWidth(b *testing.B) {
	labels := []string{
		"CPU: [yellow]72.5  %[white] ",
		"MEM: [yellow]61.2  %[white] ",
		"DISK /data:    [yellow]48.8%[white]",
	}
	infos := []string{
		" [darkcyan](limit: 4.00 CPUs)[white]",
		" [darkcyan]3.672 GB / 6.000 GB[white]",
		"[darkcyan]48.80 GB / 100.00 GB[white]",
	}

	for range b.N {
		_ = calculateBarWidth(180, labels, infos)
	}
}

func BenchmarkBuildStorageSection(b *testing.B) {
	labels := []string{
		"DISK /:        [yellow] 38.2%[white]",
		"DISK /data:    [yellow] 71.4%[white]",
		"DISK /logs:    [yellow] 12.9%[white]",
		"DISK /cache:   [yellow] 55.0%[white]",
	}
	infos := []string{
		"[darkcyan]38.20 GB / 100.00 GB[white]",
		"[darkcyan]71.40 GB / 100.00 GB[white]",
		"[darkcyan]12.90 GB / 100.00 GB[white]",
		"[darkcyan]55.00 GB / 100.00 GB[white]",
	}
	percentages := []float64{38.2, 71.4, 12.9, 55.0}
	maxLabelWidth, maxInfoWidth := calculateMaxWidthsFromSlices(labels, infos)
	layout := BarLayout{
		Columns:       2,
		BarWidth:      40,
		TotalWidth:    150,
		MaxLabelWidth: maxLabelWidth,
		MaxInfoWidth:  maxInfoWidth,
	}

	for range b.N {
		_ = buildStorageSection(layout, labels, infos, percentages)
	}
}

func BenchmarkGetGPUProcessMap(b *testing.B) {
	runner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi pmon -c 1 -s um": `# gpu     pid  type    sm   mem   enc   dec   command
    0   20131     C    15     8     -     -   python
    0   20567     C     -     -     -     -   -
    1   34567     G    22    15     -     -   Xorg`,
		},
	}

	for range b.N {
		_ = getGPUProcessMap(runner)
	}
}

func BenchmarkUpdateLiveGPUUsage(b *testing.B) {
	runner := MockCommandRunner{
		outputs: map[string]string{
			"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits": "50, 2048\n75, 4096\n",
		},
	}

	for range b.N {
		_ = updateLiveGPUUsage(2, runner)
	}
}

func BenchmarkUpdateContainerMemUsage(b *testing.B) {
	fs := MockFileReader{
		files: map[string]string{
			cgroupMemoryStatPath: "anon 1073741824\nfile 536870912\nkernel_stack 16384\n",
		},
	}

	for range b.N {
		_ = updateContainerMemUsage(CgroupV2, fs)
	}
}
