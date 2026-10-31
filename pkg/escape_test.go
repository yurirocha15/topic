package main

import (
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
	"github.com/shirou/gopsutil/v4/disk"
)

const testSpoofCommand = "[red]evil[-] --flag"

func drawOnScreen(t *testing.T, primitive tview.Primitive) string {
	t.Helper()
	screen := tcell.NewSimulationScreen("UTF-8")
	if err := screen.Init(); err != nil {
		t.Fatalf("screen.Init() error = %v", err)
	}
	defer screen.Fini()
	screen.SetSize(120, 30)
	primitive.SetRect(0, 0, 120, 30)
	primitive.Draw(screen)
	screen.Show()
	return strings.Join(simulationScreenRows(screen), "\n")
}

func assertOnScreen(t *testing.T, screenText string, expected ...string) {
	t.Helper()
	for _, value := range expected {
		if !strings.Contains(screenText, value) {
			t.Errorf("screen is missing %q:\n%s", value, screenText)
		}
	}
}

func TestProcessTableShowsUntrustedTextLiterally(t *testing.T) {
	state := &State{
		ui: UIState{ProcessSort: SortByCPU},
		dynamic: DynamicInfo{Processes: []ProcessInfo{
			{PID: 1, User: "[blue]bob", Command: testSpoofCommand},
			{PID: 2, User: "root", Command: "[kworker/0:1]"},
			{PID: 3, User: "root", Command: "[black:black]hidden"},
			{PID: 4, User: "root", Command: "[kthreadd]"},
		}},
	}
	table := configureProcessTable(tview.NewTable())
	updateProcessTable(table, state)

	screenText := drawOnScreen(t, table)
	assertOnScreen(t, screenText, testSpoofCommand, "[blue]bob", "[kworker/0:1]", "[black:black]hidden", "[kthreadd]")
}

func TestProcessTableTitleShowsUntrustedTextLiterally(t *testing.T) {
	state := &State{ui: UIState{ProcessSort: SortByCPU, Paused: true, ProcessFilter: "[red]x"}}
	table := configureProcessTable(tview.NewTable())
	updateProcessTable(table, state)

	assertOnScreen(t, drawOnScreen(t, table), "[paused]", "[filter: [red]x]")
}

func TestModalsShowUntrustedTextLiterally(t *testing.T) {
	process := ProcessInfo{PID: 1, User: "[blue]bob", Command: testSpoofCommand, GPUIndex: -1}
	pages := tview.NewPages()
	showModal(pages, "details", processDetailsText(process), []string{"OK"}, nil)

	assertOnScreen(t, drawOnScreen(t, pages), testSpoofCommand, "[blue]bob")
}

func TestFilterStatusShowsUntrustedTextLiterally(t *testing.T) {
	view := tview.NewTextView().SetDynamicColors(true).SetText(filterStatusText(UIState{ProcessFilter: "[red]x"}))
	assertOnScreen(t, drawOnScreen(t, view), "[red]x")
}

func TestMetricRowsShowUntrustedTextLiterally(t *testing.T) {
	var builder strings.Builder
	writeMetricLine(&builder, "NET [red]e0", "[blue]value", "", 80, themeAccentTag)
	builder.WriteString("\n")
	builder.WriteString(formatAlignedBar(BarData{Label: "DISK [red]/x", Info: "[blue]info"}, 10, 16, 12, 60))

	view := tview.NewTextView().SetDynamicColors(true).SetText(builder.String())
	assertOnScreen(t, drawOnScreen(t, view), "NET [red]e0", "[blue]value", "DISK [red]/x", "[blue]info")
}

func TestFormatAlignedBarWidthIgnoresEscaping(t *testing.T) {
	plain := formatAlignedBar(BarData{Label: "DISK /mntabcd", Info: "1.00 GB"}, 10, 16, 12, 60)
	bracketed := formatAlignedBar(BarData{Label: "DISK /[red]x", Info: "[N/A]"}, 10, 16, 12, 60)
	if got, want := tview.TaggedStringWidth(bracketed), tview.TaggedStringWidth(plain); got != want {
		t.Errorf("escaped row width = %d, want %d", got, want)
	}
}

func TestStorageLabelKeepsValidUTF8(t *testing.T) {
	state := &State{dynamic: DynamicInfo{StorageUsage: []StorageUsage{
		{Path: "/mnt/データストレージ領域/バックアップ"},
	}}}
	bars := calculateStorageBars(state)
	if len(bars) != 1 || !utf8.ValidString(bars[0].Label) {
		t.Fatalf("label = %q, want valid UTF-8", bars[0].Label)
	}
	const wantLabel = "DISK ...レージ領域/バックアップ"
	if bars[0].Label != wantLabel {
		t.Errorf("label = %q, want %q", bars[0].Label, wantLabel)
	}
}

func TestOnceTextHasNoDisplayMarkup(t *testing.T) {
	state := &State{
		static: StaticInfo{HostMemTotalGB: 16, ContainerCPULimit: 2},
		dynamic: DynamicInfo{
			StorageUsage: []StorageUsage{{Path: "/[red]x", UsedPercent: 75}, {Path: "/[y[]", UsedPercent: 5}},
			Processes:    []ProcessInfo{{PID: 1, Command: testSpoofCommand}},
		},
	}
	text := onceText(state)
	resources, _, _ := strings.Cut(text, "Processes:")
	for _, tag := range []string{"[white]", "[gray]", "[green]", "[gold]", "[-]"} {
		if strings.Contains(resources, tag) {
			t.Errorf("once text contains %q:\n%s", tag, resources)
		}
	}
	assertOnScreen(t, text, "CPU", "DISK /[red]x", "DISK /[y[]", testSpoofCommand)
}

func TestProcessCommandDropsControlCharacters(t *testing.T) {
	handle := MockProcessHandle{cmdline: "evil \x1b[2J\x1b]0;title\x07\nfake\tline\x7f\u0085"}
	command, err := resolveProcessCommand(handle)
	if err != nil {
		t.Fatalf("resolveProcessCommand() error = %v", err)
	}
	if want := "evil  [2J ]0;title  fake line  "; command != want {
		t.Fatalf("command = %q, want %q", command, want)
	}
}

func TestSignalIsSkippedWhenProcessChanged(t *testing.T) {
	started := time.Now()
	process := ProcessInfo{PID: 42, StartTime: started}
	tests := []struct {
		name    string
		current []ProcessInfo
		want    int
	}{
		{"same process", []ProcessInfo{{PID: 42, StartTime: started}}, 1},
		{"exited", []ProcessInfo{{PID: 7, StartTime: started}}, 0},
		{"pid reused", []ProcessInfo{{PID: 42, StartTime: started.Add(time.Second)}}, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := &State{dynamic: DynamicInfo{Processes: test.current}}
			signaler := &MockProcessSignaler{}
			pages := tview.NewPages()
			signalIfUnchanged(pages, state, signaler, process, syscall.SIGTERM)
			if len(signaler.calls) != test.want {
				t.Fatalf("signal calls = %d, want %d", len(signaler.calls), test.want)
			}
			if test.want == 0 {
				assertOnScreen(t, drawOnScreen(t, pages), "PID 42 is no longer running")
			}
		})
	}
}

type pathStater map[string]os.FileInfo

func (stater pathStater) Stat(path string) (os.FileInfo, error) {
	if info, ok := stater[path]; ok {
		return info, nil
	}
	return nil, os.ErrNotExist
}

type regularFileInfo struct{ fakeFileInfo }

func (regularFileInfo) IsDir() bool {
	return false
}

func TestStorageSkipsSingleFileMounts(t *testing.T) {
	provider := MockStorageProvider{
		partitions: []disk.PartitionStat{
			{Mountpoint: "/app/config.yaml", Fstype: "ext4"},
			{Mountpoint: "/data", Fstype: "ext4"},
			{Mountpoint: "/unreadable", Fstype: "ext4"},
		},
		usages: map[string]*disk.UsageStat{
			"/app/config.yaml": {Total: bytesPerGB},
			"/data":            {Total: bytesPerGB},
			"/unreadable":      {Total: bytesPerGB},
		},
	}
	stater := pathStater{"/app/config.yaml": regularFileInfo{}, "/data": fakeFileInfo{}}

	mounts, err := getStaticStorageInfoWithProvider(provider, stater)
	if err != nil {
		t.Fatalf("getStaticStorageInfoWithProvider() error = %v", err)
	}
	paths := make([]string, 0, len(mounts))
	for _, mount := range mounts {
		paths = append(paths, mount.Path)
	}
	if got := strings.Join(paths, ","); got != "/data,/unreadable" {
		t.Fatalf("mounts = %q, want /data,/unreadable", got)
	}
}
