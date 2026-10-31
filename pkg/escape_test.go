package main

import (
	"errors"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
	"github.com/shirou/gopsutil/v4/disk"
	"github.com/shirou/gopsutil/v4/process"
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

func TestSendProcessSignalReportsChangedProcess(t *testing.T) {
	pages := tview.NewPages()
	sendProcessSignal(pages, &MockProcessSignaler{err: errProcessChanged}, ProcessInfo{PID: 42}, syscall.SIGTERM)
	assertOnScreen(t, drawOnScreen(t, pages), "PID 42 is no longer running")
}

func TestOSProcessSignalerChecksStartTime(t *testing.T) {
	command := exec.Command("sleep", "30")
	if err := command.Start(); err != nil {
		t.Fatalf("start sleep: %v", err)
	}
	finished := make(chan error, 1)
	go func() { finished <- command.Wait() }()
	t.Cleanup(func() { _ = command.Process.Kill() })

	pid := int32(command.Process.Pid)
	handle, err := process.NewProcess(pid)
	if err != nil {
		t.Fatalf("process.NewProcess() error = %v", err)
	}
	createTime, err := handle.CreateTime()
	if err != nil {
		t.Fatalf("CreateTime() error = %v", err)
	}
	startTime := time.UnixMilli(createTime)
	signaler := OSProcessSignaler{}

	for name, stale := range map[string]time.Time{"reused pid": startTime.Add(time.Second), "unknown": {}} {
		if err = signaler.Signal(pid, stale, syscall.SIGTERM); !errors.Is(err, errProcessChanged) {
			t.Errorf("%s: Signal() error = %v, want errProcessChanged", name, err)
		}
	}
	select {
	case <-finished:
		t.Fatal("process was signalled despite a start time mismatch")
	case <-time.After(50 * time.Millisecond):
	}

	if err = signaler.Signal(pid, startTime, syscall.SIGTERM); err != nil {
		t.Fatalf("Signal() error = %v", err)
	}
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("process did not receive the signal")
	}

	if err = signaler.Signal(pid, startTime, syscall.SIGTERM); !errors.Is(err, errProcessChanged) {
		t.Errorf("Signal() after exit error = %v, want errProcessChanged", err)
	}
}

func TestStripControlCharactersRemovesFormatCharacters(t *testing.T) {
	const input = "safe\u202edr-mr\u2066x\u200by\ufeffz\u2028a\u2029b\u0085c"
	if got, want := stripControlCharacters(input), "safe dr-mr x y z a b c"; got != want {
		t.Fatalf("stripControlCharacters() = %q, want %q", got, want)
	}
}

func TestDisplayedFieldsDropControlCharacters(t *testing.T) {
	info, err := getProcessInfo(
		MockProcessHandle{user: "bo\x1b[2Jb", cmdline: "cmd"},
		&StaticInfo{},
		nil,
		0,
	)
	if err != nil {
		t.Fatalf("getProcessInfo() error = %v", err)
	}
	if strings.ContainsRune(info.User, '\x1b') {
		t.Errorf("User = %q, want no escape byte", info.User)
	}

	state := &State{
		static:  StaticInfo{HostMemTotalGB: 16},
		dynamic: DynamicInfo{StorageUsage: []StorageUsage{{Path: "/m\x1b[2Jnt"}}},
	}
	state.dynamic.NetworkUsage = []NetworkUsage{{Name: "et\x1b[2Jh0"}}
	state.dynamic.DiskIOUsage = []DiskIOUsage{{Name: "sd\x1b[2Ja"}}
	if text := onceText(state); strings.ContainsRune(text, '\x1b') {
		t.Errorf("once text contains an escape byte: %q", text)
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

func TestURLTagsAreShownLiterally(t *testing.T) {
	payloads := []string{
		"[:::https://evil.example/x]sudo apt update",
		"[red::b:https://evil.example/?a=b%20c]text",
		"[:::mailto:a@b.example]text",
		"[::b:https://evil.example/x",
		"trailing bracket [",
		"[[nested]] [abc[]",
		"[\"region\"]text",
		"[#ff0000]text",
	}
	for _, payload := range payloads {
		t.Run(payload, func(t *testing.T) {
			state := &State{
				ui:      UIState{ProcessSort: SortByCPU},
				dynamic: DynamicInfo{Processes: []ProcessInfo{{PID: 1, User: "root", Command: payload}}},
			}
			table := configureProcessTable(tview.NewTable())
			updateProcessTable(table, state)
			assertOnScreen(t, drawOnScreen(t, table), payload)

			cell := table.GetCell(1, processCommandColumn)
			if got := plainText(cell.Text); got != payload {
				t.Errorf("plainText() = %q, want %q", got, payload)
			}

			screen := tcell.NewSimulationScreen("UTF-8")
			if err := screen.Init(); err != nil {
				t.Fatalf("screen.Init() error = %v", err)
			}
			defer screen.Fini()
			screen.SetSize(120, 30)
			table.SetRect(0, 0, 120, 30)
			table.Draw(screen)
			screen.Show()
			for _, cellContent := range cellsWithLinks(screen) {
				t.Errorf("cell has a hyperlink: %q", cellContent)
			}
		})
	}
}

func cellsWithLinks(screen tcell.SimulationScreen) []string {
	cells, _, _ := screen.GetContents()
	var links []string
	for _, cell := range cells {
		if cell.Style != cell.Style.Url("") {
			links = append(links, string(cell.Runes))
		}
	}
	return links
}

func TestPlainTextWidthCountsBracketsAsText(t *testing.T) {
	for _, value := range []string{"[:::https://evil.example/x]", "[red]", "plain", "[", "]"} {
		if got := plainTextWidth(value); got != len(value) {
			t.Errorf("plainTextWidth(%q) = %d, want %d", value, got, len(value))
		}
	}
}
