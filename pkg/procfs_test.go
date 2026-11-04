package main

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/shirou/gopsutil/v4/process"
)

// fakeProcess describes one process directory in a fake /proc tree.
type fakeProcess struct {
	pid        int
	comm       string
	ppid       int
	userTicks  int
	sysTicks   int
	threads    int
	startTicks int
	rssPages   int
	uid        int
	cmdline    string
	openFiles  int
}

func (p fakeProcess) stat() string {
	// pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt
	// utime stime cutime cstime priority nice num_threads itrealvalue starttime vsize rss ...
	return strconv.Itoa(p.pid) + " (" + p.comm + ") S " + strconv.Itoa(p.ppid) +
		" 1 1 0 -1 4194560 100 0 0 0 " +
		strconv.Itoa(p.userTicks) + " " + strconv.Itoa(p.sysTicks) + " 0 0 20 0 " +
		strconv.Itoa(p.threads) + " 0 " + strconv.Itoa(p.startTicks) + " 1000000 " +
		strconv.Itoa(p.rssPages) + " 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 3 0 0 0 0 0\n"
}

func writeFakeProc(t *testing.T, root string, processes ...fakeProcess) {
	t.Helper()
	writeFile := func(path, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeFile(filepath.Join(root, "stat"), "cpu  1 2 3 4\nbtime 1700000000\nprocesses 10\n")
	for _, p := range processes {
		dir := filepath.Join(root, strconv.Itoa(p.pid))
		writeFile(filepath.Join(dir, "stat"), p.stat())
		writeFile(filepath.Join(dir, "status"), "Name:\t"+p.comm+"\nUid:\t"+strconv.Itoa(p.uid)+"\t0\t0\t0\n")
		writeFile(filepath.Join(dir, "cmdline"), p.cmdline)
		if err := os.MkdirAll(filepath.Join(dir, "fd"), 0o755); err != nil {
			t.Fatal(err)
		}
		for fd := range p.openFiles {
			writeFile(filepath.Join(dir, "fd", strconv.Itoa(fd)), "")
		}
	}
}

func newTestProcfsReader(root string, users map[string]string, lookups *int) *procfsReader {
	reader := newProcfsReader(root)
	reader.pageSize = 4096
	reader.lookupUser = func(uid string) (string, error) {
		*lookups++
		if name, ok := users[uid]; ok {
			return name, nil
		}
		return "", errors.New("unknown uid")
	}
	return reader
}

func processesByPID(t *testing.T, reader *procfsReader) map[int32]ProcessHandle {
	t.Helper()
	handles, err := reader.Processes()
	if err != nil {
		t.Fatalf("Processes() error = %v", err)
	}
	byPID := make(map[int32]ProcessHandle, len(handles))
	for _, handle := range handles {
		byPID[handle.PID()] = handle
	}
	return byPID
}

func TestProcfsReaderReadsProcessFields(t *testing.T) {
	root := t.TempDir()
	writeFakeProc(t, root,
		fakeProcess{
			pid: 42, comm: "my (odd) proc", ppid: 7, userTicks: 250, sysTicks: 50, threads: 3,
			startTicks: 12345, rssPages: 256, uid: 1000, cmdline: "python3\x00train.py\x00--fast\x00",
		},
		fakeProcess{pid: 2, comm: "kthreadd", startTicks: 1, uid: 0},
	)
	lookups := 0
	reader := newTestProcfsReader(root, map[string]string{"1000": "alice", "0": "root"}, &lookups)

	byPID := processesByPID(t, reader)
	if len(byPID) != 2 {
		t.Fatalf("expected 2 processes, got %d", len(byPID))
	}
	proc := byPID[42]
	ppid, _ := proc.ParentPID()
	times, _ := proc.Times()
	memory, _ := proc.MemoryInfo()
	threads, _ := proc.NumThreads()
	created, _ := proc.CreateTime()
	name, _ := proc.Name()
	cmdline, _ := proc.Cmdline()
	user, _ := proc.Username()

	if ppid != 7 || threads != 3 || name != "my (odd) proc" {
		t.Errorf("ppid=%d threads=%d name=%q", ppid, threads, name)
	}
	if math.Abs(times.User-2.5) > 1e-9 || math.Abs(times.System-0.5) > 1e-9 {
		t.Errorf("expected 2.5 s user and 0.5 s system, got %+v", times)
	}
	if memory.RSS != 256*4096 {
		t.Errorf("expected RSS %d, got %d", 256*4096, memory.RSS)
	}
	if want := int64(1700000000)*1000 + 123450; created != want {
		t.Errorf("expected create time %d, got %d", want, created)
	}
	if cmdline != "python3 train.py --fast" {
		t.Errorf("expected NUL-separated arguments joined by spaces, got %q", cmdline)
	}
	if user != "alice" {
		t.Errorf("expected user alice, got %q", user)
	}
	if kernelCmdline, _ := byPID[2].Cmdline(); kernelCmdline != "" {
		t.Errorf("expected an empty command line for a kernel thread, got %q", kernelCmdline)
	}
}

func TestProcfsReaderSkipsUnreadableAndNonProcessEntries(t *testing.T) {
	root := t.TempDir()
	writeFakeProc(t, root, fakeProcess{pid: 10, comm: "ok", uid: 0})
	for _, name := range []string{"self", "sys", "0"} {
		if err := os.MkdirAll(filepath.Join(root, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// A process that exited between listing and reading has no stat file.
	if err := os.MkdirAll(filepath.Join(root, "11"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A truncated stat file is skipped instead of producing zeros.
	if err := os.MkdirAll(filepath.Join(root, "12"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "12", "stat"), []byte("12 (short) S 1 1"), 0o600); err != nil {
		t.Fatal(err)
	}

	lookups := 0
	byPID := processesByPID(t, newTestProcfsReader(root, nil, &lookups))
	if len(byPID) != 1 || byPID[10] == nil {
		t.Fatalf("expected only PID 10, got %v", byPID)
	}
}

func TestProcfsReaderCachesUsersAndRereadsReusedPIDs(t *testing.T) {
	root := t.TempDir()
	writeFakeProc(t, root,
		fakeProcess{pid: 10, comm: "a", startTicks: 100, uid: 1000},
		fakeProcess{pid: 11, comm: "b", startTicks: 100, uid: 1000},
		fakeProcess{pid: 12, comm: "c", startTicks: 100, uid: 4242},
	)
	lookups := 0
	reader := newTestProcfsReader(root, map[string]string{"1000": "alice", "2000": "bob"}, &lookups)

	users := func() map[int32]string {
		out := map[int32]string{}
		for pid, proc := range processesByPID(t, reader) {
			out[pid], _ = proc.Username()
		}
		return out
	}
	first := users()
	if first[10] != "alice" || first[11] != "alice" || first[12] != "4242" {
		t.Fatalf("unexpected users %v", first)
	}
	if lookups != 2 {
		t.Fatalf("expected one lookup per uid, got %d", lookups)
	}

	// The user of a running process is read once; its status file is not needed again.
	if err := os.Remove(filepath.Join(root, "10", "status")); err != nil {
		t.Fatal(err)
	}
	if second := users(); second[10] != "alice" || lookups != 2 {
		t.Fatalf("expected cached user and no new lookups, got %v after %d lookups", second, lookups)
	}

	// A new process that reuses PID 11 has a different start time, so its user is read again.
	writeFakeProc(t, root, fakeProcess{pid: 11, comm: "b2", startTicks: 900, uid: 2000})
	if third := users(); third[11] != "bob" {
		t.Fatalf("expected the reused PID to show its new user, got %v", third)
	}
}

func TestCountOpenFilesCountsDescriptors(t *testing.T) {
	root := t.TempDir()
	writeFakeProc(t, root, fakeProcess{pid: 10, comm: "a", openFiles: 5})
	if got := countOpenFiles(root, 10); got != 5 {
		t.Fatalf("expected 5 open files, got %d", got)
	}
	if got := countOpenFiles(root, 99); got != 0 {
		t.Fatalf("expected 0 for a missing process, got %d", got)
	}
}

func TestReadProcessStartTimeMatchesProcessList(t *testing.T) {
	root := t.TempDir()
	writeFakeProc(t, root, fakeProcess{pid: 10, comm: "a", startTicks: 777})
	lookups := 0
	created, _ := processesByPID(t, newTestProcfsReader(root, nil, &lookups))[10].CreateTime()
	start, err := readProcessStartTime(root, 10)
	if err != nil || start != created {
		t.Fatalf("expected start time %d to match the process list, got %d (%v)", created, start, err)
	}
}

// TestProcfsReaderMatchesGopsutil compares the reader with gopsutil for this test process.
func TestProcfsReaderMatchesGopsutil(t *testing.T) {
	pid := int32(os.Getpid())
	proc, ok := processesByPID(t, newProcfsReader(procRoot))[pid]
	if !ok {
		t.Fatalf("test process %d not found in %s", pid, procRoot)
	}
	reference, err := process.NewProcess(pid)
	if err != nil {
		t.Fatal(err)
	}

	gotPPID, _ := proc.ParentPID()
	wantPPID, _ := reference.Ppid()
	gotCmdline, _ := proc.Cmdline()
	wantCmdline, _ := reference.Cmdline()
	gotName, _ := proc.Name()
	wantName, _ := reference.Name()
	gotUser, _ := proc.Username()
	wantUser, wantUserErr := reference.Username()
	if gotPPID != wantPPID || gotCmdline != wantCmdline || gotName != wantName {
		t.Errorf("ppid %d/%d cmdline %q/%q name %q/%q", gotPPID, wantPPID, gotCmdline, wantCmdline, gotName, wantName)
	}
	if wantUserErr == nil && gotUser != wantUser {
		t.Errorf("user %q, gopsutil %q", gotUser, wantUser)
	}

	gotTimes, _ := proc.Times()
	wantTimes, _ := reference.Times()
	if gotTimes.User > wantTimes.User+0.01 || gotTimes.System > wantTimes.System+0.01 {
		t.Errorf("times %+v read before gopsutil's %+v but are larger", gotTimes, wantTimes)
	}
	gotMemory, _ := proc.MemoryInfo()
	wantMemory, _ := reference.MemoryInfo()
	if diff := math.Abs(float64(gotMemory.RSS) - float64(wantMemory.RSS)); diff > 0.2*float64(wantMemory.RSS) {
		t.Errorf("RSS %d differs from gopsutil's %d by more than 20%%", gotMemory.RSS, wantMemory.RSS)
	}
	gotCreated, _ := proc.CreateTime()
	wantCreated, _ := reference.CreateTime()
	if diff := gotCreated - wantCreated; diff < -1000 || diff > 1000 {
		t.Errorf("create time %d differs from gopsutil's %d by %d ms", gotCreated, wantCreated, diff)
	}
	if !strings.Contains(gotCmdline, "test") {
		t.Errorf("expected the test binary's command line, got %q", gotCmdline)
	}
}
