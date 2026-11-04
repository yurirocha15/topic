package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"strconv"

	"github.com/shirou/gopsutil/v4/cpu"
	"github.com/shirou/gopsutil/v4/process"
)

// The process list is read straight from /proc. Each refresh reads /proc/<pid>/stat and
// /proc/<pid>/cmdline; /proc/<pid>/status is read once per process for its user, and user
// names are cached by uid, so a refresh costs a few small reads per process.

const (
	procRoot = "/proc"
	// clockTicksPerSecond is USER_HZ, the unit of the times in /proc/<pid>/stat. Linux fixes it at 100.
	clockTicksPerSecond = 100
	// procStatBufferSize fits a /proc/<pid>/stat or status file; larger files grow the buffer.
	procStatBufferSize = 4096
	millisecondsPerSec = 1000
)

// Field positions in /proc/<pid>/stat, counted from the state field that follows the command name.
const (
	procStatParentField  = 1
	procStatUserField    = 11
	procStatSystemField  = 12
	procStatThreadsField = 17
	procStatStartField   = 19
	procStatRSSField     = 21
)

var errProcStatFormat = errors.New("unexpected /proc stat format")

// procfsReader lists processes from a proc filesystem. It is not safe for concurrent use; the
// collector reads processes from one goroutine at a time.
type procfsReader struct {
	root       string
	lookupUser func(uid string) (string, error)
	pageSize   uint64

	bootTimeMillis int64
	uids           map[procfsKey]uint32
	userNames      map[uint32]string
	buffer         []byte
}

// procfsKey identifies a process across refreshes, because PIDs can be reused.
type procfsKey struct {
	pid        int32
	startTicks int64
}

func newProcfsReader(root string) *procfsReader {
	return &procfsReader{
		root:       root,
		lookupUser: lookupUsername,
		pageSize:   uint64(os.Getpagesize()), //nolint:gosec // The page size is a small positive number.
		uids:       map[procfsKey]uint32{},
		userNames:  map[uint32]string{},
		buffer:     make([]byte, procStatBufferSize),
	}
}

// processReader returns the reader that lists processes, keeping its caches between refreshes.
func (s *State) processReader() *procfsReader {
	if s.procfs == nil {
		s.procfs = newProcfsReader(procRoot)
	}
	return s.procfs
}

func lookupUsername(uid string) (string, error) {
	account, err := user.LookupId(uid)
	if err != nil {
		return "", fmt.Errorf("look up uid %s: %w", uid, err)
	}
	return account.Username, nil
}

// Processes reads every process in one pass. Processes that exit while being read are skipped.
func (r *procfsReader) Processes() ([]ProcessHandle, error) {
	dir, err := os.Open(r.root)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", r.root, err)
	}
	names, err := dir.Readdirnames(-1)
	_ = dir.Close()
	if err != nil {
		return nil, fmt.Errorf("list %s: %w", r.root, err)
	}

	bootTime := r.bootTime()
	seen := make(map[procfsKey]uint32, len(r.uids))
	handles := make([]ProcessHandle, 0, len(names))
	for _, name := range names {
		pid, parseErr := strconv.ParseInt(name, 10, 32)
		if parseErr != nil || pid <= 0 {
			continue
		}
		data, readErr := r.read(filepath.Join(r.root, name, "stat"))
		if readErr != nil {
			continue
		}
		proc, statErr := parseProcStat(data)
		if statErr != nil {
			continue
		}
		proc.reader = r
		proc.pid = int32(pid)
		proc.rssBytes = proc.rssPages * r.pageSize
		proc.createTime = bootTime + proc.startTicks*millisecondsPerSec/clockTicksPerSecond
		if uid, ok := r.uids[proc.key()]; ok {
			seen[proc.key()] = uid
		}
		handles = append(handles, proc)
	}
	// Forget users of processes that have exited, so the cache stays the size of the process list.
	r.uids = seen
	return handles, nil
}

// bootTime returns the boot time in milliseconds from btime in /proc/stat, read once per run.
func (r *procfsReader) bootTime() int64 {
	if r.bootTimeMillis == 0 {
		r.bootTimeMillis = readBootTimeMillis(r.root)
	}
	return r.bootTimeMillis
}

func readBootTimeMillis(root string) int64 {
	data, err := os.ReadFile(filepath.Join(root, "stat"))
	if err != nil {
		return 0
	}
	for line := range bytes.Lines(data) {
		value, found := bytes.CutPrefix(line, []byte("btime "))
		if !found {
			continue
		}
		seconds, parseErr := strconv.ParseInt(string(bytes.TrimSpace(value)), 10, 64)
		if parseErr != nil {
			return 0
		}
		return seconds * millisecondsPerSec
	}
	return 0
}

// read returns the contents of a small /proc file in the reader's reused buffer, which stays valid
// until the next read.
func (r *procfsReader) read(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err //nolint:wrapcheck // Callers only check whether the file could be read.
	}
	defer func() { _ = file.Close() }()
	size := 0
	for {
		if size == len(r.buffer) {
			r.buffer = append(r.buffer, make([]byte, len(r.buffer))...)
		}
		count, readErr := file.Read(r.buffer[size:])
		size += count
		if readErr != nil || count == 0 {
			break
		}
	}
	return r.buffer[:size], nil
}

// parseProcStat reads the fields topic uses from /proc/<pid>/stat. The command name can contain
// spaces and parentheses, so fields are counted from the last closing parenthesis.
func parseProcStat(data []byte) (*procfsProcess, error) {
	open := bytes.IndexByte(data, '(')
	closing := bytes.LastIndexByte(data, ')')
	if open < 0 || closing < open || closing+2 > len(data) {
		return nil, errProcStatFormat
	}
	proc := &procfsProcess{name: string(data[open+1 : closing])}
	field := 0
	for value := range bytes.FieldsSeq(data[closing+1:]) {
		var err error
		switch field {
		case procStatParentField:
			var parent int64
			parent, err = strconv.ParseInt(string(value), 10, 32)
			proc.ppid = int32(parent)
		case procStatUserField:
			proc.userTicks, err = strconv.ParseUint(string(value), 10, 64)
		case procStatSystemField:
			proc.systemTicks, err = strconv.ParseUint(string(value), 10, 64)
		case procStatThreadsField:
			var threads int64
			threads, err = strconv.ParseInt(string(value), 10, 32)
			proc.threads = int32(threads)
		case procStatStartField:
			proc.startTicks, err = strconv.ParseInt(string(value), 10, 64)
		case procStatRSSField:
			proc.rssPages, err = strconv.ParseUint(string(value), 10, 64)
			if err != nil {
				return nil, errProcStatFormat
			}
			return proc, nil
		}
		if err != nil {
			return nil, errProcStatFormat
		}
		field++
	}
	return nil, errProcStatFormat
}

// username returns the name of the process's real user. The uid is read once per process and
// names are cached per uid; a uid with no account shows as its number, like ps does.
func (r *procfsReader) username(key procfsKey) (string, error) {
	uid, ok := r.uids[key]
	if !ok {
		data, err := r.read(filepath.Join(r.root, strconv.Itoa(int(key.pid)), "status"))
		if err != nil {
			return "", err
		}
		uid, err = parseStatusUID(data)
		if err != nil {
			return "", err
		}
		r.uids[key] = uid
	}
	if name, cached := r.userNames[uid]; cached {
		return name, nil
	}
	name, err := r.lookupUser(strconv.FormatUint(uint64(uid), 10))
	if err != nil {
		name = strconv.FormatUint(uint64(uid), 10)
	}
	r.userNames[uid] = name
	return name, nil
}

// parseStatusUID returns the real uid from the "Uid:" line of /proc/<pid>/status.
func parseStatusUID(data []byte) (uint32, error) {
	for line := range bytes.Lines(data) {
		value, found := bytes.CutPrefix(line, []byte("Uid:"))
		if !found {
			continue
		}
		for field := range bytes.FieldsSeq(value) {
			uid, err := strconv.ParseUint(string(field), 10, 32)
			if err != nil {
				return 0, errProcStatFormat
			}
			return uint32(uid), nil
		}
	}
	return 0, errProcStatFormat
}

// countOpenFiles counts a process's open file descriptors without resolving where they point.
func countOpenFiles(root string, pid int32) int {
	dir, err := os.Open(filepath.Join(root, strconv.Itoa(int(pid)), "fd"))
	if err != nil {
		return 0
	}
	defer func() { _ = dir.Close() }()
	names, err := dir.Readdirnames(-1)
	if err != nil {
		return 0
	}
	return len(names)
}

// readProcessStartTime returns a process's start time in milliseconds, computed the same way the
// process list computes it, so the two can be compared exactly.
func readProcessStartTime(root string, pid int32) (int64, error) {
	data, err := os.ReadFile(filepath.Join(root, strconv.Itoa(int(pid)), "stat"))
	if err != nil {
		return 0, fmt.Errorf("read process stat: %w", err)
	}
	proc, err := parseProcStat(data)
	if err != nil {
		return 0, err
	}
	return readBootTimeMillis(root) + proc.startTicks*millisecondsPerSec/clockTicksPerSecond, nil
}

// procfsProcess holds one process's values from /proc/<pid>/stat. Its command line and user are
// read when asked for.
type procfsProcess struct {
	reader      *procfsReader
	pid         int32
	ppid        int32
	name        string
	userTicks   uint64
	systemTicks uint64
	threads     int32
	startTicks  int64
	rssPages    uint64
	rssBytes    uint64
	createTime  int64
}

func (p *procfsProcess) key() procfsKey {
	return procfsKey{pid: p.pid, startTicks: p.startTicks}
}

func (p *procfsProcess) PID() int32 {
	return p.pid
}

func (p *procfsProcess) ParentPID() (int32, error) {
	return p.ppid, nil
}

func (p *procfsProcess) Times() (*cpu.TimesStat, error) {
	return &cpu.TimesStat{
		User:   float64(p.userTicks) / clockTicksPerSecond,
		System: float64(p.systemTicks) / clockTicksPerSecond,
	}, nil
}

func (p *procfsProcess) MemoryInfo() (*process.MemoryInfoStat, error) {
	return &process.MemoryInfoStat{RSS: p.rssBytes}, nil
}

func (p *procfsProcess) Username() (string, error) {
	return p.reader.username(p.key())
}

// Cmdline reads the command line every refresh, because programs such as postgres rewrite it.
func (p *procfsProcess) Cmdline() (string, error) {
	data, err := p.reader.read(filepath.Join(p.reader.root, strconv.Itoa(int(p.pid)), "cmdline"))
	if err != nil {
		return "", err
	}
	data = bytes.TrimRight(data, "\x00")
	return string(bytes.ReplaceAll(data, []byte{0}, []byte{' '})), nil
}

func (p *procfsProcess) Name() (string, error) {
	return p.name, nil
}

func (p *procfsProcess) CreateTime() (int64, error) {
	return p.createTime, nil
}

func (p *procfsProcess) NumThreads() (int32, error) {
	return p.threads, nil
}
