package tools

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/eysteinn/senctl-agent/agent"
)

// bigLog writes a 30000-line log (about 2.6 MB, past the old grep size
// cap), plus a gzip copy, and returns the workspace tools.
func bigLog(t *testing.T) (*Workspace, map[string]agent.Tool) {
	t.Helper()
	dir := t.TempDir()
	var b bytes.Buffer
	for i := 1; i <= 30000; i++ {
		level := "INFO"
		switch {
		case i%1000 == 0:
			level = "ERROR"
		case i%100 == 0:
			level = "WARN"
		}
		fmt.Fprintf(&b, "2026-09-30T10:%02d:%02d %s svc-%d request %d handled\n", i/60%60, i%60, level, i%3, i)
		if i == 20000 {
			fmt.Fprintf(&b, "%s NEEDLE %s\n", strings.Repeat("a", 5000), strings.Repeat("b", 5000))
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "app.log"), b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	var z bytes.Buffer
	zw := gzip.NewWriter(&z)
	zw.Write(b.Bytes())
	zw.Close()
	if err := os.WriteFile(filepath.Join(dir, "app.log.1.gz"), z.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "crlf.txt"), []byte("one\r\ntwo\r\nthree"), 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := NewWorkspace(dir)
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]agent.Tool{}
	for _, tl := range w.Tools() {
		m[tl.Spec.Name] = tl
	}
	return w, m
}

func TestLargeFiles(t *testing.T) {
	_, tl := bigLog(t)
	tests := []struct {
		name, tool string
		in         any
		want       []string
		notWant    []string
	}{
		{"first page is bounded", "read_file", map[string]any{"path": "app.log"},
			[]string{"1: 2026-09-30T10:00:01 INFO", "(showing lines 1-", "of 30001; next page: offset="}, []string{"\n2000: "}},
		{"seek deep into the file", "read_file", map[string]any{"path": "app.log", "offset": 12345, "limit": 2},
			[]string{"12345: 2026-09-30T10:25:45 INFO svc-0 request 12345 handled\n12346: "}, nil},
		{"negative offset reads the end", "read_file", map[string]any{"path": "app.log", "offset": -2},
			[]string{"30000: ", "30001: 2026-09-30T10:20:00 ERROR svc-0 request 30000 handled\n(30001 lines in file)"}, nil},
		{"long lines are cut", "read_file", map[string]any{"path": "app.log", "offset": 20001, "limit": 1},
			[]string{"20001: aaaa", "… [+8008 bytes]"}, []string{"NEEDLE"}},
		{"gzip is read", "read_file", map[string]any{"path": "app.log.1.gz", "offset": 29999, "limit": 1},
			[]string{"29999: 2026-09-30T10:19:58 INFO svc-1 request 29998 handled"}, nil},
		{"crlf and no final newline", "read_file", map[string]any{"path": "crlf.txt"},
			[]string{"1: one\n2: two\n3: three\n(3 lines in file)"}, []string{"\r"}},
		{"file_info", "file_info", map[string]any{"path": "app.log"},
			[]string{"lines: 30001 (longest 9.8 KB)", "first line: 2026-09-30T10:00:01 INFO", "last line: 2026-09-30T10:20:00 ERROR", "too large to read whole"}, nil},
		{"grep searches big files", "grep", map[string]any{"pattern": "ERROR", "path": "app.log", "limit": 2},
			[]string{"app.log:1000:", "app.log:2000:", "[matches 1-2 of 30 (30 matches in 1 files); next page: offset=2]"}, nil},
		{"grep next page", "grep", map[string]any{"pattern": "ERROR", "path": "app.log", "offset": 29},
			[]string{"app.log:30001:", "[matches 30-30 of 30]"}, []string{"app.log:1000:"}},
		{"grep count", "grep", map[string]any{"pattern": "WARN|ERROR", "output_mode": "count"},
			[]string{"app.log:300\n", "app.log.1.gz:300\n", "total: 600 matches in 2 files"}, nil},
		{"grep files", "grep", map[string]any{"pattern": "NEEDLE", "output_mode": "files"},
			[]string{"app.log\napp.log.1.gz\n"}, nil},
		{"grep context", "grep", map[string]any{"pattern": "request (1000|3000) ", "path": "app.log", "context": 1},
			[]string{"app.log-999-", "app.log:1000:", "app.log-1001-", "--\napp.log-2999-"}, nil},
		{"grep long line window", "grep", map[string]any{"pattern": "NEEDLE", "path": "app.log"},
			[]string{"app.log:20001:…aaa", "NEEDLE bbb", "… [line is 10008 bytes]"}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := call(t, tl[tt.tool], tt.in)
			if err != nil {
				t.Fatal(err)
			}
			for _, w := range tt.want {
				if !strings.Contains(out, w) {
					t.Errorf("missing %q in:\n%s", w, out)
				}
			}
			for _, w := range tt.notWant {
				if strings.Contains(out, w) {
					t.Errorf("unexpected %q in:\n%s", w, out)
				}
			}
			if len(out) > 40000 {
				t.Errorf("output is %d bytes", len(out))
			}
		})
	}
}

func TestLineReader(t *testing.T) {
	long := strings.Repeat("x", 200<<10)
	lr := newLineReader(strings.NewReader("a\r\n"+long+"\nlast"), 10)
	var got []string
	for {
		line, n, err := lr.next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, fmt.Sprintf("%s/%d", line, n))
	}
	want := []string{"a/1", "xxxxxxxxxx/204800", "last/4"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestPipeline(t *testing.T) {
	for _, c := range []string{"grep", "sort", "uniq", "cut", "head", "tail"} {
		if _, err := exec.LookPath(c); err != nil {
			t.Skipf("%s is not installed", c)
		}
	}
	_, tl := bigLog(t)
	ok := []struct{ path, command, want string }{
		{"app.log", "grep -c ERROR", "30\n"},
		{"app.log", `grep -E 'WARN|ERROR' | cut -d' ' -f2 | sort | uniq -c | sort -rn | head -n 5`, "270 WARN\n     30 ERROR\n"},
		{"app.log.1.gz", "tail -n 1", "ERROR svc-0 request 30000 handled\n"},
		{"app.log", "head -1", "request 1 handled\n"},
		{"app.log", "grep -e NOPE", "(no output)\n[grep exited 1]"},
		{"app.log", `grep --max-count 2 --regexp=ERROR | wc -l`, "2\n"},
	}
	for _, tt := range ok {
		out, err := call(t, tl["pipeline"], map[string]any{"path": tt.path, "command": tt.command})
		if err != nil || !strings.HasSuffix(strings.TrimSpace(out), strings.TrimSpace(tt.want)) {
			t.Errorf("%s: out = %q, err = %v", tt.command, out, err)
		}
	}
	refused := []struct{ command, want string }{
		{"grep x /etc/passwd", "too many arguments"},
		{"grep -e x /etc/passwd", "too many arguments"},
		{"grep -f /etc/passwd", "-f is not allowed"},
		{"sort -o out.txt", "-o is not allowed"},
		{"sort --compress-program=sh", "--compress-program is not allowed"},
		{"sort --compress=sh", "--compress is not allowed"},
		{"sort --out=x", "--out is not allowed"},
		{"grep --fi=/etc/passwd", "--fi is not allowed"},
		{"uniq - /tmp/out", "too many arguments"},
		{"tail -f", "-f is not allowed"},
		{"cat /etc/passwd", "cat is not available"},
		{"grep x; rm -rf /", `';' is not supported`},
		{"grep x > out", `'>' is not supported`},
		{`grep "$HOME"`, `'$' is not supported`},
		{"grep $(id)", `'$' is not supported`},
		{"grep *", `'*' is not supported`},
		{"grep x | | wc", "empty command"},
		{`jq 'import "x" as y; .'`, "jq modules"},
		{"jq --rawfile x /etc/passwd .", "--rawfile is not allowed"},
	}
	for _, tt := range refused {
		_, err := call(t, tl["pipeline"], map[string]any{"path": "app.log", "command": tt.command})
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: err = %v, want %q", tt.command, err, tt.want)
		}
	}
	if _, err := call(t, tl["pipeline"], map[string]any{"path": "../../etc/passwd", "command": "head"}); err == nil {
		t.Error("read outside the workspace")
	}
}

func TestCacheReadable(t *testing.T) {
	w, tl := setup(t)
	c, err := NewCache(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	path, err := c.Spill(context.Background(), "shell", "alpha\nbeta\n")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := call(t, tl["read_file"], map[string]any{"path": path}); err == nil || !strings.Contains(err.Error(), "outside the workspace") {
		t.Fatalf("cache readable before AllowRead: %v", err)
	}
	if err := w.AllowRead(c.Dir()); err != nil {
		t.Fatal(err)
	}
	out, err := call(t, tl["read_file"], map[string]any{"path": path})
	if err != nil || !strings.Contains(out, "2: beta") {
		t.Fatalf("out = %q, err = %v", out, err)
	}
	out, err = call(t, tl["grep"], map[string]any{"pattern": "beta", "path": path})
	if err != nil || out != path+":2:beta\n" {
		t.Fatalf("grep out = %q, err = %v", out, err)
	}
	// A symlink in the cache still cannot reach outside it.
	if err := os.Symlink("/etc/hostname", filepath.Join(c.Dir(), "link")); err != nil {
		t.Fatal(err)
	}
	if _, err := call(t, tl["read_file"], map[string]any{"path": filepath.Join(c.Dir(), "link")}); err == nil || !strings.Contains(err.Error(), "outside the workspace") {
		t.Fatalf("symlink escape: %v", err)
	}
	// Editing stays confined to the workspace.
	for _, et := range w.EditTools(nil) {
		if et.Spec.Name == "write_file" {
			if _, err := call(t, et, map[string]any{"path": path, "content": "x"}); err == nil {
				t.Fatal("wrote into the cache")
			}
		}
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(c.Dir()); !os.IsNotExist(err) {
		t.Fatalf("cache not removed: %v", err)
	}
}
