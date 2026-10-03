package tools

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/eysteinn/senctl-agent/agent"
)

func setup(t *testing.T) (*Workspace, map[string]agent.Tool) {
	t.Helper()
	dir := t.TempDir()
	write := func(p, s string) {
		full := filepath.Join(dir, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(s), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("main.go", "package main\n\nfunc main() { run() }\n")
	write("internal/run/run.go", "package run\n\n// Run starts the server.\nfunc Run() {}\n")
	write("node_modules/x/index.js", "run()\n")
	write("bin.dat", "a\x00b")
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret"), []byte("s3cr3t"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "escape")); err != nil {
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

func call(t *testing.T, tl agent.Tool, in any) (string, error) {
	t.Helper()
	b, _ := json.Marshal(in)
	return tl.Run(context.Background(), b)
}

func TestWorkspaceTools(t *testing.T) {
	_, tl := setup(t)
	tests := []struct {
		name, tool string
		in         any
		want       string
		wantErr    string
	}{
		{"read", "read_file", map[string]any{"path": "internal/run/run.go"}, "3: // Run starts the server.", ""},
		{"read range", "read_file", map[string]any{"path": "main.go", "offset": 3, "limit": 1}, "3: func main() { run() }\n(3 lines in file)", ""},
		{"read dotdot stays inside", "read_file", map[string]any{"path": "../../../../etc/passwd"}, "", "does not exist"},
		{"read symlink escape", "read_file", map[string]any{"path": "escape/secret"}, "", "outside the workspace"},
		{"read absolute outside", "read_file", map[string]any{"path": "/etc/hostname"}, "", "outside the workspace"},
		{"read dir", "read_file", map[string]any{"path": "internal"}, "", "is a directory"},
		{"read binary", "read_file", map[string]any{"path": "bin.dat"}, "", "binary"},
		{"list root", "list_dir", map[string]any{}, "internal/", ""},
		{"glob", "glob", map[string]any{"pattern": "**/*.go"}, "internal/run/run.go\nmain.go", ""},
		{"glob skips node_modules", "glob", map[string]any{"pattern": "**/*.js"}, "No files match.", ""},
		{"grep", "grep", map[string]any{"pattern": "RUN\\(", "ignore_case": true, "include": "**/*.go"}, "main.go:3:func main() { run() }", ""},
		{"grep in dir", "grep", map[string]any{"pattern": "Run", "path": "internal"}, "internal/run/run.go:3:", ""},
		{"grep none", "grep", map[string]any{"pattern": "zzz"}, "No matches.", ""},
		{"grep bad regexp", "grep", map[string]any{"pattern": "("}, "", "invalid pattern"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := call(t, tl[tt.tool], tt.in)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, out = %q", err, out)
				}
				return
			}
			if err != nil || !strings.Contains(out, tt.want) {
				t.Fatalf("out = %q, err = %v", out, err)
			}
		})
	}
}

func TestShell(t *testing.T) {
	dir := t.TempDir()
	sh := Shell(dir, time.Second, nil)
	out, err := call(t, sh, map[string]any{"command": "pwd; echo oops >&2; exit 3"})
	if err != nil || !strings.Contains(out, "exit code 3") || !strings.Contains(out, dir) || !strings.Contains(out, "oops") {
		t.Fatalf("out = %q, err = %v", out, err)
	}
	if _, err := call(t, sh, map[string]any{"command": "sleep 5"}); err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("timeout err = %v", err)
	}
	deny := Shell(dir, time.Second, func(string) bool { return false })
	if _, err := call(t, deny, map[string]any{"command": "echo hi"}); err == nil || !strings.Contains(err.Error(), "declined") {
		t.Fatalf("declined err = %v", err)
	}
}

func TestNewWorkspaceErrors(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "file.txt")
	if err := os.WriteFile(file, []byte("hi\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		dir    string
		target error
		msg    string
	}{
		{filepath.Join(dir, "missing"), fs.ErrNotExist, "does not exist"},
		{file, ErrNotDir, "is not a directory"},
	} {
		_, err := NewWorkspace(tt.dir)
		var we *WorkspaceError
		if !errors.As(err, &we) || !errors.Is(err, tt.target) || err.Error() != "workspace "+tt.dir+" "+tt.msg {
			t.Errorf("NewWorkspace(%s) = %v", tt.dir, err)
		}
	}
	if _, err := NewWorkspace(dir); err != nil {
		t.Errorf("NewWorkspace(dir) = %v", err)
	}
}
