package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/eysteinn/senctl-agent/agent"
)

func editTools(t *testing.T, approve Approver) (*Workspace, map[string]agent.Tool) {
	t.Helper()
	w, _ := setup(t)
	m := map[string]agent.Tool{}
	for _, tl := range w.EditTools(approve) {
		m[tl.Spec.Name] = tl
	}
	return w, m
}

func TestEditTools(t *testing.T) {
	var seen []Change
	w, tl := editTools(t, func(c Change) bool { seen = append(seen, c); return !strings.Contains(c.Path, "deny") })
	read := func(p string) string {
		b, err := os.ReadFile(filepath.Join(w.Root(), p))
		if err != nil {
			return "ERR:" + err.Error()
		}
		return string(b)
	}

	tests := []struct {
		name, tool string
		in         map[string]any
		want       string
		wantErr    string
		check      func() bool
	}{
		{"create nested", "write_file", map[string]any{"path": "new/dir/a.txt", "content": "one\ntwo\n"}, "Created new/dir/a.txt", "",
			func() bool { return read("new/dir/a.txt") == "one\ntwo\n" }},
		{"overwrite", "write_file", map[string]any{"path": "new/dir/a.txt", "content": "three\n"}, "Updated", "",
			func() bool { return read("new/dir/a.txt") == "three\n" }},
		{"write outside", "write_file", map[string]any{"path": "/tmp/senctl-agent-escape.txt", "content": "x"}, "", "outside the workspace", nil},
		{"write through symlink escape", "write_file", map[string]any{"path": "escape/pwned.txt", "content": "x"}, "", "outside the workspace", nil},
		{"write over dir", "write_file", map[string]any{"path": "internal", "content": "x"}, "", "is a directory", nil},
		{"declined", "write_file", map[string]any{"path": "deny.txt", "content": "x"}, "", "declined", func() bool { return strings.HasPrefix(read("deny.txt"), "ERR:") }},
		{"edit", "edit_file", map[string]any{"path": "main.go", "old_string": "run()", "new_string": "run(ctx)"}, "1 replacement", "",
			func() bool { return strings.Contains(read("main.go"), "func main() { run(ctx) }") }},
		{"edit missing text", "edit_file", map[string]any{"path": "main.go", "old_string": "nope", "new_string": "x"}, "", "not found", nil},
		{"edit ambiguous", "edit_file", map[string]any{"path": "internal/run/run.go", "old_string": "Run", "new_string": "Start"}, "", "occurs 2 times", nil},
		{"edit replace all", "edit_file", map[string]any{"path": "internal/run/run.go", "old_string": "Run", "new_string": "Start", "replace_all": true}, "2 replacement", "",
			func() bool { return !strings.Contains(read("internal/run/run.go"), "Run") }},
		{"edit missing file", "edit_file", map[string]any{"path": "nope.go", "old_string": "a", "new_string": "b"}, "", "does not exist", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := call(t, tl[tt.tool], tt.in)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, out = %q", err, out)
				}
			} else if err != nil || !strings.Contains(out, tt.want) {
				t.Fatalf("out = %q, err = %v", out, err)
			}
			if tt.check != nil && !tt.check() {
				t.Fatal("file content check failed")
			}
		})
	}
	if _, err := os.Stat("/tmp/senctl-agent-escape.txt"); err == nil {
		t.Fatal("a file was written outside the workspace")
	}
	// The approver saw diff-style previews.
	var editPreview string
	for _, c := range seen {
		if c.Tool == "edit_file" && c.Path == "main.go" {
			editPreview = c.Preview
		}
	}
	if editPreview != "- run()\n+ run(ctx)\n" {
		t.Fatalf("edit preview = %q", editPreview)
	}
}
