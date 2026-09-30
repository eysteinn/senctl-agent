package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
)

// Change describes a file change awaiting approval.
type Change struct {
	Tool    string // write_file or edit_file
	Path    string // relative to the workspace
	New     bool   // the file does not exist yet
	Preview string // a short diff-style summary for the user
}

// Approver decides whether a change may be made. nil approves everything.
type Approver func(Change) bool

// maxWriteBytes bounds one write.
const maxWriteBytes = 4 << 20

// resolveForWrite maps a path to a location inside the workspace for
// writing: the file may not exist yet, but its nearest existing parent must
// resolve (through symlinks) inside the root.
func (w *Workspace) resolveForWrite(p string) (string, error) {
	if p == "" {
		return "", errors.New("path is required")
	}
	if filepath.IsAbs(p) {
		rel, err := filepath.Rel(w.root, p)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return "", fmt.Errorf("%s is outside the workspace", p)
		}
		p = rel
	}
	full := filepath.Join(w.root, filepath.Clean("/"+p))
	if full == w.root {
		return "", fmt.Errorf("%s is a directory", p)
	}
	// Walk up to the nearest existing ancestor and check where it really is.
	dir := full
	for {
		if _, err := os.Lstat(dir); err == nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return "", err
	}
	if real != w.root && !strings.HasPrefix(real, w.root+string(filepath.Separator)) {
		return "", fmt.Errorf("%s is outside the workspace", p)
	}
	if dir == full {
		// The target exists: it must be a regular file (or a symlink to one
		// inside the workspace, checked above).
		st, err := os.Stat(full)
		if err != nil {
			return "", err
		}
		if st.IsDir() {
			return "", fmt.Errorf("%s is a directory", p)
		}
		return real, nil
	}
	return full, nil
}

// preview renders removed and added lines, capped for display.
func preview(oldText, newText string) string {
	var b strings.Builder
	lines := 0
	emit := func(prefix, text string) {
		for _, l := range strings.Split(strings.TrimSuffix(text, "\n"), "\n") {
			if lines >= 40 {
				return
			}
			b.WriteString(prefix + l + "\n")
			lines++
		}
	}
	if oldText != "" {
		emit("- ", oldText)
	}
	emit("+ ", newText)
	if lines >= 40 {
		b.WriteString("  …\n")
	}
	return b.String()
}

// EditTools returns write_file and edit_file. Each change is shown to
// approve (when set) before it is made; a refused change is reported back
// to the model as declined.
func (w *Workspace) EditTools(approve Approver) []agent.Tool {
	ask := func(c Change) error {
		if approve != nil && !approve(c) {
			return errors.New("the user declined this change")
		}
		return nil
	}
	return []agent.Tool{
		{
			Spec: llm.ToolSpec{Name: "write_file", Description: "Create a file, or replace a file's entire content. Parent directories are created. Prefer edit_file for changes to existing files.",
				Schema: object(map[string]any{
					"path":    prop("string", "Path relative to the workspace"),
					"content": prop("string", "The complete new file content"),
				}, "path", "content")},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[struct{ Path, Content string }](raw)
				if err != nil {
					return "", err
				}
				if len(in.Content) > maxWriteBytes {
					return "", fmt.Errorf("content is larger than %d bytes", maxWriteBytes)
				}
				full, err := w.resolveForWrite(in.Path)
				if err != nil {
					return "", err
				}
				old, readErr := os.ReadFile(full)
				isNew := errors.Is(readErr, fs.ErrNotExist)
				if readErr != nil && !isNew {
					return "", readErr
				}
				if err := ask(Change{Tool: "write_file", Path: w.rel(full), New: isNew, Preview: preview(string(old), in.Content)}); err != nil {
					return "", err
				}
				if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
					return "", err
				}
				mode := fs.FileMode(0o644)
				if st, err := os.Stat(full); err == nil {
					mode = st.Mode().Perm()
				}
				if err := os.WriteFile(full, []byte(in.Content), mode); err != nil {
					return "", err
				}
				verb := "Updated"
				if isNew {
					verb = "Created"
				}
				lines := strings.Count(in.Content, "\n")
				if in.Content != "" && !strings.HasSuffix(in.Content, "\n") {
					lines++
				}
				unit := "lines"
				if lines == 1 {
					unit = "line"
				}
				return fmt.Sprintf("%s %s (%d %s).", verb, w.rel(full), lines, unit), nil
			},
		},
		{
			Spec: llm.ToolSpec{Name: "edit_file", Description: "Replace exact text in an existing file. old_string must match the file exactly (including whitespace) and be unique unless replace_all is true. Read the file first.",
				Schema: object(map[string]any{
					"path":        prop("string", "Path relative to the workspace"),
					"old_string":  prop("string", "Exact text to replace"),
					"new_string":  prop("string", "Replacement text"),
					"replace_all": prop("boolean", "Replace every occurrence (default false)"),
				}, "path", "old_string", "new_string")},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[struct {
					Path       string
					OldString  string `json:"old_string"`
					NewString  string `json:"new_string"`
					ReplaceAll bool   `json:"replace_all"`
				}](raw)
				if err != nil {
					return "", err
				}
				if in.OldString == "" {
					return "", errors.New("old_string is empty; use write_file to create a file")
				}
				if in.OldString == in.NewString {
					return "", errors.New("old_string and new_string are identical")
				}
				full, err := w.resolveForWrite(in.Path)
				if err != nil {
					return "", err
				}
				data, err := os.ReadFile(full)
				if err != nil {
					if errors.Is(err, fs.ErrNotExist) {
						return "", fmt.Errorf("%s does not exist", in.Path)
					}
					return "", err
				}
				content := string(data)
				n := strings.Count(content, in.OldString)
				switch {
				case n == 0:
					return "", errors.New("old_string was not found; read the file and copy the text exactly")
				case n > 1 && !in.ReplaceAll:
					return "", fmt.Errorf("old_string occurs %d times; add surrounding context to make it unique or set replace_all", n)
				}
				if err := ask(Change{Tool: "edit_file", Path: w.rel(full), Preview: preview(in.OldString, in.NewString)}); err != nil {
					return "", err
				}
				updated := strings.Replace(content, in.OldString, in.NewString, map[bool]int{true: -1, false: 1}[in.ReplaceAll])
				st, err := os.Stat(full)
				if err != nil {
					return "", err
				}
				if err := os.WriteFile(full, []byte(updated), st.Mode().Perm()); err != nil {
					return "", err
				}
				return fmt.Sprintf("Edited %s (%d replacement(s)).", w.rel(full), map[bool]int{true: n, false: 1}[in.ReplaceAll]), nil
			},
		},
	}
}
