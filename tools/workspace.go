// Package tools provides ready-made agent tools: read-only access to a
// directory tree, and an opt-in shell.
package tools

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
)

const (
	maxReadLines   = 2000
	maxFileBytes   = 2 << 20
	maxListEntries = 500
	maxGrepMatches = 100
)

// skipDirs are never walked by glob and grep.
var skipDirs = map[string]bool{".git": true, "node_modules": true, "vendor": true, ".venv": true, "__pycache__": true}

// Workspace serves read-only file tools confined to one directory.
type Workspace struct {
	root string
}

// NewWorkspace roots the tools at dir.
func NewWorkspace(dir string) (*Workspace, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return nil, err
	}
	return &Workspace{root: real}, nil
}

// Root is the workspace directory.
func (w *Workspace) Root() string { return w.root }

// resolve maps a model-supplied path to an absolute path inside the root,
// following symlinks, and rejects anything that escapes it.
func (w *Workspace) resolve(p string) (string, error) {
	if p == "" {
		p = "."
	}
	if filepath.IsAbs(p) {
		rel, err := filepath.Rel(w.root, p)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return "", fmt.Errorf("%s is outside the workspace", p)
		}
		p = rel
	}
	full := filepath.Join(w.root, filepath.Clean("/"+p))
	real, err := filepath.EvalSymlinks(full)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("%s does not exist", p)
		}
		return "", err
	}
	if real != w.root && !strings.HasPrefix(real, w.root+string(filepath.Separator)) {
		return "", fmt.Errorf("%s is outside the workspace", p)
	}
	return real, nil
}

func (w *Workspace) rel(full string) string {
	r, err := filepath.Rel(w.root, full)
	if err != nil {
		return full
	}
	return filepath.ToSlash(r)
}

func object(props map[string]any, required ...string) map[string]any {
	if required == nil {
		required = []string{}
	}
	return map[string]any{"type": "object", "properties": props, "required": required}
}

func prop(typ, desc string) map[string]any { return map[string]any{"type": typ, "description": desc} }

func decode[T any](raw json.RawMessage) (T, error) {
	var v T
	if err := json.Unmarshal(raw, &v); err != nil {
		return v, fmt.Errorf("invalid input: %v", err)
	}
	return v, nil
}

// Tools returns read_file, list_dir, glob and grep.
func (w *Workspace) Tools() []agent.Tool {
	return []agent.Tool{
		{
			Spec: llm.ToolSpec{Name: "read_file", Description: "Read a text file in the workspace, with line numbers. Use offset and limit for large files.",
				Schema: object(map[string]any{
					"path":   prop("string", "Path relative to the workspace"),
					"offset": prop("integer", "First line, 1-based (default 1)"),
					"limit":  prop("integer", fmt.Sprintf("Number of lines (default and max %d)", maxReadLines)),
				}, "path")},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[struct {
					Path          string
					Offset, Limit int
				}](raw)
				if err != nil {
					return "", err
				}
				return w.readFile(in.Path, in.Offset, in.Limit)
			},
		},
		{
			Spec: llm.ToolSpec{Name: "list_dir", Description: "List a directory in the workspace (directories end with /).",
				Schema: object(map[string]any{"path": prop("string", "Directory relative to the workspace (default: the root)")})},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[struct{ Path string }](raw)
				if err != nil {
					return "", err
				}
				return w.listDir(in.Path)
			},
		},
		{
			Spec: llm.ToolSpec{Name: "glob", Description: "Find files by glob pattern relative to the workspace, e.g. **/*.go or cmd/*/main.go.",
				Schema: object(map[string]any{"pattern": prop("string", "Glob; ** matches across directories")}, "pattern")},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[struct{ Pattern string }](raw)
				if err != nil {
					return "", err
				}
				return w.glob(ctx, in.Pattern)
			},
		},
		{
			Spec: llm.ToolSpec{Name: "grep", Description: "Search file contents with a regular expression (RE2). Returns path:line:text.",
				Schema: object(map[string]any{
					"pattern":     prop("string", "Regular expression"),
					"path":        prop("string", "Directory or file to search (default: the root)"),
					"include":     prop("string", "Only files whose path matches this glob, e.g. **/*.go"),
					"ignore_case": prop("boolean", "Case-insensitive match"),
				}, "pattern")},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[struct {
					Pattern, Path, Include string
					IgnoreCase             bool `json:"ignore_case"`
				}](raw)
				if err != nil {
					return "", err
				}
				return w.grep(ctx, in.Pattern, in.Path, in.Include, in.IgnoreCase)
			},
		},
	}
}

func (w *Workspace) readFile(p string, offset, limit int) (string, error) {
	full, err := w.resolve(p)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(full)
	if err != nil {
		return "", err
	}
	if st.IsDir() {
		return "", fmt.Errorf("%s is a directory; use list_dir", p)
	}
	f, err := os.Open(full)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if offset < 1 {
		offset = 1
	}
	if limit < 1 || limit > maxReadLines {
		limit = maxReadLines
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	var b strings.Builder
	n := 0
	for sc.Scan() {
		n++
		if n >= offset && n < offset+limit {
			if strings.ContainsRune(sc.Text(), 0) {
				return "", fmt.Errorf("%s looks like a binary file", p)
			}
			fmt.Fprintf(&b, "%d: %s\n", n, sc.Text())
		}
	}
	if err := sc.Err(); err != nil {
		return "", err
	}
	fmt.Fprintf(&b, "(%d lines in file)\n", n)
	return b.String(), nil
}

func (w *Workspace) listDir(p string) (string, error) {
	full, err := w.resolve(p)
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(full)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for i, e := range entries {
		if i >= maxListEntries {
			fmt.Fprintf(&b, "… %d more\n", len(entries)-i)
			break
		}
		name := e.Name()
		if e.IsDir() {
			name += "/"
		}
		b.WriteString(name + "\n")
	}
	if len(entries) == 0 {
		return "(empty)", nil
	}
	return b.String(), nil
}

// globRegexp converts a glob with ** into a regular expression over
// slash-separated relative paths.
func globRegexp(pattern string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("^")
	p := filepath.ToSlash(pattern)
	for i := 0; i < len(p); i++ {
		switch c := p[i]; c {
		case '*':
			if i+1 < len(p) && p[i+1] == '*' {
				i++
				if i+1 < len(p) && p[i+1] == '/' {
					i++
					b.WriteString("(?:.*/)?")
				} else {
					b.WriteString(".*")
				}
			} else {
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	b.WriteString("$")
	return regexp.Compile(b.String())
}

// walk visits regular files under dir (skipping skipDirs), stopping early
// when fn returns false.
func (w *Workspace) walk(ctx context.Context, dir string, fn func(full, rel string) bool) error {
	stop := errors.New("stop")
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if d.IsDir() {
			if path != dir && skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if !fn(path, w.rel(path)) {
			return stop
		}
		return nil
	})
	if errors.Is(err, stop) {
		return nil
	}
	return err
}

func (w *Workspace) glob(ctx context.Context, pattern string) (string, error) {
	re, err := globRegexp(pattern)
	if err != nil {
		return "", fmt.Errorf("invalid pattern: %v", err)
	}
	var hits []string
	truncated := false
	err = w.walk(ctx, w.root, func(_, rel string) bool {
		if re.MatchString(rel) {
			if len(hits) >= maxListEntries {
				truncated = true
				return false
			}
			hits = append(hits, rel)
		}
		return true
	})
	if err != nil {
		return "", err
	}
	sort.Strings(hits)
	if len(hits) == 0 {
		return "No files match.", nil
	}
	out := strings.Join(hits, "\n")
	if truncated {
		out += fmt.Sprintf("\n[first %d matches; narrow the pattern]", maxListEntries)
	}
	return out, nil
}

func (w *Workspace) grep(ctx context.Context, pattern, p, include string, ignoreCase bool) (string, error) {
	if ignoreCase {
		pattern = "(?i)" + pattern
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return "", fmt.Errorf("invalid pattern: %v", err)
	}
	var inc *regexp.Regexp
	if include != "" {
		if inc, err = globRegexp(include); err != nil {
			return "", fmt.Errorf("invalid include: %v", err)
		}
	}
	start, err := w.resolve(p)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	matches := 0
	err = w.walk(ctx, start, func(full, rel string) bool {
		if inc != nil && !inc.MatchString(rel) {
			return true
		}
		if st, err := os.Stat(full); err != nil || st.Size() > maxFileBytes {
			return true
		}
		data, err := os.ReadFile(full)
		if err != nil || strings.ContainsRune(string(data[:min(len(data), 8000)]), 0) {
			return true
		}
		for i, line := range strings.Split(string(data), "\n") {
			if re.MatchString(line) {
				matches++
				if matches > maxGrepMatches {
					return false
				}
				fmt.Fprintf(&b, "%s:%d:%s\n", rel, i+1, line)
			}
		}
		return true
	})
	if err != nil {
		return "", err
	}
	if matches == 0 {
		return "No matches.", nil
	}
	if matches > maxGrepMatches {
		fmt.Fprintf(&b, "[first %d matches; narrow the search]\n", maxGrepMatches)
	}
	return b.String(), nil
}
