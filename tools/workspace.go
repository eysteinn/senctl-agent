// Package tools provides ready-made agent tools: read-only access to a
// directory tree that copes with files far larger than a model's context,
// a read-only text pipeline, file editing, and an opt-in shell.
package tools

import (
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
	maxReadBytes   = 32000
	maxLineBytes   = 2000
	maxListEntries = 500
	maxGrepMatches = 100
	maxGrepLimit   = 500
	maxGrepBytes   = 24000
	maxGrepLine    = 400
	maxGrepContext = 10
	// largeFile is the size above which file_info suggests searching
	// rather than reading.
	largeFile = 256 << 10
)

// skipDirs are never walked by glob and grep.
var skipDirs = map[string]bool{".git": true, "node_modules": true, "vendor": true, ".venv": true, "__pycache__": true}

// Workspace serves read-only file tools confined to one directory, plus
// any extra directories allowed for reading (such as a Cache).
type Workspace struct {
	root  string
	extra []string
	idx   indexCache
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

// AllowRead lets the read-only tools also read files under dir, given by
// absolute path. Editing stays confined to the workspace root.
func (w *Workspace) AllowRead(dir string) error {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return err
	}
	w.extra = append(w.extra, real)
	return nil
}

func within(root, p string) bool {
	return p == root || strings.HasPrefix(p, root+string(filepath.Separator))
}

// resolve maps a model-supplied path to an absolute path inside the root
// (or an extra read root), following symlinks, and rejects anything that
// escapes it.
func (w *Workspace) resolve(p string) (string, error) {
	if p == "" {
		p = "."
	}
	if filepath.IsAbs(p) {
		for _, root := range w.extra {
			if clean := filepath.Clean(p); within(root, clean) {
				real, err := filepath.EvalSymlinks(clean)
				if errors.Is(err, fs.ErrNotExist) {
					return "", fmt.Errorf("%s does not exist", p)
				}
				if err != nil {
					return "", err
				}
				if !within(root, real) {
					return "", fmt.Errorf("%s is outside the workspace", p)
				}
				return real, nil
			}
		}
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
	if !within(w.root, real) {
		return "", fmt.Errorf("%s is outside the workspace", p)
	}
	return real, nil
}

// resolveFile is resolve for a path that must be a file.
func (w *Workspace) resolveFile(p string) (string, os.FileInfo, error) {
	full, err := w.resolve(p)
	if err != nil {
		return "", nil, err
	}
	st, err := os.Stat(full)
	if err != nil {
		return "", nil, err
	}
	if st.IsDir() {
		return "", nil, fmt.Errorf("%s is a directory; use list_dir", p)
	}
	return full, st, nil
}

// rel names a file for the model: relative to the workspace root, or
// absolute when it lives in an extra read root.
func (w *Workspace) rel(full string) string {
	if !within(w.root, full) {
		return full
	}
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

// Tools returns read_file, file_info, list_dir, glob, grep and (when its
// commands are installed) pipeline. Every path
// is relative to the workspace, or absolute inside a directory allowed with
// AllowRead.
func (w *Workspace) Tools() []agent.Tool {
	ts := []agent.Tool{
		{
			Spec: llm.ToolSpec{Name: "read_file", Description: fmt.Sprintf("Read a text file with line numbers, one page at a time: up to %d lines or about %d KB per call, with very long lines cut. "+
				"offset is the first line (1-based); a negative offset counts from the end, so -100 reads the last 100 lines. The footer gives the file's line count and the offset of the next page. "+
				"Files ending in .gz are decompressed. For large files, prefer grep or pipeline to find what matters, then read around it.", maxReadLines, maxReadBytes/1000),
				Schema: object(map[string]any{
					"path":   prop("string", "Path relative to the workspace, or an absolute path given by a tool"),
					"offset": prop("integer", "First line, 1-based (default 1); negative counts from the end"),
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
				return w.readFile(ctx, in.Path, in.Offset, in.Limit)
			},
		},
		{
			Spec: llm.ToolSpec{Name: "file_info", Description: "Describe a file without reading it all: size, line count, longest line, whether it is text, and its first and last lines. Use it before reading a file that may be large, such as a log.",
				Schema: object(map[string]any{"path": prop("string", "Path relative to the workspace, or an absolute path given by a tool")}, "path")},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[struct{ Path string }](raw)
				if err != nil {
					return "", err
				}
				return w.fileInfo(ctx, in.Path)
			},
		},
		{
			Spec: llm.ToolSpec{Name: "list_dir", Description: "List a directory in the workspace with file sizes (directories end with /).",
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
			Spec: llm.ToolSpec{Name: "grep", Description: "Search file contents with a regular expression (RE2), streaming through files of any size (.gz files are decompressed). " +
				"output_mode content (default) prints path:line:text, with context lines as path-line-text; files lists matching files; count prints matches per file. " +
				"Results come in pages: the footer gives the total and the offset of the next page. On big files, start with count to size the result.",
				Schema: object(map[string]any{
					"pattern":     prop("string", "Regular expression"),
					"path":        prop("string", "Directory or file to search (default: the root); absolute paths given by a tool work too"),
					"include":     prop("string", "Only files whose path matches this glob, e.g. **/*.go"),
					"ignore_case": prop("boolean", "Case-insensitive match"),
					"output_mode": map[string]any{"type": "string", "enum": []string{"content", "files", "count"}, "description": "content (default), files or count"},
					"context":     prop("integer", fmt.Sprintf("Lines of context before and after each match (content mode, max %d)", maxGrepContext)),
					"offset":      prop("integer", "Skip this many results (for the next page)"),
					"limit":       prop("integer", fmt.Sprintf("Results per page (default %d, max %d)", maxGrepMatches, maxGrepLimit)),
				}, "pattern")},
			Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
				in, err := decode[grepInput](raw)
				if err != nil {
					return "", err
				}
				return w.grep(ctx, in)
			},
		},
	}
	if t, ok := w.pipelineTool(); ok {
		ts = append(ts, t)
	}
	return ts
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
		} else if info, err := e.Info(); err == nil && info.Mode().IsRegular() {
			name += "  " + humanBytes(info.Size())
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
