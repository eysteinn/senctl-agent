package tools

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

// readLines calls fn for lines from..to (1-based, inclusive) of full, each
// cut to keep bytes, until fn returns false.
func readLines(ctx context.Context, full string, idx *lineIndex, from, to, keep int, fn func(n int, line []byte, length int) bool) error {
	r, at, err := openAt(full, idx, from)
	if err != nil {
		return err
	}
	defer r.Close()
	lr := newLineReader(r, keep)
	for n := at; n <= to; n++ {
		if n%4096 == 0 && ctx.Err() != nil {
			return ctx.Err()
		}
		line, length, err := lr.next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if n >= from && !fn(n, line, length) {
			return nil
		}
	}
	return nil
}

func (w *Workspace) readFile(ctx context.Context, p string, offset, limit int) (string, error) {
	full, st, err := w.resolveFile(p)
	if err != nil {
		return "", err
	}
	idx, err := w.idx.get(ctx, full)
	if err != nil {
		return "", err
	}
	if idx.binary {
		return "", fmt.Errorf("%s looks like a binary file (%s)", p, humanBytes(st.Size()))
	}
	total := idx.lines
	if total == 0 {
		return "(empty file)\n", nil
	}
	start := offset
	switch {
	case offset < 0:
		start = max(1, total+offset+1)
	case offset == 0:
		start = 1
	case offset > total:
		return "", fmt.Errorf("offset %d is past the end of %s (%d lines)", offset, p, total)
	}
	if limit < 1 || limit > maxReadLines {
		limit = maxReadLines
	}
	var b strings.Builder
	last := start - 1
	err = readLines(ctx, full, idx, start, start+limit-1, maxLineBytes, func(n int, line []byte, length int) bool {
		if b.Len() >= maxReadBytes {
			return false
		}
		fmt.Fprintf(&b, "%d: %s\n", n, clip(line, length, maxLineBytes))
		last = n
		return true
	})
	if err != nil {
		return "", err
	}
	if last >= total {
		fmt.Fprintf(&b, "(%d lines in file)\n", total)
	} else {
		fmt.Fprintf(&b, "(showing lines %d-%d of %d; next page: offset=%d)\n", start, last, total, last+1)
	}
	return b.String(), nil
}

func (w *Workspace) fileInfo(ctx context.Context, p string) (string, error) {
	full, st, err := w.resolveFile(p)
	if err != nil {
		return "", err
	}
	idx, err := w.idx.get(ctx, full)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "path: %s\nsize: %s (%d bytes)", w.rel(full), humanBytes(st.Size()), st.Size())
	if isGzip(full) {
		b.WriteString(", gzip-compressed; the text tools decompress it")
	}
	fmt.Fprintf(&b, "\nmodified: %s\n", st.ModTime().UTC().Format(time.RFC3339))
	if idx.binary {
		b.WriteString("type: binary; the text tools do not read it\n")
		return b.String(), nil
	}
	fmt.Fprintf(&b, "lines: %d (longest %s)\n", idx.lines, humanBytes(int64(idx.longest)))
	if idx.lines == 0 {
		return b.String(), nil
	}
	show := func(label string, n int) error {
		return readLines(ctx, full, idx, n, n, 300, func(_ int, line []byte, length int) bool {
			fmt.Fprintf(&b, "%s: %s\n", label, clip(line, length, 300))
			return false
		})
	}
	if err := show("first line", 1); err != nil {
		return "", err
	}
	if idx.lines > 1 {
		if err := show("last line", idx.lines); err != nil {
			return "", err
		}
	}
	if st.Size() > largeFile || idx.lines > maxReadLines || idx.longest > maxLineBytes {
		b.WriteString("This file is too large to read whole. Find what matters with grep (output_mode count first) or pipeline, then read_file around it; a negative offset reads from the end.\n")
	}
	return b.String(), nil
}

// Resolve maps a path the way the read-only tools do: relative to the
// workspace, or absolute inside a directory allowed with AllowRead. It
// fails for anything outside those.
func (w *Workspace) Resolve(p string) (string, error) { return w.resolve(p) }

// FileInfo describes a file the way the file_info tool does.
func (w *Workspace) FileInfo(ctx context.Context, p string) (string, error) {
	return w.fileInfo(ctx, p)
}
