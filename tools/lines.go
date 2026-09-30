package tools

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	// indexStep is how many lines apart the line index records offsets.
	indexStep = 1000
	// sniffBytes is how much of a file is checked for NUL bytes.
	sniffBytes = 8000
)

// isGzip reports whether a path names a gzip-compressed file.
func isGzip(full string) bool { return strings.HasSuffix(strings.ToLower(full), ".gz") }

// openText opens a file for line reading, decompressing .gz files.
func openText(full string) (io.ReadCloser, error) {
	f, err := os.Open(full)
	if err != nil {
		return nil, err
	}
	if !isGzip(full) {
		return f, nil
	}
	zr, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: not valid gzip: %v", full, err)
	}
	return struct {
		io.Reader
		io.Closer
	}{zr, f}, nil
}

// looksBinary reports whether data (the start of a file) contains NUL.
func looksBinary(data []byte) bool {
	return bytes.IndexByte(data[:min(len(data), sniffBytes)], 0) >= 0
}

// lineReader reads lines of any length, keeping at most keep bytes of each.
type lineReader struct {
	r    *bufio.Reader
	keep int
	buf  []byte
}

func newLineReader(r io.Reader, keep int) *lineReader {
	return &lineReader{r: bufio.NewReaderSize(r, 64<<10), keep: keep}
}

// next returns the next line without its line ending, cut to keep bytes, and
// the line's full length. It returns io.EOF after the last line.
func (lr *lineReader) next() ([]byte, int, error) {
	lr.buf = lr.buf[:0]
	n := 0
	for {
		chunk, err := lr.r.ReadSlice('\n')
		n += len(chunk)
		if room := lr.keep - len(lr.buf); room > 0 {
			lr.buf = append(lr.buf, chunk[:min(room, len(chunk))]...)
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if err != nil && !errors.Is(err, io.EOF) {
			return nil, 0, err
		}
		ended := len(chunk) > 0 && chunk[len(chunk)-1] == '\n'
		if !ended && n == 0 {
			return nil, 0, io.EOF
		}
		if ended {
			n--
			lr.buf = lr.buf[:min(len(lr.buf), n)]
		}
		if n > 0 && len(lr.buf) == n && lr.buf[n-1] == '\r' {
			n--
			lr.buf = lr.buf[:n]
		}
		return lr.buf, n, nil
	}
}

// clip returns line cut to max bytes on a rune boundary, noting what was cut.
func clip(line []byte, full, max int) string {
	if full <= max && len(line) == full {
		return string(line)
	}
	cut := min(len(line), max)
	for cut > 0 && cut < len(line) && !utf8.RuneStart(line[cut]) {
		cut--
	}
	return fmt.Sprintf("%s… [+%d bytes]", line[:cut], full-cut)
}

// lineIndex describes a file's lines, so later reads can seek instead of
// scanning from the start.
type lineIndex struct {
	size    int64
	mod     time.Time
	lines   int
	longest int
	binary  bool
	// marks[k] is the byte offset of line k*indexStep+1 (plain files only).
	marks []int64
}

// indexCache remembers line indexes by path, dropping them when the file
// changes.
type indexCache struct {
	mu sync.Mutex
	m  map[string]*lineIndex
}

func (c *indexCache) get(ctx context.Context, full string) (*lineIndex, error) {
	st, err := os.Stat(full)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	idx := c.m[full]
	c.mu.Unlock()
	if idx != nil && idx.size == st.Size() && idx.mod.Equal(st.ModTime()) {
		return idx, nil
	}
	idx, err = buildIndex(ctx, full)
	if err != nil {
		return nil, err
	}
	idx.size, idx.mod = st.Size(), st.ModTime()
	c.mu.Lock()
	if c.m == nil {
		c.m = map[string]*lineIndex{}
	}
	c.m[full] = idx
	c.mu.Unlock()
	return idx, nil
}

func buildIndex(ctx context.Context, full string) (*lineIndex, error) {
	r, err := openText(full)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	idx := &lineIndex{marks: []int64{0}}
	buf := make([]byte, 1<<20)
	var pos, start int64
	first := true
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := r.Read(buf)
		chunk := buf[:n]
		if first && n > 0 {
			idx.binary = looksBinary(chunk)
			first = false
		}
		for off := 0; ; {
			i := bytes.IndexByte(chunk[off:], '\n')
			if i < 0 {
				break
			}
			end := pos + int64(off+i)
			idx.longest = max(idx.longest, int(end-start))
			idx.lines++
			start = end + 1
			if idx.lines%indexStep == 0 {
				idx.marks = append(idx.marks, start)
			}
			off += i + 1
		}
		pos += int64(n)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	if start < pos {
		idx.lines++
		idx.longest = max(idx.longest, int(pos-start))
	}
	if isGzip(full) {
		idx.marks = nil
	}
	return idx, nil
}

// openAt opens a file positioned at the start of line (1-based), using the
// index to seek where it can. It returns the reader and the line it is
// positioned at, which may be earlier than asked; the caller skips ahead.
func openAt(full string, idx *lineIndex, line int) (io.ReadCloser, int, error) {
	r, err := openText(full)
	if err != nil {
		return nil, 0, err
	}
	k := (line - 1) / indexStep
	if idx == nil || len(idx.marks) == 0 || k == 0 {
		return r, 1, nil
	}
	k = min(k, len(idx.marks)-1)
	f := r.(*os.File)
	if _, err := f.Seek(idx.marks[k], io.SeekStart); err != nil {
		f.Close()
		return nil, 0, err
	}
	return f, k*indexStep + 1, nil
}

// humanBytes formats a byte count for people and models.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
