package tools

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"
	"unicode/utf8"
)

// maxMatchLine is how much of one line grep keeps for matching.
const maxMatchLine = 1 << 20

type grepInput struct {
	Pattern, Path, Include string
	IgnoreCase             bool   `json:"ignore_case"`
	OutputMode             string `json:"output_mode"`
	Context                int
	Offset, Limit          int
}

// grepState collects results across files. A result is a matching line in
// content mode and a matching file otherwise; offset and limit page them.
type grepState struct {
	in      grepInput
	re      *regexp.Regexp
	lits    []literal // see prefilter
	b       strings.Builder
	results int // results seen so far
	shown   int
	full    bool // the page is full; keep counting only
	files   int  // files with a match
	matches int
}

func (g *grepState) wants() bool {
	return !g.full && g.results > g.in.Offset && g.shown < g.in.Limit
}

func (g *grepState) emit(s string) {
	g.b.WriteString(s)
	if g.b.Len() >= maxGrepBytes {
		g.full = true
	}
}

// matchLine renders a matching line, cut to a window around the match.
func matchLine(re *regexp.Regexp, line []byte, length int) string {
	if length <= maxGrepLine && len(line) == length {
		return string(line)
	}
	loc := re.FindIndex(line)
	start := 0
	if loc != nil {
		start = max(0, loc[0]-maxGrepLine/3)
	}
	for start > 0 && !utf8.RuneStart(line[start]) {
		start--
	}
	end := min(len(line), start+maxGrepLine)
	for end < len(line) && !utf8.RuneStart(line[end]) {
		end--
	}
	s := string(line[start:end])
	if start > 0 {
		s = "…" + s
	}
	return s + fmt.Sprintf("… [line is %d bytes]", length)
}

func (w *Workspace) grep(ctx context.Context, in grepInput) (string, error) {
	pattern := in.Pattern
	if in.IgnoreCase {
		pattern = "(?i)" + pattern
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return "", fmt.Errorf("invalid pattern: %v", err)
	}
	var inc *regexp.Regexp
	if in.Include != "" {
		if inc, err = globRegexp(in.Include); err != nil {
			return "", fmt.Errorf("invalid include: %v", err)
		}
	}
	switch in.OutputMode {
	case "":
		in.OutputMode = "content"
	case "content", "files", "count":
	default:
		return "", fmt.Errorf("output_mode must be content, files or count")
	}
	in.Context = min(max(in.Context, 0), maxGrepContext)
	in.Offset = max(in.Offset, 0)
	if in.Limit < 1 {
		in.Limit = maxGrepMatches
	}
	in.Limit = min(in.Limit, maxGrepLimit)
	start, err := w.resolve(in.Path)
	if err != nil {
		return "", err
	}
	g := &grepState{in: in, re: re, lits: prefilter(pattern)}
	var fileErr error
	err = w.walk(ctx, start, func(full, rel string) bool {
		if inc != nil && !inc.MatchString(rel) {
			return true
		}
		if err := g.file(ctx, full, rel); err != nil {
			if ctx.Err() != nil {
				fileErr = ctx.Err()
				return false
			}
			// Unreadable files are skipped, as a recursive grep would.
		}
		return true
	})
	if fileErr != nil {
		return "", fileErr
	}
	if err != nil {
		return "", err
	}
	if g.matches == 0 {
		return "No matches.", nil
	}
	unit := map[string]string{"content": "matches", "files": "files", "count": "files"}[in.OutputMode]
	if in.OutputMode == "count" {
		fmt.Fprintf(&g.b, "total: %d matches in %d files\n", g.matches, g.files)
	}
	switch first := in.Offset + 1; {
	case g.shown == 0:
		fmt.Fprintf(&g.b, "[no %s past offset %d; there are %d]\n", unit, in.Offset, g.results)
	case in.Offset+g.shown < g.results:
		fmt.Fprintf(&g.b, "[%s %d-%d of %d (%d matches in %d files); next page: offset=%d]\n",
			unit, first, in.Offset+g.shown, g.results, g.matches, g.files, in.Offset+g.shown)
	case in.Offset > 0:
		fmt.Fprintf(&g.b, "[%s %d-%d of %d]\n", unit, first, in.Offset+g.shown, g.results)
	}
	return g.b.String(), nil
}

// file searches one file, skipping binaries.
func (g *grepState) file(ctx context.Context, full, rel string) error {
	r, err := openText(full)
	if err != nil {
		return err
	}
	defer r.Close()
	lr := newLineReader(r, maxMatchLine)
	if head, _ := lr.r.Peek(sniffBytes); looksBinary(head) {
		return nil
	}
	type held struct {
		n    int
		text string
	}
	var before []held
	after, printed, count := 0, 0, 0
	mode := g.in.OutputMode
	for n := 1; ; n++ {
		if n%4096 == 0 && ctx.Err() != nil {
			return ctx.Err()
		}
		line, length, err := lr.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if !mayMatch(g.lits, line) || !g.re.Match(line) {
			if after > 0 && mode == "content" {
				g.emit(fmt.Sprintf("%s-%d-%s\n", rel, n, clip(line, length, maxGrepLine)))
				printed, after = n, after-1
			} else if g.in.Context > 0 {
				before = append(before, held{n, clip(line, length, maxGrepLine)})
				if len(before) > g.in.Context {
					before = before[1:]
				}
			}
			continue
		}
		count++
		g.matches++
		if count == 1 {
			g.files++
			if mode != "content" {
				g.results++
			}
			if mode == "files" {
				if g.wants() {
					g.emit(rel + "\n")
					g.shown++
				}
				return nil
			}
		}
		if mode != "content" {
			continue
		}
		g.results++
		if !g.wants() {
			before, after = before[:0], 0
			continue
		}
		if g.in.Context > 0 && g.shown > 0 && (printed == 0 || len(before) > 0 && before[0].n > printed+1) {
			g.emit("--\n")
		}
		for _, h := range before {
			g.emit(fmt.Sprintf("%s-%d-%s\n", rel, h.n, h.text))
		}
		before = before[:0]
		g.emit(fmt.Sprintf("%s:%d:%s\n", rel, n, matchLine(g.re, line, length)))
		g.shown++
		printed, after = n, g.in.Context
	}
	if mode == "count" && count > 0 && g.wants() {
		g.emit(fmt.Sprintf("%s:%d\n", rel, count))
		g.shown++
	}
	return nil
}
