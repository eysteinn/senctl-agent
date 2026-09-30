package cli

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

// maxInline is the most input put into a prompt directly; larger input is
// saved to the cache for the model to search.
const maxInline = 32 << 10

// lineCounter counts the lines written through it.
type lineCounter struct {
	w     io.Writer
	lines int
	last  byte
}

func (c *lineCounter) Write(p []byte) (int, error) {
	c.lines += bytes.Count(p, []byte("\n"))
	if len(p) > 0 {
		c.last = p[len(p)-1]
	}
	return c.w.Write(p)
}

// withInput adds piped input to prompt: inline when small, otherwise saved
// to the cache with a note and its first lines.
func (e *env) withInput(prompt string, r io.Reader) (string, error) {
	head := make([]byte, maxInline+1)
	n, err := io.ReadFull(r, head)
	head = head[:n]
	if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
		return "", err
	}
	if n <= maxInline {
		if s := strings.TrimSpace(string(head)); s != "" {
			prompt = strings.TrimSpace(prompt + "\n\n<stdin>\n" + s + "\n</stdin>")
		}
		return prompt, nil
	}
	f, err := e.cache.Create("stdin")
	if err != nil {
		return "", err
	}
	lc := &lineCounter{w: f}
	size, err := io.Copy(lc, io.MultiReader(bytes.NewReader(head), r))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", fmt.Errorf("saving standard input: %w", err)
	}
	lines := lc.lines
	if lc.last != '\n' {
		lines++
	}
	preview := firstLines(head, 40, 4000)
	return strings.TrimSpace(fmt.Sprintf("%s\n\n<stdin>\nStandard input is too large to include (%d bytes, %d lines). It is saved at %s: look at it with file_info, grep, pipeline and read_file. It begins:\n%s\n</stdin>",
		prompt, size, lines, f.Name(), preview)), nil
}

// firstLines returns up to n lines and max bytes from the start of data.
func firstLines(data []byte, n, max int) string {
	data = data[:min(len(data), max)]
	lines := bytes.SplitAfterN(data, []byte("\n"), n+1)
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.TrimRight(string(bytes.Join(lines, nil)), "\n")
}

var mention = regexp.MustCompile(`(^|\s)@(\S+)`)

// attachMentions adds files named as @path in a console message: small text
// files inline, others as a description the model can follow up with tools.
func (e *env) attachMentions(ctx context.Context, text string) string {
	var parts []string
	seen := map[string]bool{}
	for _, m := range mention.FindAllStringSubmatch(text, -1) {
		p := m[2]
		full, err := e.ws.Resolve(p)
		if err != nil {
			p = strings.TrimRight(p, ".,;:!?)'\"")
			if full, err = e.ws.Resolve(p); err != nil {
				continue
			}
		}
		st, err := os.Stat(full)
		if err != nil || st.IsDir() || seen[full] {
			continue
		}
		seen[full] = true
		if st.Size() <= maxInline {
			data, err := os.ReadFile(full)
			if err == nil && !bytes.Contains(data, []byte{0}) {
				parts = append(parts, fmt.Sprintf("<file path=%q>\n%s\n</file>", p, strings.TrimRight(string(data), "\n")))
				continue
			}
		}
		if info, err := e.ws.FileInfo(ctx, p); err == nil {
			parts = append(parts, fmt.Sprintf("<file path=%q>\nNot included, it is too large. Look at it with grep, pipeline and read_file.\n%s</file>", p, info))
		}
	}
	if len(parts) == 0 {
		return text
	}
	return text + "\n\n" + strings.Join(parts, "\n\n")
}
