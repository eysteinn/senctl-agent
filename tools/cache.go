package tools

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Cache is a directory for text too large to put in front of the model:
// tool output, piped input and the like. Files saved there are meant to be
// searched and paged through with the workspace tools (see
// Workspace.AllowRead) instead of being sent whole.
type Cache struct {
	dir string
	mu  sync.Mutex
	n   int
}

// NewCache creates a fresh cache directory under parent (the system temp
// directory when empty). Close removes it.
func NewCache(parent string) (*Cache, error) {
	if parent != "" {
		if err := os.MkdirAll(parent, 0o700); err != nil {
			return nil, err
		}
	}
	dir, err := os.MkdirTemp(parent, "session-")
	if err != nil {
		return nil, err
	}
	return &Cache{dir: dir}, nil
}

// PruneCaches removes cache directories under parent last changed before
// maxAge ago, left behind by sessions that did not exit cleanly.
func PruneCaches(parent string, maxAge time.Duration) {
	entries, err := os.ReadDir(parent)
	if err != nil {
		return
	}
	for _, e := range entries {
		info, err := e.Info()
		if err == nil && e.IsDir() && time.Since(info.ModTime()) > maxAge {
			_ = os.RemoveAll(filepath.Join(parent, e.Name()))
		}
	}
}

// Dir is the cache directory.
func (c *Cache) Dir() string { return c.dir }

// Close removes the cache directory and everything in it.
func (c *Cache) Close() error { return os.RemoveAll(c.dir) }

var unsafeName = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// Create opens a new file in the cache whose name includes label.
func (c *Cache) Create(label string) (*os.File, error) {
	c.mu.Lock()
	c.n++
	n := c.n
	c.mu.Unlock()
	label = unsafeName.ReplaceAllString(label, "_")
	if len(label) > 40 {
		label = label[:40]
	}
	return os.OpenFile(filepath.Join(c.dir, fmt.Sprintf("%03d-%s.txt", n, label)), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
}

// Save writes content to a new cache file and returns its path.
func (c *Cache) Save(label string, r io.Reader) (string, error) {
	f, err := c.Create(label)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		return "", err
	}
	return f.Name(), f.Close()
}

// Spill saves a tool's output; it fits agent.Config.Spill.
func (c *Cache) Spill(_ context.Context, tool, output string) (string, error) {
	return c.Save(tool+"-output", strings.NewReader(output))
}
