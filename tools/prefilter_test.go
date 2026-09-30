package tools

import (
	"fmt"
	"strings"
	"testing"
)

func TestPrefilter(t *testing.T) {
	for pattern, want := range map[string]string{
		"ERROR":                "[ERROR]",
		"ERROR|panic":          "[ERROR panic]",
		`status=(500|503) \d`:  "[status=]",
		`time=\d+ms (GET|PUT)`: "[time=]",
		`(?i)error`:            "[{ERROR true}]",
		`(?i)fehlerü`:          "[]",
		`a.*b`:                 "[]",
		`\d+`:                  "[]",
		`err(or)?`:             "[err]",
		`foo|x`:                "[]",
		`x*`:                   "[]",
	} {
		var parts []string
		for _, l := range prefilter(pattern) {
			if l.fold {
				parts = append(parts, fmt.Sprintf("{%s true}", l.b))
			} else {
				parts = append(parts, string(l.b))
			}
		}
		if got := "[" + strings.Join(parts, " ") + "]"; got != want {
			t.Errorf("%s: got %s, want %s", pattern, got, want)
		}
	}
}

func TestContainsFold(t *testing.T) {
	for _, tt := range []struct {
		s, lit string
		want   bool
	}{
		{"an ERROR here", "error", true},
		{"an Error", "error", true},
		{"errr eRRor", "error", true},
		{"erro", "error", false},
		{"", "error", false},
		{"x1-Y", "1-y", true},
	} {
		if got := containsFold([]byte(tt.s), []byte(tt.lit)); got != tt.want {
			t.Errorf("containsFold(%q, %q) = %v", tt.s, tt.lit, got)
		}
	}
}
