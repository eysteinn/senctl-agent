// Package evidence checks that text an agent quotes as evidence really
// occurs in what it was shown, so an analysis cannot cite invented text.
package evidence

import (
	"regexp"
	"strings"
)

var lineNumberPrefix = regexp.MustCompile(`(?m)^\d+: `)

func normalizeSpace(s string) string {
	s = strings.NewReplacer(" ", " ", "…", "...").Replace(s)
	return strings.Join(strings.Fields(s), " ")
}

// Corpus is the text an agent was shown, prepared for lookups.
type Corpus struct {
	plain, noLineNumbers string
}

// NewCorpus prepares seen text for Contains.
func NewCorpus(seen string) Corpus {
	return Corpus{
		plain:         normalizeSpace(seen),
		noLineNumbers: normalizeSpace(lineNumberPrefix.ReplaceAllString(seen, "")),
	}
}

// Contains reports whether excerpt occurs in the corpus, ignoring
// whitespace differences and "N: " line-number prefixes that tools add.
// "..." in the excerpt may skip a gap between fragments.
func (c Corpus) Contains(excerpt string) bool {
	found := 0
	for _, frag := range strings.Split(normalizeSpace(excerpt), "...") {
		frag = strings.TrimSpace(frag)
		if frag == "" {
			continue
		}
		if !strings.Contains(c.plain, frag) && !strings.Contains(c.noLineNumbers, frag) {
			return false
		}
		found++
	}
	return found > 0
}
