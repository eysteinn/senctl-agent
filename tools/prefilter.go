package tools

import (
	"bytes"
	"regexp/syntax"
	"unicode/utf8"
)

// literal is a string a match must contain, possibly ignoring ASCII case.
type literal struct {
	b    []byte
	fold bool
}

// prefilter returns literals of which every match of pattern contains at
// least one, or nil when there are none worth checking. Testing a line with
// bytes.Contains first is far cheaper than running the regexp on it.
func prefilter(pattern string) []literal {
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil
	}
	lits := required(re.Simplify())
	for _, l := range lits {
		if len(l.b) < 2 {
			return nil
		}
	}
	return lits
}

// required computes prefilter literals for one syntax node.
func required(re *syntax.Regexp) []literal {
	switch re.Op {
	case syntax.OpLiteral:
		s := string(re.Rune)
		fold := re.Flags&syntax.FoldCase != 0
		if fold && !isASCII(s) {
			// Unicode case folding has more than two forms per letter.
			return nil
		}
		return []literal{{[]byte(s), fold}}
	case syntax.OpCapture, syntax.OpPlus:
		return required(re.Sub[0])
	case syntax.OpRepeat:
		if re.Min >= 1 {
			return required(re.Sub[0])
		}
	case syntax.OpConcat:
		// Any child's literals are required; keep the most selective.
		var best []literal
		for _, sub := range re.Sub {
			if l := required(sub); l != nil && (best == nil || shortest(l) > shortest(best)) {
				best = l
			}
		}
		return best
	case syntax.OpAlternate:
		var all []literal
		for _, sub := range re.Sub {
			l := required(sub)
			if l == nil {
				return nil
			}
			all = append(all, l...)
		}
		return all
	}
	return nil
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

func shortest(lits []literal) int {
	n := -1
	for _, l := range lits {
		if n < 0 || len(l.b) < n {
			n = len(l.b)
		}
	}
	return n
}

// mayMatch reports whether line contains one of lits (always true when
// there are none).
func mayMatch(lits []literal, line []byte) bool {
	if lits == nil {
		return true
	}
	for _, l := range lits {
		if l.fold && containsFold(line, l.b) || !l.fold && bytes.Contains(line, l.b) {
			return true
		}
	}
	return false
}

// containsFold is bytes.Contains ignoring ASCII case, for an ASCII lit.
func containsFold(s, lit []byte) bool {
	lo, up := lower(lit[0]), upper(lit[0])
	for len(s) >= len(lit) {
		i := bytes.IndexByte(s, lo)
		if lo != up {
			if j := bytes.IndexByte(s, up); j >= 0 && (i < 0 || j < i) {
				i = j
			}
		}
		if i < 0 || len(s)-i < len(lit) {
			return false
		}
		if bytes.EqualFold(s[i:i+len(lit)], lit) {
			return true
		}
		s = s[i+1:]
	}
	return false
}

func lower(c byte) byte {
	if 'A' <= c && c <= 'Z' {
		return c + 'a' - 'A'
	}
	return c
}

func upper(c byte) byte {
	if 'a' <= c && c <= 'z' {
		return c - ('a' - 'A')
	}
	return c
}
