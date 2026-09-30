package evidence

import "testing"

func TestContains(t *testing.T) {
	c := NewCorpus("Test orders::create failed\nFailure text:\nconnect ECONNREFUSED   127.0.0.1:5432\n" +
		"Lines 10-12 of 40:\n10: starting db\n11: dial tcp 127.0.0.1:5432: connection refused\n12: retrying")
	tests := []struct {
		excerpt string
		want    bool
	}{
		{"connect ECONNREFUSED 127.0.0.1:5432", true},
		{"connect  ECONNREFUSED\n127.0.0.1:5432", true},
		{"dial tcp 127.0.0.1:5432: connection refused retrying", true},
		{"starting db ... retrying", true},
		{"starting db … retrying", true},
		{"connection reset by peer", false},
		{"starting db ... invented line", false},
		{"...", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := c.Contains(tt.excerpt); got != tt.want {
			t.Errorf("Contains(%q) = %v, want %v", tt.excerpt, got, tt.want)
		}
	}
}
