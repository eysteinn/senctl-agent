package llm

import "testing"

func TestNew(t *testing.T) {
	off := false
	tests := []struct {
		cfg     Config
		want    string
		wantErr bool
	}{
		{Config{Provider: "openai", BaseURL: "http://proxy/v1"}, "openai", false},
		{Config{Provider: "Anthropic"}, "anthropic", false},
		{Config{Provider: "anthropic", Fallbacks: &off}, "anthropic", false},
		{Config{}, "", true},
		{Config{Provider: "other"}, "", true},
	}
	for _, tt := range tests {
		p, err := New(tt.cfg)
		if (err != nil) != tt.wantErr || (err == nil && p.Name() != tt.want) {
			t.Fatalf("New(%+v) = %v, %v", tt.cfg, p, err)
		}
	}
	if DefaultModel("anthropic") != DefaultAnthropicModel || DefaultModel("openai") != "" {
		t.Fatal("DefaultModel")
	}
}
