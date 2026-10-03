package llm

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestNew(t *testing.T) {
	off := false
	tests := []struct {
		cfg     Config
		want    string
		wantErr bool
	}{
		{Config{Provider: "openai", BaseURL: "http://proxy/v1"}, "openai", false},
		{Config{Provider: "Anthropic", APIKey: "k"}, "anthropic", false},
		{Config{Provider: "anthropic", APIKey: "k", Fallbacks: &off}, "anthropic", false},
		{Config{BaseURL: "http://proxy"}, "openai", false},
		{Config{APIKey: "k"}, "openai", false},
		{Config{Provider: "other"}, "", true},
	}
	for _, tt := range tests {
		p, err := New(tt.cfg)
		if (err != nil) != tt.wantErr || (err == nil && p.Name() != tt.want) {
			t.Fatalf("New(%+v) = %v, %v", tt.cfg, p, err)
		}
	}
	for _, provider := range []string{"", "openai", "anthropic"} {
		if _, err := New(Config{Provider: provider}); !errors.Is(err, ErrNoAPIKey) {
			t.Errorf("New(%q) without key or base URL: %v, want ErrNoAPIKey", provider, err)
		}
	}
	if DefaultModel("anthropic") != DefaultAnthropicModel || DefaultModel("openai") != "" {
		t.Fatal("DefaultModel")
	}
}

type listing struct {
	Provider
	ids []string
}

func (l listing) ListModels(context.Context) ([]string, error) { return l.ids, nil }

func TestResolveModel(t *testing.T) {
	ctx := context.Background()
	openai := NewOpenAI(OpenAIConfig{})
	for _, tt := range []struct {
		p           Provider
		model, want string
		err         string
	}{
		{openai, "given", "given", ""},
		{NewAnthropic(AnthropicConfig{}), "", DefaultAnthropicModel, ""},
		{listing{openai, []string{"only"}}, "", "only", ""},
		{listing{openai, []string{"a", "b"}}, "", "", "the endpoint offers a, b"},
		{listing{openai, nil}, "", "", "lists none"},
	} {
		got, err := ResolveModel(ctx, tt.p, tt.model)
		if got != tt.want || (tt.err == "") != (err == nil) || err != nil && (!strings.Contains(err.Error(), tt.err) || !errors.Is(err, ErrNoModel)) {
			t.Errorf("ResolveModel(%q) = %q, %v", tt.model, got, err)
		}
	}
}
