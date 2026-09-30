package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Provider names accepted by New.
const (
	ProviderAnthropic = "anthropic"
	ProviderOpenAI    = "openai"
)

// DefaultMaxTokens caps output tokens per model call when Options.MaxTokens
// is not set.
const DefaultMaxTokens = 16000

// Config selects and configures a provider. Only what is set here is used:
// nothing is read from the environment, so callers decide where settings
// come from (the CLI reads flags, environment and a config file).
type Config struct {
	// Provider is "openai" (the default: the OpenAI Responses API, which
	// OpenAI and LLM proxies and gateways serve) or "anthropic" (the Anthropic
	// Messages API, for talking to it directly).
	Provider string
	APIKey   string
	// BaseURL points at the endpoint. For openai it is the API root such
	// as https://llm-proxy.example.com/v1 (default https://api.openai.com/v1).
	BaseURL string
	// Fallbacks (anthropic) asks the API to re-serve a declined request on
	// a fallback model. Nil means on for the first-party API (no BaseURL).
	Fallbacks *bool
}

// New builds the provider described by cfg.
func New(cfg Config) (Provider, error) {
	switch strings.ToLower(strings.TrimSpace(cfg.Provider)) {
	case ProviderOpenAI, "":
		return NewOpenAI(OpenAIConfig{APIKey: cfg.APIKey, BaseURL: cfg.BaseURL}), nil
	case ProviderAnthropic:
		fallbacks := cfg.BaseURL == ""
		if cfg.Fallbacks != nil {
			fallbacks = *cfg.Fallbacks
		}
		return NewAnthropic(AnthropicConfig{APIKey: cfg.APIKey, BaseURL: cfg.BaseURL, Fallbacks: fallbacks}), nil
	}
	return nil, fmt.Errorf("llm: unknown provider %q (openai or anthropic)", cfg.Provider)
}

// DefaultModel is the model used when none is configured, or "" when the
// provider has no sensible default (a proxy decides
// which models exist).
func DefaultModel(provider string) string {
	if strings.EqualFold(provider, ProviderAnthropic) {
		return DefaultAnthropicModel
	}
	return ""
}

// ResolveModel returns model when it is set. Otherwise it returns the
// provider's default model, or the endpoint's only model when it lists
// exactly one; with several, the error names them, since the choice is the
// caller's.
func ResolveModel(ctx context.Context, p Provider, model string) (string, error) {
	if model != "" {
		return model, nil
	}
	if m := DefaultModel(p.Name()); m != "" {
		return m, nil
	}
	ml, ok := p.(ModelLister)
	if !ok {
		return "", errors.New("llm: no model given")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	ids, err := ml.ListModels(ctx)
	switch {
	case err != nil:
		return "", fmt.Errorf("llm: no model given, and listing the endpoint's models failed: %w", err)
	case len(ids) == 1:
		return ids[0], nil
	case len(ids) == 0:
		return "", errors.New("llm: no model given and the endpoint lists none")
	}
	shown, more := ids, ""
	if len(shown) > 30 {
		shown = shown[:30]
		more = fmt.Sprintf(" and %d more", len(ids)-len(shown))
	}
	return "", fmt.Errorf("llm: no model given; the endpoint offers %s%s", strings.Join(shown, ", "), more)
}
