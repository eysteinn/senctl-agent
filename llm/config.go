package llm

import (
	"fmt"
	"strings"
)

// Provider names accepted by New.
const (
	ProviderAnthropic = "anthropic"
	ProviderOpenAI    = "openai"
)

// Config selects and configures a provider. It is what a config file or
// environment maps onto.
type Config struct {
	// Provider is "openai" (any OpenAI-compatible chat completions
	// endpoint, including LLM proxies/gateways) or "anthropic".
	Provider string
	APIKey   string
	// BaseURL points at the endpoint. For openai it is the API root such
	// as https://llm-proxy.example.com/v1 (default https://api.openai.com/v1).
	BaseURL string
	// Fallbacks (anthropic) asks the API to re-serve a declined request on
	// a fallback model. Nil means on for the first-party API (no BaseURL).
	Fallbacks *bool
	// SendEffort (openai) forwards Options.Effort as reasoning_effort.
	SendEffort bool
}

// New builds the provider described by cfg.
func New(cfg Config) (Provider, error) {
	switch strings.ToLower(strings.TrimSpace(cfg.Provider)) {
	case ProviderOpenAI:
		return NewOpenAI(OpenAIConfig{APIKey: cfg.APIKey, BaseURL: cfg.BaseURL, SendEffort: cfg.SendEffort}), nil
	case ProviderAnthropic:
		fallbacks := cfg.BaseURL == ""
		if cfg.Fallbacks != nil {
			fallbacks = *cfg.Fallbacks
		}
		return NewAnthropic(AnthropicConfig{APIKey: cfg.APIKey, BaseURL: cfg.BaseURL, Fallbacks: fallbacks}), nil
	case "":
		return nil, fmt.Errorf("llm: no provider configured (openai or anthropic)")
	}
	return nil, fmt.Errorf("llm: unknown provider %q (openai or anthropic)", cfg.Provider)
}

// DefaultModel is the model used when none is configured, or "" when the
// provider has no sensible default (an OpenAI-compatible proxy decides
// which models exist).
func DefaultModel(provider string) string {
	if strings.EqualFold(provider, ProviderAnthropic) {
		return DefaultAnthropicModel
	}
	return ""
}
