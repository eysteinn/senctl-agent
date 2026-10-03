package llm

import (
	"context"
	"fmt"
	"net/url"
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
	// RequestTimeout bounds one attempt at a model call (default
	// DefaultRequestTimeout). IdleTimeout bounds silence once the reply's
	// text has started (default DefaultIdleTimeout); before that a reasoning
	// model may send nothing for a long time. MaxRetries is how often a
	// call is retried after a timeout, a network error or a 408/409/429/5xx
	// answer, if none of its text reached the caller yet (nil:
	// DefaultMaxRetries; 0 turns retries off).
	RequestTimeout time.Duration
	IdleTimeout    time.Duration
	MaxRetries     *int
}

// New builds the provider described by cfg. Without a BaseURL it talks to
// the provider's own API, which needs an APIKey: without one it returns
// ErrNoAPIKey. A proxy may work without a key, so it is not checked then.
func New(cfg Config) (Provider, error) {
	name := strings.ToLower(strings.TrimSpace(cfg.Provider))
	if name == "" {
		name = ProviderOpenAI
	}
	if name != ProviderOpenAI && name != ProviderAnthropic {
		return nil, &ConfigError{Field: "Provider", Msg: fmt.Sprintf("unknown provider %q; use %s or %s", cfg.Provider, ProviderOpenAI, ProviderAnthropic)}
	}
	if cfg.BaseURL != "" {
		if u, err := url.Parse(cfg.BaseURL); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return nil, &ConfigError{Field: "BaseURL", Msg: fmt.Sprintf("base URL %q must be an http:// or https:// address", cfg.BaseURL)}
		}
	}
	if cfg.APIKey == "" && cfg.BaseURL == "" {
		return nil, fmt.Errorf("%w: the %s API needs one", ErrNoAPIKey, name)
	}
	if name == ProviderAnthropic {
		fallbacks := cfg.BaseURL == ""
		if cfg.Fallbacks != nil {
			fallbacks = *cfg.Fallbacks
		}
		return NewAnthropic(AnthropicConfig{APIKey: cfg.APIKey, BaseURL: cfg.BaseURL, Fallbacks: fallbacks,
			RequestTimeout: cfg.RequestTimeout, IdleTimeout: cfg.IdleTimeout, MaxRetries: cfg.MaxRetries}), nil
	}
	return NewOpenAI(OpenAIConfig{APIKey: cfg.APIKey, BaseURL: cfg.BaseURL,
		RequestTimeout: cfg.RequestTimeout, IdleTimeout: cfg.IdleTimeout, MaxRetries: cfg.MaxRetries}), nil
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
		return "", ErrNoModel
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	ids, err := ml.ListModels(ctx)
	switch {
	case err != nil:
		return "", fmt.Errorf("%w, and listing the endpoint's models failed: %w", ErrNoModel, err)
	case len(ids) == 1:
		return ids[0], nil
	case len(ids) == 0:
		return "", fmt.Errorf("%w and the endpoint lists none", ErrNoModel)
	}
	shown, more := ids, ""
	if len(shown) > 30 {
		shown = shown[:30]
		more = fmt.Sprintf(" and %d more", len(ids)-len(shown))
	}
	return "", fmt.Errorf("%w; the endpoint offers %s%s", ErrNoModel, strings.Join(shown, ", "), more)
}
