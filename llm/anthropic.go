package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// DefaultAnthropicModel is used when no model is configured.
const DefaultAnthropicModel = "claude-opus-5-5"

// AnthropicConfig configures the Anthropic provider.
type AnthropicConfig struct {
	APIKey  string
	BaseURL string
	// Fallbacks asks the API to re-serve a declined request on a fallback
	// model inside the same call. Only the first-party API supports it.
	Fallbacks bool
}

type anthropicProvider struct {
	client    anthropic.Client
	fallbacks bool
}

// NewAnthropic returns a provider backed by the official Anthropic SDK.
func NewAnthropic(cfg AnthropicConfig) Provider {
	var opts []option.RequestOption
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &anthropicProvider{client: anthropic.NewClient(opts...), fallbacks: cfg.Fallbacks}
}

func (p *anthropicProvider) Name() string { return "anthropic" }

func (p *anthropicProvider) NewConversation(opts Options, system string, tools []ToolSpec) Conversation {
	if opts.Model == "" {
		opts.Model = DefaultAnthropicModel
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 16000
	}
	params := anthropic.BetaMessageNewParams{
		Model:     anthropic.Model(opts.Model),
		MaxTokens: opts.MaxTokens,
		// Caching the stable system prompt and tool list keeps each agent
		// turn from re-billing them in full.
		System: []anthropic.BetaTextBlockParam{{
			Text:         system,
			CacheControl: anthropic.NewBetaCacheControlEphemeralParam(),
		}},
	}
	if opts.Effort != "" {
		params.OutputConfig = anthropic.BetaOutputConfigParam{Effort: anthropic.BetaOutputConfigEffort(opts.Effort)}
	}
	for _, t := range tools {
		props := t.Schema["properties"]
		var required []string
		if r, ok := t.Schema["required"].([]string); ok {
			required = r
		}
		tp := anthropic.BetaToolParam{
			Name:        t.Name,
			Description: anthropic.String(t.Description),
			InputSchema: anthropic.BetaToolInputSchemaParam{Properties: props, Required: required},
		}
		params.Tools = append(params.Tools, anthropic.BetaToolUnionParam{OfTool: &tp})
	}
	if p.fallbacks {
		params.Betas = append(params.Betas, anthropic.AnthropicBetaServerSideFallback2026_07_01)
		params.Fallbacks = anthropic.BetaFallbacksParamOfDefault()
	}
	return &anthropicConversation{client: &p.client, params: params}
}

type anthropicConversation struct {
	client *anthropic.Client
	params anthropic.BetaMessageNewParams
}

func (c *anthropicConversation) Send(ctx context.Context, text string, results []ToolResult) (*Turn, error) {
	var blocks []anthropic.BetaContentBlockParamUnion
	for _, r := range results {
		blocks = append(blocks, anthropic.NewBetaToolResultBlock(r.CallID, r.Content, r.IsError))
	}
	if strings.TrimSpace(text) != "" {
		blocks = append(blocks, anthropic.NewBetaTextBlock(text))
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("llm: empty user turn")
	}
	c.params.Messages = append(c.params.Messages, anthropic.NewBetaUserMessage(blocks...))

	resp, err := c.client.Beta.Messages.New(ctx, c.params)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	// Append the reply unchanged: reasoning blocks must go back as-is.
	c.params.Messages = append(c.params.Messages, resp.ToParam())

	turn := &Turn{Usage: Usage{InputTokens: resp.Usage.InputTokens, OutputTokens: resp.Usage.OutputTokens}}
	switch resp.StopReason {
	case anthropic.BetaStopReasonRefusal:
		return turn, ErrRefused
	case anthropic.BetaStopReasonMaxTokens:
		turn.Truncated = true
	}
	var texts []string
	for _, block := range resp.Content {
		switch b := block.AsAny().(type) {
		case anthropic.BetaTextBlock:
			texts = append(texts, b.Text)
		case anthropic.BetaToolUseBlock:
			turn.ToolCalls = append(turn.ToolCalls, ToolCall{ID: b.ID, Name: b.Name, Input: json.RawMessage(b.JSON.Input.Raw())})
		}
	}
	turn.Text = strings.Join(texts, "\n")
	return turn, nil
}
