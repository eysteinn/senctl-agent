package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
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
	// Only cfg counts: the SDK would otherwise pick up ANTHROPIC_API_KEY,
	// ANTHROPIC_BASE_URL and other variables from the environment.
	opts := []option.RequestOption{option.WithoutEnvironmentDefaults()}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &anthropicProvider{client: anthropic.NewClient(opts...), fallbacks: cfg.Fallbacks}
}

func (p *anthropicProvider) Name() string { return "anthropic" }

// ListModels returns the model ids the API key can use.
func (p *anthropicProvider) ListModels(ctx context.Context) ([]string, error) {
	var ids []string
	pager := p.client.Models.ListAutoPaging(ctx, anthropic.ModelListParams{})
	for pager.Next() {
		ids = append(ids, pager.Current().ID)
	}
	if err := pager.Err(); err != nil {
		return nil, fmt.Errorf("anthropic: list models: %w", err)
	}
	sort.Strings(ids)
	return ids, nil
}

func (p *anthropicProvider) NewConversation(opts Options, system string, tools []ToolSpec) Conversation {
	c := &anthropicConversation{client: &p.client, fallbacks: p.fallbacks, system: system}
	c.SetOptions(opts)
	c.SetTools(tools)
	return c
}

type anthropicConversation struct {
	client    *anthropic.Client
	fallbacks bool
	system    string
	opts      Options
	tools     []ToolSpec
	messages  []anthropic.BetaMessageParam
}

func (c *anthropicConversation) SetOptions(opts Options) {
	if opts.Model == "" {
		opts.Model = DefaultAnthropicModel
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = DefaultMaxTokens
	}
	c.opts = opts
}

func (c *anthropicConversation) SetTools(tools []ToolSpec) { c.tools = tools }

// params builds a request from the current options, tools and history.
func (c *anthropicConversation) params(stream bool) anthropic.BetaMessageNewParams {
	params := anthropic.BetaMessageNewParams{
		Model:     anthropic.Model(c.opts.Model),
		MaxTokens: c.opts.MaxTokens,
		// Caching the stable system prompt and tool list keeps each agent
		// turn from re-billing them in full.
		System: []anthropic.BetaTextBlockParam{{
			Text:         c.system,
			CacheControl: anthropic.NewBetaCacheControlEphemeralParam(),
		}},
		Messages: c.messages,
	}
	if c.opts.Effort != "" {
		params.OutputConfig = anthropic.BetaOutputConfigParam{Effort: anthropic.BetaOutputConfigEffort(c.opts.Effort)}
	}
	for _, t := range c.tools {
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
		if stream {
			// Stream tool inputs as they are generated instead of in one
			// burst; the agent validates each input before running it.
			tp.EagerInputStreaming = anthropic.Bool(true)
		}
		params.Tools = append(params.Tools, anthropic.BetaToolUnionParam{OfTool: &tp})
	}
	if c.fallbacks {
		params.Betas = append(params.Betas, anthropic.AnthropicBetaServerSideFallback2026_07_01)
		params.Fallbacks = anthropic.BetaFallbacksParamOfDefault()
	}
	return params
}

func (c *anthropicConversation) addUser(text string, results []ToolResult) error {
	var blocks []anthropic.BetaContentBlockParamUnion
	for _, r := range results {
		blocks = append(blocks, anthropic.NewBetaToolResultBlock(r.CallID, r.Content, r.IsError))
	}
	if strings.TrimSpace(text) != "" {
		blocks = append(blocks, anthropic.NewBetaTextBlock(text))
	}
	if len(blocks) == 0 {
		return fmt.Errorf("llm: empty user turn")
	}
	c.messages = append(c.messages, anthropic.NewBetaUserMessage(blocks...))
	return nil
}

func (c *anthropicConversation) Send(ctx context.Context, text string, results []ToolResult) (*Turn, error) {
	if err := c.addUser(text, results); err != nil {
		return nil, err
	}
	resp, err := c.client.Beta.Messages.New(ctx, c.params(false))
	if err != nil {
		c.messages = c.messages[:len(c.messages)-1]
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	return c.finish(resp)
}

// SendStream is Send with the model's text delivered to onText as it is
// generated.
func (c *anthropicConversation) SendStream(ctx context.Context, text string, results []ToolResult, onText func(string)) (*Turn, error) {
	if err := c.addUser(text, results); err != nil {
		return nil, err
	}
	stream := c.client.Beta.Messages.NewStreaming(ctx, c.params(true))
	msg := anthropic.BetaMessage{}
	for stream.Next() {
		ev := stream.Current()
		if err := msg.Accumulate(ev); err != nil {
			return nil, fmt.Errorf("anthropic: stream: %w", err)
		}
		if d, ok := ev.AsAny().(anthropic.BetaRawContentBlockDeltaEvent); ok && onText != nil {
			if td, ok := d.Delta.AsAny().(anthropic.BetaTextDelta); ok {
				onText(td.Text)
			}
		}
	}
	if err := stream.Err(); err != nil {
		c.messages = c.messages[:len(c.messages)-1]
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	return c.finish(&msg)
}

func (c *anthropicConversation) finish(resp *anthropic.BetaMessage) (*Turn, error) {
	// Append the reply unchanged: reasoning blocks must go back as-is.
	c.messages = append(c.messages, resp.ToParam())

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
