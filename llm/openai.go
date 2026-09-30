package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAIConfig configures a provider for any endpoint that speaks the
// OpenAI chat completions API (OpenAI, Azure OpenAI, vLLM, Ollama, LLM
// gateways).
type OpenAIConfig struct {
	APIKey string
	// BaseURL is the API root, e.g. https://api.openai.com/v1.
	BaseURL string
	// SendEffort forwards Options.Effort as reasoning_effort. Off by
	// default because many compatible servers reject unknown fields.
	SendEffort bool
	HTTPClient *http.Client
}

type openAIProvider struct {
	cfg OpenAIConfig
}

// NewOpenAI returns a provider for an OpenAI-compatible endpoint.
func NewOpenAI(cfg OpenAIConfig) Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 10 * time.Minute}
	}
	return &openAIProvider{cfg: cfg}
}

func (p *openAIProvider) Name() string { return "openai" }

type oaTool struct {
	Type     string     `json:"type"`
	Function oaFunction `json:"function"`
}

type oaFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

func (p *openAIProvider) NewConversation(opts Options, system string, tools []ToolSpec) Conversation {
	c := &openAIConversation{cfg: p.cfg, opts: opts}
	sys, _ := json.Marshal(map[string]string{"role": "system", "content": system})
	c.messages = append(c.messages, sys)
	for _, t := range tools {
		schema := t.Schema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		c.tools = append(c.tools, oaTool{Type: "function", Function: oaFunction{Name: t.Name, Description: t.Description, Parameters: schema}})
	}
	return c
}

type openAIConversation struct {
	cfg  OpenAIConfig
	opts Options
	// messages holds the history as raw JSON so assistant messages go back
	// exactly as the server sent them.
	messages []json.RawMessage
	tools    []oaTool
}

type oaToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaResponse struct {
	Choices []struct {
		Message      json.RawMessage `json:"message"`
		FinishReason string          `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (c *openAIConversation) Send(ctx context.Context, text string, results []ToolResult) (*Turn, error) {
	for _, r := range results {
		content := r.Content
		if r.IsError {
			content = "ERROR: " + content
		}
		m, _ := json.Marshal(map[string]string{"role": "tool", "tool_call_id": r.CallID, "content": content})
		c.messages = append(c.messages, m)
	}
	if strings.TrimSpace(text) != "" {
		m, _ := json.Marshal(map[string]string{"role": "user", "content": text})
		c.messages = append(c.messages, m)
	}

	body := map[string]any{"model": c.opts.Model, "messages": c.messages}
	if len(c.tools) > 0 {
		body["tools"] = c.tools
		body["tool_choice"] = "auto"
	}
	if c.opts.MaxTokens > 0 {
		body["max_completion_tokens"] = c.opts.MaxTokens
	}
	if c.cfg.SendEffort && c.opts.Effort != "" {
		body["reasoning_effort"] = c.opts.Effort
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	res, err := c.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 32<<20))
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	var out oaResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("openai: HTTP %d: %s", res.StatusCode, snippet(raw))
	}
	if res.StatusCode >= 300 || out.Error != nil {
		msg := snippet(raw)
		if out.Error != nil {
			msg = out.Error.Message
		}
		return nil, fmt.Errorf("openai: HTTP %d: %s", res.StatusCode, msg)
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("openai: response has no choices")
	}
	choice := out.Choices[0]
	c.messages = append(c.messages, choice.Message)

	var msg struct {
		Content   *string      `json:"content"`
		Refusal   *string      `json:"refusal"`
		ToolCalls []oaToolCall `json:"tool_calls"`
	}
	if err := json.Unmarshal(choice.Message, &msg); err != nil {
		return nil, fmt.Errorf("openai: decode message: %w", err)
	}
	turn := &Turn{Usage: Usage{InputTokens: out.Usage.PromptTokens, OutputTokens: out.Usage.CompletionTokens}}
	if msg.Refusal != nil && *msg.Refusal != "" || choice.FinishReason == "content_filter" {
		return turn, ErrRefused
	}
	turn.Truncated = choice.FinishReason == "length"
	if msg.Content != nil {
		turn.Text = *msg.Content
	}
	for _, tc := range msg.ToolCalls {
		// Invalid JSON is passed through; the caller reports it back to the
		// model as a tool error so it can retry.
		args := json.RawMessage(tc.Function.Arguments)
		turn.ToolCalls = append(turn.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Input: args})
	}
	return turn, nil
}

func snippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 300 {
		s = s[:300] + "…"
	}
	return s
}
