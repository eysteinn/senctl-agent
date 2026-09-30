package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
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
	c.SetTools(tools)
	return c
}

func (c *openAIConversation) SetOptions(opts Options) { c.opts = opts }

func (c *openAIConversation) SetTools(tools []ToolSpec) {
	c.tools = nil
	for _, t := range tools {
		schema := t.Schema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		c.tools = append(c.tools, oaTool{Type: "function", Function: oaFunction{Name: t.Name, Description: t.Description, Parameters: schema}})
	}
}

// ListModels returns the ids from the endpoint's /models listing.
func (p *openAIProvider) ListModels(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.cfg.BaseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	if p.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}
	res, err := p.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: list models: %w", err)
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(res.Body, 8<<20))
	if res.StatusCode >= 300 {
		return nil, fmt.Errorf("openai: list models: HTTP %d: %s", res.StatusCode, snippet(raw))
	}
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("openai: list models: %w", err)
	}
	ids := make([]string, 0, len(out.Data))
	for _, m := range out.Data {
		ids = append(ids, m.ID)
	}
	sort.Strings(ids)
	return ids, nil
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

// addUser appends tool results and the user's text; it returns how many
// messages were added so a failed request can take them back.
func (c *openAIConversation) addUser(text string, results []ToolResult) int {
	n := len(c.messages)
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
	return len(c.messages) - n
}

func (c *openAIConversation) post(ctx context.Context, stream bool) (*http.Response, error) {
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
	if stream {
		body["stream"] = true
		body["stream_options"] = map[string]any{"include_usage": true}
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
	if res.StatusCode >= 300 {
		defer res.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		msg := snippet(raw)
		var e oaResponse
		if json.Unmarshal(raw, &e) == nil && e.Error != nil {
			msg = e.Error.Message
		}
		return nil, fmt.Errorf("openai: HTTP %d: %s", res.StatusCode, msg)
	}
	return res, nil
}

func (c *openAIConversation) Send(ctx context.Context, text string, results []ToolResult) (*Turn, error) {
	added := c.addUser(text, results)
	turn, err := c.send(ctx)
	if err != nil && turn == nil {
		c.messages = c.messages[:len(c.messages)-added]
	}
	return turn, err
}

func (c *openAIConversation) send(ctx context.Context) (*Turn, error) {
	res, err := c.post(ctx, false)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	return c.decode(res)
}

// decode handles a complete (non-streamed) chat completion response.
func (c *openAIConversation) decode(res *http.Response) (*Turn, error) {
	raw, err := io.ReadAll(io.LimitReader(res.Body, 32<<20))
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	var out oaResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("openai: decode response: %s", snippet(raw))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("openai: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("openai: response has no choices")
	}
	choice := out.Choices[0]
	var msg oaMessage
	if err := json.Unmarshal(choice.Message, &msg); err != nil {
		return nil, fmt.Errorf("openai: decode message: %w", err)
	}
	return c.finish(choice.Message, msg, choice.FinishReason, Usage{InputTokens: out.Usage.PromptTokens, OutputTokens: out.Usage.CompletionTokens})
}

type oaMessage struct {
	Content   *string      `json:"content"`
	Refusal   *string      `json:"refusal"`
	ToolCalls []oaToolCall `json:"tool_calls"`
}

// finish records the assistant message and turns it into a Turn.
func (c *openAIConversation) finish(raw json.RawMessage, msg oaMessage, finishReason string, usage Usage) (*Turn, error) {
	c.messages = append(c.messages, raw)
	turn := &Turn{Usage: usage}
	if msg.Refusal != nil && *msg.Refusal != "" || finishReason == "content_filter" {
		return turn, ErrRefused
	}
	turn.Truncated = finishReason == "length"
	if msg.Content != nil {
		turn.Text = *msg.Content
	}
	for _, tc := range msg.ToolCalls {
		// Invalid JSON is passed through; the caller reports it back to the
		// model as a tool error so it can retry.
		turn.ToolCalls = append(turn.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Input: json.RawMessage(tc.Function.Arguments)})
	}
	return turn, nil
}

// SendStream is Send over server-sent events, delivering text to onText as
// it arrives and assembling tool calls from their streamed fragments.
func (c *openAIConversation) SendStream(ctx context.Context, text string, results []ToolResult, onText func(string)) (*Turn, error) {
	added := c.addUser(text, results)
	turn, err := c.sendStream(ctx, onText)
	if err != nil && turn == nil {
		c.messages = c.messages[:len(c.messages)-added]
	}
	return turn, err
}

func (c *openAIConversation) sendStream(ctx context.Context, onText func(string)) (*Turn, error) {
	res, err := c.post(ctx, true)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if !strings.Contains(res.Header.Get("Content-Type"), "text/event-stream") {
		// Some compatible servers ignore stream=true and answer in one piece.
		turn, err := c.decode(res)
		if err == nil && onText != nil && turn.Text != "" {
			onText(turn.Text)
		}
		return turn, err
	}
	var content, refusal strings.Builder
	var calls []oaToolCall
	var finish string
	var usage Usage
	sc := bufio.NewScanner(res.Body)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   *string `json:"content"`
					Refusal   *string `json:"refusal"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int64 `json:"prompt_tokens"`
				CompletionTokens int64 `json:"completion_tokens"`
			} `json:"usage"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("openai: decode stream chunk: %s", snippet([]byte(data)))
		}
		if chunk.Error != nil {
			return nil, fmt.Errorf("openai: %s", chunk.Error.Message)
		}
		if chunk.Usage != nil {
			usage = Usage{InputTokens: chunk.Usage.PromptTokens, OutputTokens: chunk.Usage.CompletionTokens}
		}
		for _, ch := range chunk.Choices {
			if d := ch.Delta.Content; d != nil && *d != "" {
				content.WriteString(*d)
				if onText != nil {
					onText(*d)
				}
			}
			if d := ch.Delta.Refusal; d != nil {
				refusal.WriteString(*d)
			}
			for _, tc := range ch.Delta.ToolCalls {
				for len(calls) <= tc.Index {
					calls = append(calls, oaToolCall{Type: "function"})
				}
				call := &calls[tc.Index]
				if tc.ID != "" {
					call.ID = tc.ID
				}
				call.Function.Name += tc.Function.Name
				call.Function.Arguments += tc.Function.Arguments
			}
			if ch.FinishReason != nil {
				finish = *ch.FinishReason
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("openai: read stream: %w", err)
	}
	msg := oaMessage{ToolCalls: calls}
	assistant := map[string]any{"role": "assistant", "content": nil}
	if content.Len() > 0 {
		s := content.String()
		msg.Content = &s
		assistant["content"] = s
	}
	if refusal.Len() > 0 {
		r := refusal.String()
		msg.Refusal = &r
	}
	if len(calls) > 0 {
		assistant["tool_calls"] = calls
	}
	raw, _ := json.Marshal(assistant)
	return c.finish(raw, msg, finish, usage)
}

func snippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 300 {
		s = s[:300] + "…"
	}
	return s
}
