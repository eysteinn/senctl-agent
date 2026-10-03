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
	"sync"
	"time"
)

// OpenAIConfig configures a provider for any endpoint that speaks the
// OpenAI Responses API (/v1/responses): OpenAI itself and LLM proxies and
// gateways such as LiteLLM.
type OpenAIConfig struct {
	APIKey string
	// BaseURL is the API root, e.g. https://api.openai.com/v1. A root
	// without /v1 works too: when it answers 404, /v1 is tried and kept.
	BaseURL    string
	HTTPClient *http.Client
}

type openAIProvider struct {
	cfg OpenAIConfig
	mu  sync.Mutex
	// base is the API root in use; it gains /v1 when the configured root
	// turns out to need it.
	base string
}

// NewOpenAI returns a provider for a Responses API endpoint.
func NewOpenAI(cfg OpenAIConfig) Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 10 * time.Minute}
	}
	return &openAIProvider{cfg: cfg, base: cfg.BaseURL}
}

func (p *openAIProvider) Name() string { return "openai" }

// do sends a request to path under the API root. Proxies differ on whether
// the root includes /v1, so a 404 for the path itself (not an API error
// such as an unknown model) from a root without /v1 is retried with /v1,
// which is then kept for later requests.
func (p *openAIProvider) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	p.mu.Lock()
	base := p.base
	p.mu.Unlock()
	send := func(base string) (*http.Response, error) {
		var rd io.Reader
		if body != nil {
			rd = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, method, base+path, rd)
		if err != nil {
			return nil, err
		}
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		if p.cfg.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
		}
		return p.cfg.HTTPClient.Do(req)
	}
	res, err := send(base)
	if err != nil {
		return nil, markUnreachable(err)
	}
	if res.StatusCode != http.StatusNotFound || strings.HasSuffix(base, "/v1") {
		return res, nil
	}
	raw, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	res.Body.Close()
	res.Body = io.NopCloser(bytes.NewReader(raw))
	if apiError(raw) != "" {
		return res, nil
	}
	retry, err := send(base + "/v1")
	if err != nil || retry.StatusCode == http.StatusNotFound {
		if retry != nil {
			retry.Body.Close()
		}
		return res, nil
	}
	p.mu.Lock()
	p.base = base + "/v1"
	p.mu.Unlock()
	return retry, nil
}

// apiError returns the message of an API error body, or "".
func apiError(raw []byte) string {
	var e struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Error != nil {
		return e.Error.Message
	}
	return ""
}

// ListModels returns the ids from the endpoint's /models listing.
func (p *openAIProvider) ListModels(ctx context.Context) ([]string, error) {
	res, err := p.do(ctx, http.MethodGet, "/models", nil)
	if err != nil {
		return nil, fmt.Errorf("openai: list models: %w", err)
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(res.Body, 8<<20))
	if res.StatusCode >= 300 {
		return nil, fmt.Errorf("openai: list models: %w", newAPIError(res.StatusCode, raw, p.cfg.APIKey == ""))
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

type oaTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	// Strict must be off: strict schemas require every property, and tool
	// inputs here have optional ones.
	Strict bool `json:"strict"`
}

func (p *openAIProvider) NewConversation(opts Options, system string, tools []ToolSpec) Conversation {
	c := &openAIConversation{p: p, opts: opts, system: system}
	c.SetTools(tools)
	return c
}

// openAIConversation keeps the conversation on the client (store: false):
// every request carries the full history as input items, and the model's
// output items, including encrypted reasoning, go back unchanged so its
// reasoning carries across tool calls.
type openAIConversation struct {
	p      *openAIProvider
	opts   Options
	system string
	items  []json.RawMessage
	tools  []oaTool
}

func (c *openAIConversation) SetOptions(opts Options) { c.opts = opts }

func (c *openAIConversation) SetTools(tools []ToolSpec) {
	c.tools = nil
	for _, t := range tools {
		schema := t.Schema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		c.tools = append(c.tools, oaTool{Type: "function", Name: t.Name, Description: t.Description, Parameters: schema})
	}
}

// addUser appends tool results and text as input items, returning how many
// were added so a failed request can take them back.
func (c *openAIConversation) addUser(text string, results []ToolResult) int {
	n := len(c.items)
	for _, r := range results {
		out := r.Content
		if r.IsError {
			out = "ERROR: " + out
		}
		m, _ := json.Marshal(map[string]string{"type": "function_call_output", "call_id": r.CallID, "output": out})
		c.items = append(c.items, m)
	}
	if strings.TrimSpace(text) != "" {
		m, _ := json.Marshal(map[string]string{"role": "user", "content": text})
		c.items = append(c.items, m)
	}
	return len(c.items) - n
}

func (c *openAIConversation) post(ctx context.Context, stream bool) (*http.Response, error) {
	body := map[string]any{
		"model":   c.opts.Model,
		"input":   c.items,
		"store":   false,
		"include": []string{"reasoning.encrypted_content"},
	}
	if c.system != "" {
		body["instructions"] = c.system
	}
	if len(c.tools) > 0 {
		body["tools"] = c.tools
		body["tool_choice"] = "auto"
	}
	body["max_output_tokens"] = c.opts.MaxTokens
	if c.opts.MaxTokens <= 0 {
		body["max_output_tokens"] = DefaultMaxTokens
	}
	if c.opts.Effort != "" {
		body["reasoning"] = map[string]string{"effort": c.opts.Effort}
	}
	if stream {
		body["stream"] = true
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	res, err := c.p.do(ctx, http.MethodPost, "/responses", buf)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	if res.StatusCode >= 300 {
		defer res.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		return nil, fmt.Errorf("openai: %w", newAPIError(res.StatusCode, raw, c.p.cfg.APIKey == ""))
	}
	return res, nil
}

func (c *openAIConversation) Send(ctx context.Context, text string, results []ToolResult) (*Turn, error) {
	return c.SendStream(ctx, text, results, nil)
}

// SendStream is Send over server-sent events, delivering text to onText as
// it arrives. With onText nil the response comes in one piece.
func (c *openAIConversation) SendStream(ctx context.Context, text string, results []ToolResult, onText func(string)) (*Turn, error) {
	added := c.addUser(text, results)
	turn, err := c.exchange(ctx, onText)
	if err != nil && turn == nil {
		c.items = c.items[:len(c.items)-added]
	}
	return turn, err
}

func (c *openAIConversation) exchange(ctx context.Context, onText func(string)) (*Turn, error) {
	res, err := c.post(ctx, onText != nil)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if !strings.Contains(res.Header.Get("Content-Type"), "text/event-stream") {
		// Also taken when a server ignores stream=true.
		raw, err := io.ReadAll(io.LimitReader(res.Body, 64<<20))
		if err != nil {
			return nil, fmt.Errorf("openai: read response: %w", err)
		}
		var r oaResponse
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, fmt.Errorf("openai: decode response: %s", snippet(raw))
		}
		turn, err := c.finish(&r)
		if err == nil && onText != nil && turn.Text != "" {
			onText(turn.Text)
		}
		return turn, err
	}
	sc := bufio.NewScanner(res.Body)
	// The final event carries the whole response, encrypted reasoning
	// included.
	sc.Buffer(make([]byte, 64<<10), 64<<20)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		var ev struct {
			Type     string      `json:"type"`
			Delta    string      `json:"delta"`
			Message  string      `json:"message"`
			Response *oaResponse `json:"response"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &ev); err != nil {
			return nil, fmt.Errorf("openai: decode stream event: %s", snippet([]byte(data)))
		}
		switch ev.Type {
		case "response.output_text.delta":
			if onText != nil && ev.Delta != "" {
				onText(ev.Delta)
			}
		case "response.completed", "response.incomplete", "response.failed":
			if ev.Response == nil {
				return nil, fmt.Errorf("openai: %s event without a response", ev.Type)
			}
			return c.finish(ev.Response)
		case "error":
			return nil, fmt.Errorf("openai: %s", ev.Message)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("openai: read stream: %w", err)
	}
	return nil, fmt.Errorf("openai: the stream ended before the response was complete")
}

type oaResponse struct {
	Status            string `json:"status"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
	Output []json.RawMessage `json:"output"`
	Usage  struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

// finish records the output items and turns them into a Turn.
func (c *openAIConversation) finish(r *oaResponse) (*Turn, error) {
	if r.Status == "failed" || r.Error != nil && r.Error.Message != "" {
		msg := "the response failed"
		if r.Error != nil && r.Error.Message != "" {
			msg = r.Error.Message
		}
		return nil, fmt.Errorf("openai: %s", msg)
	}
	c.items = append(c.items, r.Output...)
	turn := &Turn{Usage: Usage{InputTokens: r.Usage.InputTokens, OutputTokens: r.Usage.OutputTokens}}
	if r.Status == "incomplete" && r.IncompleteDetails != nil {
		switch r.IncompleteDetails.Reason {
		case "content_filter":
			return turn, ErrRefused
		case "max_output_tokens":
			turn.Truncated = true
		}
	}
	var texts []string
	for _, raw := range r.Output {
		var item struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
			Content   []struct {
				Type    string `json:"type"`
				Text    string `json:"text"`
				Refusal string `json:"refusal"`
			} `json:"content"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, fmt.Errorf("openai: decode output item: %s", snippet(raw))
		}
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				switch part.Type {
				case "output_text":
					texts = append(texts, part.Text)
				case "refusal":
					return turn, ErrRefused
				}
			}
		case "function_call":
			// Invalid JSON is passed through; the caller reports it back to
			// the model as a tool error so it can retry.
			turn.ToolCalls = append(turn.ToolCalls, ToolCall{ID: item.CallID, Name: item.Name, Input: json.RawMessage(item.Arguments)})
		}
	}
	turn.Text = strings.Join(texts, "\n")
	return turn, nil
}

func snippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 300 {
		s = s[:300] + "…"
	}
	return s
}
