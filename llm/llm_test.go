package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var lookupTool = ToolSpec{
	Name:        "lookup",
	Description: "Look something up",
	Schema: map[string]any{
		"type":       "object",
		"properties": map[string]any{"q": map[string]any{"type": "string"}},
		"required":   []string{"q"},
	},
}

// recorder serves canned responses in order and keeps the request bodies.
type recorder struct {
	responses []string
	bodies    []map[string]any
	headers   []http.Header
}

func (rc *recorder) server(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(b, &body)
		rc.bodies = append(rc.bodies, body)
		rc.headers = append(rc.headers, r.Header.Clone())
		i := len(rc.bodies) - 1
		w.Header().Set("Content-Type", "application/json")
		if i >= len(rc.responses) {
			w.WriteHeader(500)
			return
		}
		_, _ = w.Write([]byte(rc.responses[i]))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestAnthropicToolRoundTrip(t *testing.T) {
	rc := &recorder{responses: []string{
		`{"id":"m1","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"tool_use",
		  "content":[{"type":"thinking","thinking":"","signature":"sig1"},{"type":"text","text":"checking"},
		             {"type":"tool_use","id":"tu1","name":"lookup","input":{"q":"x"}}],
		  "usage":{"input_tokens":10,"output_tokens":5}}`,
		`{"id":"m2","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"end_turn",
		  "content":[{"type":"text","text":"done"}],"usage":{"input_tokens":20,"output_tokens":3}}`,
		`{"id":"m3","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"refusal",
		  "content":[],"usage":{"input_tokens":1,"output_tokens":0}}`,
	}}
	srv := rc.server(t)
	p := NewAnthropic(AnthropicConfig{APIKey: "k", BaseURL: srv.URL, Fallbacks: true})
	conv := p.NewConversation(Options{Effort: "high"}, "system prompt", []ToolSpec{lookupTool})

	turn, err := conv.Send(context.Background(), "hello", nil)
	if err != nil {
		t.Fatal(err)
	}
	if turn.Text != "checking" || len(turn.ToolCalls) != 1 || turn.ToolCalls[0].ID != "tu1" || string(turn.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Fatalf("turn = %+v", turn)
	}
	first := rc.bodies[0]
	if first["model"] != DefaultAnthropicModel || first["fallbacks"] != "default" {
		t.Fatalf("model/fallbacks: %v %v", first["model"], first["fallbacks"])
	}
	if oc, _ := first["output_config"].(map[string]any); oc["effort"] != "high" {
		t.Fatalf("output_config = %v", first["output_config"])
	}
	if !strings.Contains(rc.headers[0].Get("anthropic-beta"), "server-side-fallback-2026-07-01") {
		t.Fatalf("beta header = %q", rc.headers[0].Get("anthropic-beta"))
	}
	sys := first["system"].([]any)[0].(map[string]any)
	if sys["cache_control"] == nil {
		t.Fatalf("system prompt not cached: %v", sys)
	}

	turn, err = conv.Send(context.Background(), "", []ToolResult{{CallID: "tu1", Content: "result"}})
	if err != nil {
		t.Fatal(err)
	}
	if turn.Text != "done" || len(turn.ToolCalls) != 0 {
		t.Fatalf("second turn = %+v", turn)
	}
	msgs := rc.bodies[1]["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("history len %d", len(msgs))
	}
	// The assistant turn, including its thinking block, goes back unchanged.
	asst := msgs[1].(map[string]any)["content"].([]any)
	if asst[0].(map[string]any)["type"] != "thinking" || asst[0].(map[string]any)["signature"] != "sig1" {
		t.Fatalf("assistant history = %v", asst)
	}
	res := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if res["type"] != "tool_result" || res["tool_use_id"] != "tu1" {
		t.Fatalf("tool result = %v", res)
	}

	if _, err := conv.Send(context.Background(), "again", nil); !errors.Is(err, ErrRefused) {
		t.Fatalf("refusal err = %v", err)
	}
}

func TestOpenAIToolRoundTrip(t *testing.T) {
	rc := &recorder{responses: []string{
		`{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
		  "tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]}}],
		  "usage":{"prompt_tokens":7,"completion_tokens":2}}`,
		`{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}],"usage":{}}`,
		`{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null,"refusal":"no"}}]}`,
	}}
	srv := rc.server(t)
	p := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL + "/v1/"})
	conv := p.NewConversation(Options{Model: "m", Effort: "high", MaxTokens: 100}, "sys", []ToolSpec{lookupTool})

	turn, err := conv.Send(context.Background(), "hello", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(turn.ToolCalls) != 1 || turn.ToolCalls[0].Name != "lookup" || string(turn.ToolCalls[0].Input) != `{"q":"x"}` || turn.Usage.InputTokens != 7 {
		t.Fatalf("turn = %+v", turn)
	}
	if rc.headers[0].Get("Authorization") != "Bearer k" || rc.bodies[0]["reasoning_effort"] != nil {
		t.Fatalf("auth/effort: %q %v", rc.headers[0].Get("Authorization"), rc.bodies[0]["reasoning_effort"])
	}
	turn, err = conv.Send(context.Background(), "", []ToolResult{{CallID: "c1", Content: "boom", IsError: true}})
	if err != nil || turn.Text != "done" {
		t.Fatalf("second turn = %+v, %v", turn, err)
	}
	msgs := rc.bodies[1]["messages"].([]any)
	if len(msgs) != 4 || msgs[3].(map[string]any)["role"] != "tool" || msgs[3].(map[string]any)["content"] != "ERROR: boom" {
		t.Fatalf("history = %v", msgs)
	}
	if _, err := conv.Send(context.Background(), "again", nil); !errors.Is(err, ErrRefused) {
		t.Fatalf("refusal err = %v", err)
	}
	// Past the canned responses the server returns 500.
	if _, err := conv.Send(context.Background(), "more", nil); err == nil || !strings.Contains(err.Error(), "HTTP 500") {
		t.Fatalf("http error = %v", err)
	}
}

func TestOpenAIFindsV1(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		switch r.URL.Path {
		case "/v1/models":
			_, _ = io.WriteString(w, `{"data":[{"id":"m"}]}`)
		case "/v1/chat/completions":
			_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	p := NewOpenAI(OpenAIConfig{BaseURL: srv.URL})
	ids, err := p.(ModelLister).ListModels(context.Background())
	if err != nil || len(ids) != 1 {
		t.Fatalf("models = %v, %v", ids, err)
	}
	turn, err := p.NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hello", nil)
	if err != nil || turn.Text != "hi" {
		t.Fatalf("turn = %+v, %v", turn, err)
	}
	// Once found, /v1 is used directly.
	if got := strings.Join(paths, " "); got != "/models /v1/models /v1/chat/completions" {
		t.Fatalf("paths = %s", got)
	}

	// A real 404 is still reported.
	p = NewOpenAI(OpenAIConfig{BaseURL: srv.URL + "/nowhere"})
	if _, err := p.NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hello", nil); err == nil || !strings.Contains(err.Error(), "HTTP 404") {
		t.Fatalf("err = %v", err)
	}
}
