package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
		`{"status":"completed","output":[
		   {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"ENC"},
		   {"type":"function_call","id":"fc_1","call_id":"c1","name":"lookup","arguments":"{\"q\":\"x\"}","status":"completed"}],
		  "usage":{"input_tokens":7,"output_tokens":2}}`,
		`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}],"usage":{}}`,
		`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"no"}]}]}`,
		`{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"par"}]}]}`,
		`{"status":"failed","error":{"message":"server overloaded"},"output":[]}`,
	}}
	srv := rc.server(t)
	p := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL + "/v1/"})
	conv := p.NewConversation(Options{Model: "m", Effort: "high", MaxTokens: 100}, "sys", []ToolSpec{lookupTool})

	turn, err := conv.Send(context.Background(), "hello", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(turn.ToolCalls) != 1 || turn.ToolCalls[0].ID != "c1" || turn.ToolCalls[0].Name != "lookup" || string(turn.ToolCalls[0].Input) != `{"q":"x"}` || turn.Usage.InputTokens != 7 {
		t.Fatalf("turn = %+v", turn)
	}
	req := rc.bodies[0]
	tool := req["tools"].([]any)[0].(map[string]any)
	if rc.headers[0].Get("Authorization") != "Bearer k" || req["instructions"] != "sys" || req["store"] != false ||
		req["reasoning"].(map[string]any)["effort"] != "high" || req["max_output_tokens"] != float64(100) ||
		fmt.Sprint(req["include"]) != "[reasoning.encrypted_content]" || tool["name"] != "lookup" || tool["strict"] != false {
		t.Fatalf("first request = %v", req)
	}
	turn, err = conv.Send(context.Background(), "", []ToolResult{{CallID: "c1", Content: "boom", IsError: true}})
	if err != nil || turn.Text != "done" {
		t.Fatalf("second turn = %+v, %v", turn, err)
	}
	// The reasoning and call go back unchanged, followed by the result.
	items := rc.bodies[1]["input"].([]any)
	if len(items) != 4 || items[0].(map[string]any)["content"] != "hello" || items[1].(map[string]any)["encrypted_content"] != "ENC" ||
		items[2].(map[string]any)["call_id"] != "c1" || items[3].(map[string]any)["type"] != "function_call_output" || items[3].(map[string]any)["output"] != "ERROR: boom" {
		t.Fatalf("history = %v", items)
	}
	if _, err := conv.Send(context.Background(), "again", nil); !errors.Is(err, ErrRefused) {
		t.Fatalf("refusal err = %v", err)
	}
	if turn, err := conv.Send(context.Background(), "long", nil); err != nil || !turn.Truncated || turn.Text != "par" {
		t.Fatalf("truncated turn = %+v, %v", turn, err)
	}
	n := len(conv.(*openAIConversation).items)
	if _, err := conv.Send(context.Background(), "fail", nil); err == nil || !strings.Contains(err.Error(), "server overloaded") {
		t.Fatalf("failed err = %v", err)
	}
	if len(conv.(*openAIConversation).items) != n {
		t.Fatal("a failed request stayed in the history")
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
		case "/v1/responses":
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req["model"] != "m" {
				w.WriteHeader(http.StatusNotFound)
				_, _ = io.WriteString(w, `{"error":{"message":"The model does not exist","code":"model_not_found"}}`)
				return
			}
			_, _ = io.WriteString(w, `{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}`)
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
	if got := strings.Join(paths, " "); got != "/models /v1/models /v1/responses" {
		t.Fatalf("paths = %s", got)
	}

	// An unknown model is a 404 too, but an API error, reported as is.
	paths = nil
	p = NewOpenAI(OpenAIConfig{BaseURL: srv.URL + "/v1"})
	if _, err := p.NewConversation(Options{Model: "nope"}, "", nil).Send(context.Background(), "hello", nil); err == nil || !strings.Contains(err.Error(), "HTTP 404: The model does not exist") || len(paths) != 1 {
		t.Fatalf("err = %v, paths = %v", err, paths)
	}

	// A real 404 is still reported.
	p = NewOpenAI(OpenAIConfig{BaseURL: srv.URL + "/nowhere"})
	if _, err := p.NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hello", nil); err == nil || !strings.Contains(err.Error(), "HTTP 404") {
		t.Fatalf("err = %v", err)
	}
}

func TestNoEnvironment(t *testing.T) {
	// The library uses only what it is given, whatever the environment says.
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	t.Setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
	rc := &recorder{responses: []string{
		`{"id":"m","type":"message","role":"assistant","model":"x","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`,
		`{"status":"completed","output":[]}`,
	}}
	srv := rc.server(t)
	if _, err := NewAnthropic(AnthropicConfig{BaseURL: srv.URL}).NewConversation(Options{}, "", nil).Send(context.Background(), "hi", nil); err != nil {
		t.Fatal(err)
	}
	if k := rc.headers[0].Get("X-Api-Key"); k != "" {
		t.Fatalf("anthropic sent key %q from the environment", k)
	}
	if rc.bodies[0]["max_tokens"] != float64(DefaultMaxTokens) || rc.bodies[0]["model"] != DefaultAnthropicModel {
		t.Fatalf("anthropic defaults: %v", rc.bodies[0])
	}
	t.Setenv("OPENAI_API_KEY", "env-key")
	if _, err := NewOpenAI(OpenAIConfig{BaseURL: srv.URL + "/v1"}).NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hi", nil); err != nil {
		t.Fatal(err)
	}
	if k := rc.headers[1].Get("Authorization"); k != "" || rc.bodies[1]["max_output_tokens"] != float64(DefaultMaxTokens) {
		t.Fatalf("openai auth %q, body %v", k, rc.bodies[1])
	}
}
