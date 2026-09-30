package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// sse serves canned server-sent-event bodies in order and records requests.
type sse struct {
	bodies   []string
	requests []map[string]any
	paths    []string
}

func (s *sse) server(t *testing.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.paths = append(s.paths, r.Method+" "+r.URL.Path)
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			if strings.HasSuffix(r.URL.Path, "/models") {
				_, _ = io.WriteString(w, `{"data":[{"id":"zeta","type":"model"},{"id":"alpha","type":"model"}],"has_more":false,"first_id":"zeta","last_id":"alpha"}`)
			}
			return
		}
		b, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(b, &req)
		s.requests = append(s.requests, req)
		body := s.bodies[0]
		s.bodies = s.bodies[1:]
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func openAIEvents(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		var probe struct{ Type string }
		_ = json.Unmarshal([]byte(e), &probe)
		var one bytes.Buffer
		_ = json.Compact(&one, []byte(e))
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", probe.Type, one.String())
	}
	return b.String()
}

func TestOpenAIStream(t *testing.T) {
	s := &sse{bodies: []string{
		openAIEvents(
			`{"type":"response.created","response":{"status":"in_progress","output":[]}}`,
			`{"type":"response.output_text.delta","delta":"Hel"}`,
			`{"type":"response.output_text.delta","delta":"lo"}`,
			`{"type":"response.function_call_arguments.delta","delta":"{\"q\""}`,
			`{"type":"response.completed","response":{"status":"completed","output":[
			  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]},
			  {"type":"function_call","call_id":"c1","name":"lookup","arguments":"{\"q\":\"x\"}"}],
			  "usage":{"input_tokens":11,"output_tokens":3}}}`,
		),
		openAIEvents(`{"type":"response.output_text.delta","delta":"done"}`,
			`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}}`),
		openAIEvents(`{"type":"error","message":"rate limited"}`),
		openAIEvents(`{"type":"response.output_text.delta","delta":"cut"}`),
	}}
	srv := s.server(t)
	p := NewOpenAI(OpenAIConfig{BaseURL: srv.URL})
	conv := p.NewConversation(Options{Model: "m1"}, "sys", []ToolSpec{lookupTool})
	var got strings.Builder
	turn, err := conv.(Streamer).SendStream(context.Background(), "hi", nil, func(d string) { got.WriteString(d + "|") })
	if err != nil {
		t.Fatal(err)
	}
	if got.String() != "Hel|lo|" || turn.Text != "Hello" || turn.Usage.InputTokens != 11 {
		t.Fatalf("stream text %q turn %+v", got.String(), turn)
	}
	if len(turn.ToolCalls) != 1 || turn.ToolCalls[0].ID != "c1" || string(turn.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Fatalf("tool calls = %+v", turn.ToolCalls)
	}
	if s.requests[0]["stream"] != true || s.paths[0] != "POST /responses" {
		t.Fatalf("request = %v %v", s.paths[0], s.requests[0])
	}

	conv.(Configurable).SetOptions(Options{Model: "m2"})
	if _, err := conv.(Streamer).SendStream(context.Background(), "", []ToolResult{{CallID: "c1", Content: "r"}}, func(string) {}); err != nil {
		t.Fatal(err)
	}
	items := s.requests[1]["input"].([]any)
	if s.requests[1]["model"] != "m2" || len(items) != 4 || items[1].(map[string]any)["type"] != "message" || items[3].(map[string]any)["output"] != "r" {
		t.Fatalf("second request = model %v input %v", s.requests[1]["model"], items)
	}
	if _, err := conv.(Streamer).SendStream(context.Background(), "x", nil, func(string) {}); err == nil || !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("error event: %v", err)
	}
	if _, err := conv.(Streamer).SendStream(context.Background(), "y", nil, func(string) {}); err == nil || !strings.Contains(err.Error(), "ended before") {
		t.Fatalf("cut stream: %v", err)
	}

	ids, err := p.(ModelLister).ListModels(context.Background())
	if err != nil || strings.Join(ids, ",") != "alpha,zeta" {
		t.Fatalf("models = %v, %v", ids, err)
	}
}

func anthropicEvents(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		var probe struct{ Type string }
		_ = json.Unmarshal([]byte(e), &probe)
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", probe.Type, e)
	}
	return b.String()
}

func TestAnthropicStream(t *testing.T) {
	s := &sse{bodies: []string{anthropicEvents(
		`{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":20,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Look"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ing"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu1","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"x\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
		`{"type":"message_stop"}`,
	)}}
	srv := s.server(t)
	p := NewAnthropic(AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	conv := p.NewConversation(Options{}, "sys", []ToolSpec{lookupTool})
	var got []string
	turn, err := conv.(Streamer).SendStream(context.Background(), "hi", nil, func(d string) { got = append(got, d) })
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Look|ing" || turn.Text != "Looking" || len(turn.ToolCalls) != 1 || string(turn.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Fatalf("got %v turn %+v", got, turn)
	}
	tool := s.requests[0]["tools"].([]any)[0].(map[string]any)
	if tool["eager_input_streaming"] != true || s.requests[0]["stream"] != true {
		t.Fatalf("streamed request = %v", s.requests[0])
	}
	ids, err := p.(ModelLister).ListModels(context.Background())
	if err != nil || strings.Join(ids, ",") != "alpha,zeta" {
		t.Fatalf("models = %v, %v", ids, err)
	}
}
