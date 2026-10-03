package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// event writes one Responses API stream event.
func event(w http.ResponseWriter, typ, extra string) {
	fmt.Fprintf(w, "event: %s\ndata: {\"type\":%q%s}\n\n", typ, typ, extra)
	w.(http.Flusher).Flush()
}

func completed(w http.ResponseWriter, text string) {
	event(w, "response.completed", `,"response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"`+text+`"}]}]}`)
}

// script serves one behaviour per request, in order.
func script(t *testing.T, steps ...func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := int(n.Add(1)) - 1
		if i >= len(steps) {
			http.Error(w, "unexpected request", http.StatusTeapot)
			return
		}
		steps[i](w, r)
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

func streamOK(text string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		event(w, "response.created", "")
		event(w, "response.output_text.delta", `,"delta":"`+text+`"`)
		completed(w, text)
	}
}

func fast(srv *httptest.Server, retries int) Provider {
	return NewOpenAI(OpenAIConfig{BaseURL: srv.URL + "/v1", RequestTimeout: 300 * time.Millisecond, IdleTimeout: 100 * time.Millisecond, MaxRetries: &retries})
}

func TestRetriesTransientFailures(t *testing.T) {
	srv, n := script(t,
		func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, `{"error":{"message":"overloaded"}}`, http.StatusServiceUnavailable)
		},
		streamOK("hello"))
	turn, err := fast(srv, 1).NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hi", nil)
	if err != nil || turn.Text != "hello" || n.Load() != 2 {
		t.Fatalf("turn %+v err %v after %d requests", turn, err, n.Load())
	}
}

func TestRetryAfterIsHonoured(t *testing.T) {
	srv, n := script(t,
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", "1")
			http.Error(w, `{"error":{"message":"rate limited"}}`, http.StatusTooManyRequests)
		},
		streamOK("ok"))
	start := time.Now()
	turn, err := fast(srv, 1).NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hi", nil)
	if err != nil || turn.Text != "ok" || n.Load() != 2 || time.Since(start) < time.Second {
		t.Fatalf("turn %+v err %v, %d requests in %s", turn, err, n.Load(), time.Since(start))
	}
}

func TestNoRetryForClientErrors(t *testing.T) {
	srv, n := script(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"bad request"}}`, http.StatusBadRequest)
	})
	_, err := fast(srv, 3).NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hi", nil)
	var api *APIError
	if !errors.As(err, &api) || api.StatusCode != 400 || n.Load() != 1 {
		t.Fatalf("err %v after %d requests", err, n.Load())
	}
}

func TestSilentThinkingIsAllowedButAHungRequestIsRetried(t *testing.T) {
	srv, n := script(t,
		// Created, then nothing: longer than the idle timeout (fine, the model
		// may be thinking) but past the request timeout, so it is cut and retried.
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			event(w, "response.created", "")
			select {
			case <-r.Context().Done():
			case <-time.After(5 * time.Second):
			}
		},
		// Thinks quietly for longer than the idle timeout, then answers.
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			event(w, "response.created", "")
			time.Sleep(200 * time.Millisecond)
			event(w, "response.output_text.delta", `,"delta":"done"`)
			completed(w, "done")
		})
	start := time.Now()
	turn, err := fast(srv, 1).NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hi", nil)
	if err != nil || turn.Text != "done" || n.Load() != 2 {
		t.Fatalf("turn %+v err %v after %d requests", turn, err, n.Load())
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Fatalf("took %s: the hung attempt was not cut at the request timeout", d)
	}
}

func TestStalledReplyIsNotRetriedAfterText(t *testing.T) {
	srv, n := script(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		event(w, "response.output_text.delta", `,"delta":"half an "`)
		<-r.Context().Done()
	})
	var got strings.Builder
	conv := fast(srv, 2).NewConversation(Options{Model: "m"}, "", nil)
	_, err := conv.(Streamer).SendStream(context.Background(), "hi", nil, func(s string) { got.WriteString(s) })
	if !errors.Is(err, ErrTimeout) || !strings.Contains(err.Error(), "stopped arriving") || n.Load() != 1 || got.String() != "half an " {
		t.Fatalf("err %v, %d requests, text %q", err, n.Load(), got.String())
	}
	// The failed turn is not kept: the next message starts clean.
	if items := conv.(*openAIConversation).items; len(items) != 0 {
		t.Fatalf("history kept %d items", len(items))
	}
}

func TestRetriesCanBeTurnedOff(t *testing.T) {
	srv, n := script(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"overloaded"}}`, http.StatusBadGateway)
	})
	if _, err := fast(srv, 0).NewConversation(Options{Model: "m"}, "", nil).Send(context.Background(), "hi", nil); err == nil || n.Load() != 1 {
		t.Fatalf("err %v after %d requests", err, n.Load())
	}
}

func TestTimingDefaults(t *testing.T) {
	if got := resolveTiming(0, 0, nil); got.request != DefaultRequestTimeout || got.idle != DefaultIdleTimeout || got.retries != DefaultMaxRetries {
		t.Fatalf("defaults = %+v", got)
	}
	neg := -3
	if got := resolveTiming(time.Second, 2*time.Second, &neg); got.request != time.Second || got.idle != 2*time.Second || got.retries != 0 {
		t.Fatalf("explicit = %+v", got)
	}
}

func TestAnthropicStalledReply(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range []string{
			`{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"half"}}`,
		} {
			var probe struct{ Type string }
			_ = jsonUnmarshal(e, &probe)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", probe.Type, e)
		}
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	zero := 0
	p := NewAnthropic(AnthropicConfig{APIKey: "k", BaseURL: srv.URL, IdleTimeout: 100 * time.Millisecond, MaxRetries: &zero})
	start := time.Now()
	_, err := p.NewConversation(Options{Model: "m"}, "", nil).(Streamer).SendStream(context.Background(), "hi", nil, func(string) {})
	if !errors.Is(err, ErrTimeout) || time.Since(start) > 3*time.Second {
		t.Fatalf("err %v after %s", err, time.Since(start))
	}
}

func jsonUnmarshal(s string, v any) error { return json.Unmarshal([]byte(s), v) }
