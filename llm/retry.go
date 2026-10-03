package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// Request timing defaults (Config, OpenAIConfig and AnthropicConfig).
const (
	// DefaultRequestTimeout bounds one attempt at a model call, from sending
	// the request to the end of the response.
	DefaultRequestTimeout = 5 * time.Minute
	// DefaultIdleTimeout bounds silence once the reply's text has started.
	// Before that the model may be reasoning, which sends nothing.
	DefaultIdleTimeout = 60 * time.Second
	// DefaultMaxRetries is how often a call is retried after a timeout, a
	// network error or a 408/409/429/5xx answer, if none of its text reached
	// the caller yet.
	DefaultMaxRetries = 1
	// headerTimeout bounds the wait for response headers. Requests stream,
	// so headers come right away even while the model thinks.
	headerTimeout = 60 * time.Second
	maxRetryWait  = 30 * time.Second
)

// ErrTimeout reports a model call that did not finish in time: no complete
// response within the request timeout, or a reply that stopped arriving.
var ErrTimeout = errors.New("llm: request timed out")

// timing is the resolved timeouts and retries of a provider.
type timing struct {
	request, idle time.Duration
	retries       int
}

func resolveTiming(request, idle time.Duration, retries *int) timing {
	t := timing{request: request, idle: idle, retries: DefaultMaxRetries}
	if t.request <= 0 {
		t.request = DefaultRequestTimeout
	}
	if t.idle <= 0 {
		t.idle = DefaultIdleTimeout
	}
	if retries != nil {
		t.retries = max(*retries, 0)
	}
	return t
}

// defaultHTTPClient streams without an overall timeout (each attempt has
// its own) but gives up on servers that never send headers.
func defaultHTTPClient() *http.Client {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = headerTimeout
	return &http.Client{Transport: tr}
}

// retryAfter parses a Retry-After header given in seconds.
func retryAfter(h http.Header) time.Duration {
	if s, err := strconv.Atoi(h.Get("Retry-After")); err == nil && s > 0 {
		return time.Duration(s) * time.Second
	}
	return 0
}

// retryable reports whether a failed attempt may be repeated, and how long
// the server asked to wait.
func retryable(err error) (bool, time.Duration) {
	var api *APIError
	if errors.As(err, &api) {
		switch api.StatusCode {
		case 408, 409, 429, 500, 502, 503, 504, 529:
			return true, api.RetryAfter
		}
		return false, 0
	}
	var ne net.Error
	return errors.Is(err, ErrTimeout) || errors.Is(err, errIncomplete) || errors.Is(err, io.ErrUnexpectedEOF) || errors.As(err, &ne), 0
}

// errIncomplete reports a stream that ended before its final event.
var errIncomplete = errors.New("the stream ended before the response was complete")

// withRetries runs attempt until it succeeds, returns a turn, fails in a
// way that is not worth repeating, or delivered text (repeating would
// deliver it twice).
func withRetries(ctx context.Context, t timing, attempt func(ctx context.Context, delivered *bool) (*Turn, error)) (*Turn, error) {
	for n := 0; ; n++ {
		delivered := false
		turn, err := attempt(ctx, &delivered)
		if err == nil || turn != nil {
			return turn, err
		}
		ok, wait := retryable(err)
		if !ok || delivered || n >= t.retries || ctx.Err() != nil {
			return nil, err
		}
		if wait <= 0 {
			wait = time.Second << n
		}
		select {
		case <-time.After(min(wait, maxRetryWait)):
		case <-ctx.Done():
			return nil, err
		}
	}
}

// attemptContext bounds one attempt by the request timeout and returns a
// watchdog for silence once text has started.
func attemptContext(ctx context.Context, t timing) (context.Context, *idleWatch, func()) {
	actx, cancel := context.WithCancelCause(ctx)
	deadline := time.AfterFunc(t.request, func() {
		cancel(fmt.Errorf("%w: no complete response within %s", ErrTimeout, t.request))
	})
	idle := &idleWatch{d: t.idle, cancel: cancel}
	return actx, idle, func() {
		deadline.Stop()
		idle.stop()
		cancel(nil)
	}
}

// cause prefers the reason an attempt was cut short over the error it
// surfaced as (usually "context canceled").
func cause(actx context.Context, err error) error {
	if c := context.Cause(actx); c != nil && errors.Is(c, ErrTimeout) {
		return c
	}
	return err
}

// idleWatch cancels an attempt when no stream event arrives for d after
// arm. Before arm it is inert: a reasoning model sends nothing while it
// thinks.
type idleWatch struct {
	d      time.Duration
	cancel context.CancelCauseFunc

	mu    sync.Mutex
	timer *time.Timer
}

func (w *idleWatch) arm() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.timer == nil {
		w.timer = time.AfterFunc(w.d, func() {
			w.cancel(fmt.Errorf("%w: the reply stopped arriving for %s", ErrTimeout, w.d))
		})
	}
}

func (w *idleWatch) touch() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.timer != nil {
		w.timer.Reset(w.d)
	}
}

func (w *idleWatch) stop() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.timer != nil {
		w.timer.Stop()
	}
}
