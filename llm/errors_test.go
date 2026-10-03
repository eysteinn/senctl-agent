package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenAIErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer good" {
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error": {"message": "Missing bearer authentication in header", "type": "invalid_request_error"}}`))
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("upstream down"))
	}))
	defer srv.Close()
	ctx := context.Background()
	list := func(key string) error {
		_, err := NewOpenAI(OpenAIConfig{BaseURL: srv.URL + "/v1", APIKey: key}).(ModelLister).ListModels(ctx)
		return err
	}

	err := list("")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 || apiErr.Message != "Missing bearer authentication in header" {
		t.Fatalf("no key: %v", err)
	}
	if !errors.Is(err, ErrNoAPIKey) || !errors.Is(err, ErrUnauthorized) {
		t.Errorf("no key: %v should match ErrNoAPIKey and ErrUnauthorized", err)
	}
	if err := list("bad"); !errors.Is(err, ErrUnauthorized) || errors.Is(err, ErrNoAPIKey) {
		t.Errorf("bad key: %v should match ErrUnauthorized only", err)
	}
	err = list("good")
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 500 || errors.Is(err, ErrUnauthorized) || err.Error() != "openai: list models: HTTP 500: upstream down" {
		t.Errorf("server error: %v", err)
	}

	conv := NewOpenAI(OpenAIConfig{BaseURL: srv.URL + "/v1"}).NewConversation(Options{Model: "m"}, "", nil)
	if _, err := conv.Send(ctx, "hi", nil); !errors.Is(err, ErrNoAPIKey) || err.Error() != "openai: HTTP 401: Missing bearer authentication in header" {
		t.Errorf("send without key: %v", err)
	}

	srv.Close()
	if err := list("good"); !errors.Is(err, ErrUnreachable) {
		t.Errorf("closed server: %v should match ErrUnreachable", err)
	}
	if _, err := ResolveModel(ctx, NewOpenAI(OpenAIConfig{BaseURL: srv.URL}), ""); !errors.Is(err, ErrNoModel) || !errors.Is(err, ErrUnreachable) {
		t.Errorf("ResolveModel against a closed server: %v should match ErrNoModel and ErrUnreachable", err)
	}
}

func TestAnthropicErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}`))
	}))
	defer srv.Close()
	_, err := NewAnthropic(AnthropicConfig{BaseURL: srv.URL, APIKey: "bad"}).(ModelLister).ListModels(context.Background())
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Message != "invalid x-api-key" || !errors.Is(err, ErrUnauthorized) || errors.Is(err, ErrNoAPIKey) {
		t.Fatalf("anthropic bad key: %v", err)
	}
}
