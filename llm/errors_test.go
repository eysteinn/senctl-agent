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

func TestErrorKinds(t *testing.T) {
	for _, tt := range []struct {
		name   string
		status int
		body   string
		is     []error
		isNot  []error
		msg    string
	}{
		{"openai unknown model", 404, `{"error": {"message": "The model x does not exist", "type": "invalid_request_error", "code": "model_not_found"}}`,
			[]error{ErrModelNotFound}, []error{ErrNotAPI, ErrUnauthorized}, "HTTP 404: The model x does not exist"},
		{"anthropic unknown model", 404, `{"type": "error", "error": {"type": "not_found_error", "message": "model: x"}}`,
			[]error{ErrModelNotFound}, []error{ErrNotAPI}, "HTTP 404: model: x"},
		{"other 404", 404, `{"error": {"message": "no such route", "type": "not_found_error"}}`,
			nil, []error{ErrModelNotFound, ErrNotAPI}, "HTTP 404: no such route"},
		{"web page", 404, "<!DOCTYPE html>\n<html><head><title>404 &amp; gone</title></head></html>",
			[]error{ErrNotAPI}, []error{ErrModelNotFound}, `HTTP 404: a web page ("404 & gone"), not an API response`},
		{"login page", 401, "<html><body>sign in</body></html>",
			[]error{ErrNotAPI}, []error{ErrUnauthorized, ErrNoAPIKey}, "HTTP 401: a web page, not an API response"},
		{"plain text", 502, "upstream down",
			nil, []error{ErrNotAPI}, "HTTP 502: upstream down"},
	} {
		err := error(newAPIError(tt.status, []byte(tt.body), true))
		for _, target := range tt.is {
			if !errors.Is(err, target) {
				t.Errorf("%s: %v should match %v", tt.name, err, target)
			}
		}
		for _, target := range tt.isNot {
			if errors.Is(err, target) {
				t.Errorf("%s: %v should not match %v", tt.name, err, target)
			}
		}
		if err.Error() != tt.msg {
			t.Errorf("%s: message %q, want %q", tt.name, err, tt.msg)
		}
	}

	for _, cfg := range []Config{{Provider: "gemini", APIKey: "k"}, {BaseURL: "proxy.example.com"}, {BaseURL: "ftp://proxy"}, {BaseURL: "http://"}} {
		var ce *ConfigError
		if _, err := New(cfg); !errors.As(err, &ce) {
			t.Errorf("New(%+v) = %v, want a ConfigError", cfg, err)
		}
	}
}
