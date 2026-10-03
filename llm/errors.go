package llm

import (
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"net"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// Errors callers can test for with errors.Is, whichever provider failed.
var (
	// ErrNoAPIKey: no API key was given and the endpoint needs one. New
	// returns it for a first-party API without a key; a request sent
	// without a key that the endpoint answers with HTTP 401 or 403 matches
	// it too.
	ErrNoAPIKey = errors.New("llm: no API key")
	// ErrUnauthorized: the endpoint rejected the request's credentials
	// (HTTP 401 or 403).
	ErrUnauthorized = errors.New("llm: the endpoint rejected the API key")
	// ErrUnreachable: no connection could be made to the endpoint (DNS,
	// connection refused, TLS certificate).
	ErrUnreachable = errors.New("llm: the endpoint could not be reached")
	// ErrNoModel: no model was given and none could be chosen for the
	// caller (see ResolveModel).
	ErrNoModel = errors.New("llm: no model given")
	// ErrModelNotFound: the endpoint does not serve the requested model, or
	// not to this key.
	ErrModelNotFound = errors.New("llm: the endpoint does not serve the model")
	// ErrNotAPI: the endpoint answered with a web page instead of an API
	// response, which usually means the base URL is wrong.
	ErrNotAPI = errors.New("llm: the endpoint did not answer like an LLM API")
)

// ConfigError reports a Config field that cannot be used.
type ConfigError struct {
	Field string // the Config field: "Provider" or "BaseURL"
	Msg   string
}

func (e *ConfigError) Error() string { return "llm: " + e.Msg }

// APIError is an error response from the endpoint.
type APIError struct {
	StatusCode int
	// Message is the API's own error message, or the start of the
	// response body when it has none.
	Message string
	// Code is the API's machine-readable error code (OpenAI's error.code,
	// else the error type), or "".
	Code string
	// RetryAfter is how long the server asked to wait before retrying
	// (Retry-After, in seconds), or 0.
	RetryAfter time.Duration
	// keyless records that the request carried no API key.
	keyless bool
	// html records that the body was a web page, not an API error.
	html bool
}

func (e *APIError) Error() string { return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message) }

// Is matches ErrUnauthorized for HTTP 401 and 403 (and ErrNoAPIKey as
// well when the request was sent without a key), ErrModelNotFound when the
// API says the model does not exist, and ErrNotAPI for a web page.
func (e *APIError) Is(target error) bool {
	auth := e.StatusCode == http.StatusUnauthorized || e.StatusCode == http.StatusForbidden
	switch target {
	case ErrUnauthorized:
		return auth && !e.html
	case ErrNoAPIKey:
		return auth && !e.html && e.keyless
	case ErrModelNotFound:
		// OpenAI: code model_not_found. Anthropic: not_found_error, "model: …".
		return e.Code == "model_not_found" || e.Code == "not_found_error" && strings.HasPrefix(e.Message, "model:")
	case ErrNotAPI:
		return e.html
	}
	return false
}

// newAPIError builds an APIError from a response body.
func newAPIError(status int, raw []byte, keyless bool) *APIError {
	e := &APIError{StatusCode: status, keyless: keyless}
	var body struct {
		Error *struct {
			Message string `json:"message"`
			Code    any    `json:"code"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	switch {
	case json.Unmarshal(raw, &body) == nil && body.Error != nil:
		e.Message, e.Code = body.Error.Message, body.Error.Type
		if code, ok := body.Error.Code.(string); ok && code != "" {
			e.Code = code
		}
	case isHTML(raw):
		e.html = true
		e.Message = "a web page, not an API response"
		if title := htmlTitle(raw); title != "" {
			e.Message = fmt.Sprintf("a web page (%q), not an API response", title)
		}
	default:
		e.Message = snippet(raw)
	}
	if e.Message == "" {
		e.Message = http.StatusText(status)
	}
	return e
}

func isHTML(raw []byte) bool {
	s := strings.ToLower(strings.TrimSpace(string(raw[:min(len(raw), 512)])))
	return strings.HasPrefix(s, "<!doctype html") || strings.HasPrefix(s, "<html")
}

var titleRE = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)

func htmlTitle(raw []byte) string {
	if m := titleRE.FindSubmatch(raw); m != nil {
		return strings.Join(strings.Fields(html.UnescapeString(string(m[1]))), " ")
	}
	return ""
}

type unreachableError struct{ err error }

func (e unreachableError) Error() string        { return e.err.Error() }
func (e unreachableError) Unwrap() error        { return e.err }
func (e unreachableError) Is(target error) bool { return target == ErrUnreachable }

// markUnreachable makes err, from sending a request, match ErrUnreachable
// when no connection to the endpoint could be made.
func markUnreachable(err error) error {
	var dns *net.DNSError
	var op *net.OpError
	var cert *tls.CertificateVerificationError
	if errors.As(err, &dns) || errors.As(err, &op) && op.Op == "dial" || errors.As(err, &cert) {
		return unreachableError{err}
	}
	return err
}
