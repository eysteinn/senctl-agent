package llm

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
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
)

// APIError is an error response from the endpoint.
type APIError struct {
	StatusCode int
	// Message is the API's own error message, or the start of the
	// response body when it has none.
	Message string
	// keyless records that the request carried no API key.
	keyless bool
}

func (e *APIError) Error() string { return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message) }

// Is matches ErrUnauthorized for HTTP 401 and 403, and ErrNoAPIKey as well
// when the request was sent without a key.
func (e *APIError) Is(target error) bool {
	auth := e.StatusCode == http.StatusUnauthorized || e.StatusCode == http.StatusForbidden
	switch target {
	case ErrUnauthorized:
		return auth
	case ErrNoAPIKey:
		return auth && e.keyless
	}
	return false
}

// newAPIError builds an APIError from a response body.
func newAPIError(status int, raw []byte, keyless bool) *APIError {
	msg := apiError(raw)
	if msg == "" {
		msg = snippet(raw)
	}
	if msg == "" {
		msg = http.StatusText(status)
	}
	return &APIError{StatusCode: status, Message: msg, keyless: keyless}
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
