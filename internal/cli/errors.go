package cli

import (
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/eysteinn/senctl-agent/llm"
)

// Exit codes.
const (
	exitOK     = 0
	exitError  = 1 // the command failed
	exitUsage  = 2 // bad arguments or flags
	exitConfig = 3 // the settings are missing or wrong: no or a rejected API key, an unreachable endpoint, no model
)

// usageError is a mistake in how the command was called.
type usageError struct {
	err     error
	cmdPath string
}

func (e usageError) Error() string { return e.err.Error() }
func (e usageError) Unwrap() error { return e.err }

// configError is a config file or setting that cannot be used.
type configError struct{ err error }

func (e configError) Error() string { return e.err.Error() }
func (e configError) Unwrap() error { return e.err }

// explain turns err into a message for a person, a hint on how to fix it
// when the fix is a setting, and an exit code.
func explain(err error) (msg, hint string, code int) {
	var (
		apiErr *llm.APIError
		urlErr *url.Error
		usage  usageError
		cfg    configError
	)
	errors.As(err, &apiErr)
	switch {
	case errors.As(err, &usage):
		return err.Error(), fmt.Sprintf("run '%s --help' for usage", usage.cmdPath), exitUsage
	case errors.As(err, &cfg):
		return err.Error(), "", exitConfig
	case errors.Is(err, llm.ErrNoAPIKey):
		return "no API key set",
			"set SENCTL_AGENT_API_KEY, api_key: in " + configFile() + " or --api-key (OPENAI_API_KEY and ANTHROPIC_API_KEY work too)",
			exitConfig
	case errors.Is(err, llm.ErrUnauthorized):
		return fmt.Sprintf("the endpoint rejected the API key (%s)", apiErr),
			"check SENCTL_AGENT_API_KEY or api_key: in the config file; 'senctl-agent config' shows the key in use, masked",
			exitConfig
	case errors.Is(err, llm.ErrUnreachable):
		msg = "cannot reach the endpoint: " + err.Error()
		if errors.As(err, &urlErr) {
			if u, perr := url.Parse(urlErr.URL); perr == nil {
				msg = fmt.Sprintf("cannot reach %s://%s: %v", u.Scheme, u.Host, urlErr.Err)
			}
		}
		return msg, "check --base-url, SENCTL_AGENT_BASE_URL or base_url: in the config file", exitConfig
	case errors.Is(err, llm.ErrNoModel):
		hint = "set --model, SENCTL_AGENT_MODEL or model: in the config file"
		if apiErr == nil {
			hint += " ('senctl-agent models' lists them)"
		}
		return strings.TrimPrefix(err.Error(), "llm: "), hint, exitConfig
	}
	return err.Error(), "", exitError
}

// report writes err to w as "senctl-agent: message", with the hint on the
// next line, and returns the exit code. With debug the full error follows
// when the message leaves part of it out.
func report(w io.Writer, err error, debug bool) int {
	msg, hint, code := explain(err)
	fmt.Fprintf(w, "senctl-agent: %s\n", msg)
	if hint != "" {
		fmt.Fprintf(w, "  %s\n", hint)
	}
	if debug && msg != err.Error() {
		fmt.Fprintf(w, "  error: %v\n", err)
	}
	return code
}
