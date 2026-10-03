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

// configError is a config file or setting that cannot be used, with a
// hint on where to fix it.
type configError struct {
	err  error
	hint string
}

func (e configError) Error() string { return e.err.Error() }
func (e configError) Unwrap() error { return e.err }

// settingHint names the places setting key (snake_case) can be set.
func settingHint(key string) string {
	return fmt.Sprintf("check --%s, SENCTL_AGENT_%s or %s: in %s",
		strings.ReplaceAll(key, "_", "-"), strings.ToUpper(key), key, configFile())
}

// llmSettings maps llm.ConfigError fields to settings.
var llmSettings = map[string]string{"Provider": "provider", "BaseURL": "base_url"}

// explain turns err into a message for a person, a hint on how to fix it
// when the fix is a setting, and an exit code.
func explain(err error) (msg, hint string, code int) {
	var (
		apiErr *llm.APIError
		urlErr *url.Error
		usage  usageError
		cfg    configError
		llmCfg *llm.ConfigError
	)
	errors.As(err, &apiErr)
	switch {
	case errors.As(err, &usage):
		return err.Error(), fmt.Sprintf("run '%s --help' for usage", usage.cmdPath), exitUsage
	case errors.As(err, &cfg):
		return err.Error(), cfg.hint, exitConfig
	case errors.As(err, &llmCfg):
		return strings.TrimPrefix(err.Error(), "llm: "), settingHint(llmSettings[llmCfg.Field]), exitConfig
	case errors.Is(err, llm.ErrTimeout):
		return "the model did not answer in time, even after retrying: " + strings.TrimPrefix(err.Error(), "llm: "),
			"a slow model or high effort may need longer; raise --request-timeout, SENCTL_AGENT_REQUEST_TIMEOUT or request_timeout: in " + configFile(),
			exitError
	case errors.Is(err, llm.ErrNotAPI):
		return fmt.Sprintf("the endpoint did not answer like an LLM API (%s)", apiErr),
			settingHint("base_url") + "; it should be the API root, e.g. https://llm-proxy.example.com/v1",
			exitConfig
	case errors.Is(err, llm.ErrModelNotFound):
		return fmt.Sprintf("the endpoint does not serve this model (%s)", apiErr),
			settingHint("model") + " ('senctl-agent models' lists them)",
			exitConfig
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
		return msg, settingHint("base_url"), exitConfig
	case errors.Is(err, llm.ErrNoModel):
		hint = "set --model, SENCTL_AGENT_MODEL or model: in " + configFile()
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
	fmt.Fprintf(w, "senctl-agent: %s\n", tildePath(msg))
	if hint != "" {
		fmt.Fprintf(w, "  %s\n", tildePath(hint))
	}
	if debug && msg != err.Error() {
		fmt.Fprintf(w, "  error: %v\n", err)
	}
	return code
}

// flagError drops Go's parser detail from a bad flag value:
// invalid argument "lots" for "--max-turns" flag: strconv.ParseInt: … becomes
// invalid argument "lots" for "--max-turns" flag: not a whole number.
func flagError(err error) error {
	msg := err.Error()
	i := strings.Index(msg, ": strconv.Parse")
	if i < 0 {
		return err
	}
	want := "not a number"
	switch {
	case strings.HasPrefix(msg[i:], ": strconv.ParseInt"), strings.HasPrefix(msg[i:], ": strconv.ParseUint"):
		want = "not a whole number"
	case strings.HasPrefix(msg[i:], ": strconv.ParseBool"):
		want = "not true or false"
	}
	return errors.New(msg[:i] + ": " + want)
}
