package cli

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/go-viper/mapstructure/v2"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
)

// Config is the effective CLI configuration. Every setting is resolved by
// viper in the same order: a flag, then its environment variable
// (SENCTL_AGENT_*), then the config file, then the default, which is the
// library's own default where there is one.
type Config struct {
	Provider  string `mapstructure:"provider"`
	BaseURL   string `mapstructure:"base_url"`
	APIKey    string `mapstructure:"api_key"`
	Model     string `mapstructure:"model"`
	Effort    string `mapstructure:"effort"`
	MaxTurns  int    `mapstructure:"max_turns"`
	MaxTokens int64  `mapstructure:"max_tokens"`
	Fallbacks string `mapstructure:"fallbacks"`
	System    string `mapstructure:"system"`
	Dir       string `mapstructure:"dir"`
	Shell     string `mapstructure:"shell"`
	Edit      string `mapstructure:"edit"`
}

// Environment variables the CLI reads besides SENCTL_AGENT_*, as viper keys.
const (
	keyOpenAIKey    = "openai_api_key"    // OPENAI_API_KEY, when api_key is unset
	keyAnthropicKey = "anthropic_api_key" // ANTHROPIC_API_KEY, likewise for anthropic
	keyNoColor      = "no_color"          // NO_COLOR
	keyStateHome    = "xdg_state_home"    // XDG_STATE_HOME, for the console history
	keyConfigFile   = "config_file"       // SENCTL_AGENT_CONFIG, when --config is not given
)

// localConfig is a config file some tools read from the working directory.
// senctl-agent does not: a repository could use it to send the API key and
// its code to an endpoint of its choosing. Name it with --config instead.
const localConfig = ".senctl-agent.yaml"

// bindFlags declares the persistent flags and wires them, the environment
// and defaults into v.
func bindFlags(cmd *cobra.Command, v *viper.Viper) {
	f := cmd.PersistentFlags()
	f.String("config", "", "config file (default $XDG_CONFIG_HOME/senctl-agent/config.yaml; also SENCTL_AGENT_CONFIG)")
	f.String("base-url", "", "LLM proxy (or any endpoint with the OpenAI Responses API), e.g. https://llm-proxy.example.com")
	f.String("model", "", "model id (default: the endpoint's only model, if it lists one)")
	f.String("api-key", "", "API key (prefer SENCTL_AGENT_API_KEY or the config file: flags show up in the process list)")
	f.String("provider", "", "API to speak: openai (default; the OpenAI Responses API, which proxies serve) or anthropic (the Anthropic API directly)")
	f.String("fallbacks", "", "anthropic: server-side refusal fallback, true or false (default: on for the first-party API)")
	f.String("effort", "", "reasoning effort: low, medium, high")
	f.Int("max-turns", agent.DefaultMaxTurns, "maximum model calls per prompt")
	f.Int64("max-tokens", llm.DefaultMaxTokens, "maximum output tokens per model call")
	f.String("system", "", "system prompt (replaces the default)")
	f.String("dir", ".", "workspace directory the file tools can read")
	f.String("shell", "off", "shell tool: off, ask (confirm each command) or auto")
	f.String("edit", "", "file editing: off, ask (confirm each change) or auto (default: ask in the console, off for run)")
	f.Bool("debug", false, "show the full error when a command fails")
	for _, name := range []string{"debug", "api-key", "provider", "fallbacks", "base-url", "model", "effort", "max-turns", "max-tokens", "system", "dir", "shell", "edit"} {
		_ = v.BindPFlag(strings.ReplaceAll(name, "-", "_"), f.Lookup(name))
	}
	v.SetEnvPrefix("SENCTL_AGENT")
	v.SetEnvKeyReplacer(strings.NewReplacer("-", "_"))
	v.AutomaticEnv()
	_ = v.BindEnv(keyOpenAIKey, "OPENAI_API_KEY")
	_ = v.BindEnv(keyAnthropicKey, "ANTHROPIC_API_KEY")
	_ = v.BindEnv(keyNoColor, "NO_COLOR")
	_ = v.BindEnv(keyStateHome, "XDG_STATE_HOME")
	_ = v.BindEnv(keyConfigFile, "SENCTL_AGENT_CONFIG")
}

// configFile is the user config file's path, for messages.
func configFile() string {
	if dir, err := os.UserConfigDir(); err == nil {
		return filepath.Join(dir, "senctl-agent", "config.yaml")
	}
	return "the config file"
}

// load reads the config file (if any) and returns the effective config.
// The file is the one named by --config or SENCTL_AGENT_CONFIG, else the
// user's own; never one found in the working directory.
func load(cmd *cobra.Command, v *viper.Viper) (*Config, error) {
	path, _ := cmd.Flags().GetString("config")
	if path == "" {
		path = v.GetString(keyConfigFile)
	}
	if path != "" {
		v.SetConfigFile(path)
		if err := v.ReadInConfig(); err != nil {
			return nil, configFileError(path, err)
		}
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		if dir, err := os.UserConfigDir(); err == nil {
			v.AddConfigPath(filepath.Join(dir, "senctl-agent"))
		}
		if err := v.ReadInConfig(); err != nil {
			if _, notFound := err.(viper.ConfigFileNotFoundError); !notFound {
				return nil, configFileError(v.ConfigFileUsed(), err)
			}
		}
		if _, err := os.Stat(localConfig); err == nil {
			fmt.Fprintf(cmd.ErrOrStderr(), "senctl-agent: ignoring ./%s; pass --config %s to use it\n", localConfig, localConfig)
		}
	}
	var c Config
	if err := v.Unmarshal(&c); err != nil {
		return nil, decodeError(err)
	}
	c.Provider = strings.ToLower(strings.TrimSpace(c.Provider))
	if c.APIKey == "" {
		if c.Provider == llm.ProviderAnthropic {
			c.APIKey = v.GetString(keyAnthropicKey)
		} else {
			c.APIKey = v.GetString(keyOpenAIKey)
		}
	}
	for name, v := range map[string]string{"shell": c.Shell, "edit": c.Edit} {
		switch v {
		case "", "off", "ask", "auto":
		default:
			return nil, configError{fmt.Errorf("%s must be off, ask or auto, not %q", name, v), settingHint(name)}
		}
	}
	if _, err := c.fallbacks(); err != nil {
		return nil, err
	}
	return &c, nil
}

// fallbacks parses the fallbacks setting; nil means unset.
func (c *Config) fallbacks() (*bool, error) {
	var b bool
	switch strings.ToLower(strings.TrimSpace(c.Fallbacks)) {
	case "":
		return nil, nil
	case "true", "1", "yes", "on":
		b = true
	case "false", "0", "no", "off":
	default:
		return nil, configError{fmt.Errorf("fallbacks must be true or false, not %q", c.Fallbacks), settingHint("fallbacks")}
	}
	return &b, nil
}

// configFileError explains why the config file at path cannot be read.
func configFileError(path string, err error) error {
	var parse viper.ConfigParseError
	switch {
	case errors.Is(err, fs.ErrNotExist):
		err = fmt.Errorf("config file %s does not exist", path)
	case errors.As(err, &parse):
		err = fmt.Errorf("config file %s: %w", path, parse.Unwrap())
	default:
		err = fmt.Errorf("config file %s: %w", path, err)
	}
	return configError{err, "check --config or SENCTL_AGENT_CONFIG"}
}

// decodeError explains a setting whose value has the wrong type, such as
// max_turns: lots.
func decodeError(err error) error {
	var de *mapstructure.DecodeError
	if !errors.As(err, &de) {
		return configError{fmt.Errorf("config: %w", err), ""}
	}
	var pe *mapstructure.ParseError
	if errors.As(err, &pe) {
		want := "a " + pe.Expected.Type().String()
		switch pe.Expected.Kind() {
		case reflect.Int, reflect.Int64:
			want = "a whole number"
		case reflect.Bool:
			want = "true or false"
		}
		err = fmt.Errorf("%s must be %s, not %q", de.Name(), want, fmt.Sprint(pe.Value))
	} else {
		err = fmt.Errorf("%s: %w", de.Name(), de.Unwrap())
	}
	return configError{err, settingHint(de.Name())}
}

// provider builds the configured model provider. Without a model it takes
// the library's default (llm.ResolveModel).
func (c *Config) provider(ctx context.Context) (llm.Provider, llm.Options, error) {
	fallbacks, err := c.fallbacks()
	if err != nil {
		return nil, llm.Options{}, err
	}
	p, err := llm.New(llm.Config{Provider: c.Provider, APIKey: c.APIKey, BaseURL: c.BaseURL, Fallbacks: fallbacks})
	if err != nil {
		return nil, llm.Options{}, err
	}
	model, err := llm.ResolveModel(ctx, p, c.Model)
	if err != nil {
		return nil, llm.Options{}, err
	}
	c.Model = model
	return p, llm.Options{Model: c.Model, Effort: c.Effort, MaxTokens: c.MaxTokens}, nil
}

func mask(key string) string {
	if key == "" {
		return "(not set)"
	}
	if len(key) <= 8 {
		return "****"
	}
	return key[:4] + "…" + key[len(key)-4:]
}
