package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/eysteinn/senctl-agent/llm"
)

// Config is the effective CLI configuration: flags override environment
// (SENCTL_AGENT_*), which overrides the config file.
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

// bindFlags declares the persistent flags and wires them, the environment
// and defaults into v.
func bindFlags(cmd *cobra.Command, v *viper.Viper) {
	f := cmd.PersistentFlags()
	f.String("config", "", "config file (default $XDG_CONFIG_HOME/senctl-agent/config.yaml, then ./.senctl-agent.yaml)")
	f.String("base-url", "", "LLM proxy (or any endpoint with the OpenAI Responses API), e.g. https://llm-proxy.example.com")
	f.String("model", "", "model id (default: the endpoint's only model, if it lists one)")
	f.String("provider", "", "API to speak: openai (default; the OpenAI Responses API, which proxies serve) or anthropic (the Anthropic API directly)")
	f.String("effort", "", "reasoning effort: low, medium, high")
	f.Int("max-turns", 30, "maximum model calls per prompt")
	f.Int64("max-tokens", 16000, "maximum output tokens per model call")
	f.String("system", "", "system prompt (replaces the default)")
	f.String("dir", ".", "workspace directory the file tools can read")
	f.String("shell", "off", "shell tool: off, ask (confirm each command) or auto")
	f.String("edit", "", "file editing: off, ask (confirm each change) or auto (default: ask in the console, off for run)")
	for _, name := range []string{"provider", "base-url", "model", "effort", "max-turns", "max-tokens", "system", "dir", "shell", "edit"} {
		_ = v.BindPFlag(strings.ReplaceAll(name, "-", "_"), f.Lookup(name))
	}
	v.SetEnvPrefix("SENCTL_AGENT")
	v.SetEnvKeyReplacer(strings.NewReplacer("-", "_"))
	v.AutomaticEnv()
	for _, k := range []string{"api_key", "fallbacks"} {
		_ = v.BindEnv(k)
	}
}

// load reads the config file (if any) and returns the effective config.
func load(cmd *cobra.Command, v *viper.Viper) (*Config, error) {
	if path, _ := cmd.Flags().GetString("config"); path != "" {
		v.SetConfigFile(path)
		if err := v.ReadInConfig(); err != nil {
			return nil, fmt.Errorf("read config %s: %w", path, err)
		}
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		if dir, err := os.UserConfigDir(); err == nil {
			v.AddConfigPath(filepath.Join(dir, "senctl-agent"))
		}
		if err := v.ReadInConfig(); err != nil {
			if _, notFound := err.(viper.ConfigFileNotFoundError); !notFound {
				return nil, fmt.Errorf("read config: %w", err)
			}
			v.SetConfigName(".senctl-agent")
			v.AddConfigPath(".")
			if err := v.ReadInConfig(); err != nil {
				if _, notFound := err.(viper.ConfigFileNotFoundError); !notFound {
					return nil, fmt.Errorf("read config: %w", err)
				}
			}
		}
	}
	var c Config
	if err := v.Unmarshal(&c); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	c.Provider = strings.ToLower(strings.TrimSpace(c.Provider))
	if c.APIKey == "" {
		if c.Provider == llm.ProviderAnthropic {
			c.APIKey = os.Getenv("ANTHROPIC_API_KEY")
		} else {
			c.APIKey = os.Getenv("OPENAI_API_KEY")
		}
	}
	if c.Model == "" {
		c.Model = llm.DefaultModel(c.Provider)
	}
	for name, v := range map[string]string{"shell": c.Shell, "edit": c.Edit} {
		switch v {
		case "", "off", "ask", "auto":
		default:
			return nil, fmt.Errorf("--%s must be off, ask or auto", name)
		}
	}
	return &c, nil
}

// provider builds the configured model provider. Without a model it uses
// the endpoint's only model, or says which ones there are.
func (c *Config) provider(ctx context.Context) (llm.Provider, llm.Options, error) {
	cfg := llm.Config{Provider: c.Provider, APIKey: c.APIKey, BaseURL: c.BaseURL}
	switch strings.ToLower(c.Fallbacks) {
	case "true", "1", "yes", "on":
		t := true
		cfg.Fallbacks = &t
	case "false", "0", "no", "off":
		f := false
		cfg.Fallbacks = &f
	}
	p, err := llm.New(cfg)
	if err != nil {
		return nil, llm.Options{}, err
	}
	if c.Model == "" {
		if c.Model, err = onlyModel(ctx, p); err != nil {
			return nil, llm.Options{}, err
		}
	}
	return p, llm.Options{Model: c.Model, Effort: c.Effort, MaxTokens: c.MaxTokens}, nil
}

// onlyModel asks the endpoint which models it serves and returns the one
// there is; with several, the choice is the user's.
func onlyModel(ctx context.Context, p llm.Provider) (string, error) {
	const hint = "set --model, SENCTL_AGENT_MODEL or model: in the config file"
	ml, ok := p.(llm.ModelLister)
	if !ok {
		return "", fmt.Errorf("no model configured: %s", hint)
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	ids, err := ml.ListModels(ctx)
	switch {
	case err != nil:
		return "", fmt.Errorf("no model configured, and listing the endpoint's models failed: %w (%s)", err, hint)
	case len(ids) == 1:
		return ids[0], nil
	case len(ids) == 0:
		return "", fmt.Errorf("no model configured and the endpoint lists none: %s", hint)
	}
	shown := ids
	if len(shown) > 30 {
		shown = shown[:30]
	}
	more := ""
	if len(ids) > len(shown) {
		more = fmt.Sprintf(" (and %d more; senctl-agent models lists them all)", len(ids)-len(shown))
	}
	return "", fmt.Errorf("no model configured; the endpoint offers: %s%s. Pick one: %s", strings.Join(shown, ", "), more, hint)
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
