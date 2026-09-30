package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/eysteinn/senctl-agent/llm"
)

// Config is the effective CLI configuration: flags override environment
// (SENCTL_AGENT_*), which overrides the config file.
type Config struct {
	Provider   string `mapstructure:"provider"`
	BaseURL    string `mapstructure:"base_url"`
	APIKey     string `mapstructure:"api_key"`
	Model      string `mapstructure:"model"`
	Effort     string `mapstructure:"effort"`
	MaxTurns   int    `mapstructure:"max_turns"`
	MaxTokens  int64  `mapstructure:"max_tokens"`
	Fallbacks  string `mapstructure:"fallbacks"`
	SendEffort bool   `mapstructure:"send_effort"`
	System     string `mapstructure:"system"`
	Dir        string `mapstructure:"dir"`
	Shell      string `mapstructure:"shell"`
	Edit       string `mapstructure:"edit"`
}

// bindFlags declares the persistent flags and wires them, the environment
// and defaults into v.
func bindFlags(cmd *cobra.Command, v *viper.Viper) {
	f := cmd.PersistentFlags()
	f.String("config", "", "config file (default $XDG_CONFIG_HOME/senctl-agent/config.yaml, then ./.senctl-agent.yaml)")
	f.String("provider", "", "model provider: openai (any OpenAI-compatible endpoint or LLM proxy) or anthropic")
	f.String("base-url", "", "endpoint, e.g. https://llm-proxy.example.com/v1")
	f.String("model", "", "model id (default claude-opus-5-5 for anthropic)")
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
	for _, k := range []string{"api_key", "fallbacks", "send_effort"} {
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
		switch c.Provider {
		case llm.ProviderAnthropic:
			c.APIKey = os.Getenv("ANTHROPIC_API_KEY")
		case llm.ProviderOpenAI:
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

// provider builds the configured model provider.
func (c *Config) provider() (llm.Provider, llm.Options, error) {
	if c.Provider == "" {
		return nil, llm.Options{}, fmt.Errorf("no provider configured: set --provider or SENCTL_AGENT_PROVIDER (openai or anthropic); see senctl-agent config --help")
	}
	if c.Model == "" {
		return nil, llm.Options{}, fmt.Errorf("no model configured: set --model or SENCTL_AGENT_MODEL")
	}
	cfg := llm.Config{Provider: c.Provider, APIKey: c.APIKey, BaseURL: c.BaseURL, SendEffort: c.SendEffort}
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
