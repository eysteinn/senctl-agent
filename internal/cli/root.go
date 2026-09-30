// Package cli is the senctl-agent command-line interface.
package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ergochat/readline"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
	"github.com/eysteinn/senctl-agent/tools"
)

// Version is set at build time with -ldflags "-X github.com/eysteinn/senctl-agent/internal/cli.Version=…".
var Version = "dev"

const defaultSystem = `You are a careful software assistant working in a directory on the user's machine (%s).
Use the tools provided to look things up instead of guessing; the available tools can change during the conversation. Files can be far larger than you can read at once: for logs and other big files, check the size with file_info, find what matters with grep or pipeline, then read around it. Output too large to show whole is saved to a file you can search the same way. Before changing a file, read it. Keep changes minimal and explain what you changed. Quote the files you rely on, answer concisely, and say when you are unsure.`

// NewRootCmd builds the senctl-agent command tree.
func NewRootCmd() *cobra.Command {
	v := viper.New()
	root := &cobra.Command{
		Use:   "senctl-agent [prompt]",
		Short: "A tool-using LLM agent for the terminal",
		Long: `senctl-agent talks to an LLM through an LLM proxy (or any endpoint that speaks
the OpenAI Responses API). Point it at the proxy with an API key;
the model defaults to the proxy's only model, if it lists one. Run it without
arguments for an interactive console (/help lists its commands), or with a
prompt to start the console with that message. Use "senctl-agent run" for
one-shot, scriptable answers.

The model can read files in the workspace (--dir), edit them (--edit, asks
first by default in the console) and optionally run shell commands (--shell).

Configuration comes from flags, then environment variables (SENCTL_AGENT_BASE_URL,
SENCTL_AGENT_API_KEY, SENCTL_AGENT_MODEL, …), then a config file
($XDG_CONFIG_HOME/senctl-agent/config.yaml or ./.senctl-agent.yaml) with the same
keys in snake_case. The API key falls back to OPENAI_API_KEY. To call the
Anthropic API directly instead of a proxy, set provider to anthropic.`,
		Args:         cobra.ArbitraryArgs,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runConsole(cmd, v, strings.TrimSpace(strings.Join(args, " ")))
		},
	}
	bindFlags(root, v)
	root.AddCommand(newRunCmd(v), newChatCmd(v), newConfigCmd(v), newModelsCmd(v), &cobra.Command{
		Use:   "version",
		Short: "Print the version",
		Run:   func(cmd *cobra.Command, _ []string) { fmt.Fprintln(cmd.OutOrStdout(), Version) },
	})
	return root
}

// env is what a command needs to talk to the model.
type env struct {
	cfg      *Config
	provider llm.Provider
	opts     llm.Options
	system   string
	ws       *tools.Workspace
	// cache holds tool output and input too large for the model's context;
	// the file tools can read it. close removes it.
	cache *tools.Cache
}

func (e *env) close() { _ = e.cache.Close() }

// agentConfig is the agent configuration for a session.
func (e *env) agentConfig() agent.Config {
	return agent.Config{MaxTurns: e.cfg.MaxTurns, Spill: e.cache.Spill}
}

// newCache makes this session's cache directory under the user cache
// directory, clearing out ones older sessions left behind.
func newCache() (*tools.Cache, error) {
	if base, err := os.UserCacheDir(); err == nil {
		dir := filepath.Join(base, "senctl-agent", "sessions")
		tools.PruneCaches(dir, 24*time.Hour)
		if c, err := tools.NewCache(dir); err == nil {
			return c, nil
		}
	}
	return tools.NewCache("")
}

func setup(cmd *cobra.Command, v *viper.Viper) (*env, error) {
	cfg, err := load(cmd, v)
	if err != nil {
		return nil, err
	}
	p, opts, err := cfg.provider(cmd.Context())
	if err != nil {
		return nil, err
	}
	ws, err := tools.NewWorkspace(cfg.Dir)
	if err != nil {
		return nil, fmt.Errorf("workspace: %w", err)
	}
	cache, err := newCache()
	if err != nil {
		return nil, fmt.Errorf("cache: %w", err)
	}
	if err := ws.AllowRead(cache.Dir()); err != nil {
		cache.Close()
		return nil, fmt.Errorf("cache: %w", err)
	}
	e := &env{cfg: cfg, provider: p, opts: opts, ws: ws, system: cfg.System, cache: cache}
	if e.system == "" {
		e.system = fmt.Sprintf(defaultSystem, ws.Root())
	}
	return e, nil
}

// recorder prints tool activity to w when verbose.
func recorder(w io.Writer, verbose bool) agent.Recorder {
	if !verbose {
		return nil
	}
	return func(_ context.Context, e agent.Event) {
		switch e.Kind {
		case agent.EventToolCall, agent.EventToolError:
			fmt.Fprintf(w, "· %s: %s\n", e.Kind, oneLine(e.Content, 200))
		case agent.EventModelText:
			fmt.Fprintf(w, "· %s\n", oneLine(e.Content, 200))
		}
	}
}

func oneLine(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

func isTerminal(f any) bool {
	file, ok := f.(*os.File)
	if !ok {
		return false
	}
	st, err := file.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

// ttyApprove asks on the controlling terminal, for one-shot runs.
func ttyApprove() (func(question string) bool, error) {
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("\"ask\" needs a terminal to confirm actions: %w", err)
	}
	return func(question string) bool {
		fmt.Fprintf(tty, "\n%s\n[y/N] ", question)
		line, _ := bufio.NewReader(tty).ReadString('\n')
		a := strings.ToLower(strings.TrimSpace(line))
		return a == "y" || a == "yes"
	}, nil
}

// runTools is the tool set for one-shot runs.
func (e *env) runTools() ([]agent.Tool, error) {
	ts := e.ws.Tools()
	var ask func(string) bool
	needAsk := e.cfg.Shell == modeAsk || e.cfg.Edit == modeAsk
	if needAsk {
		var err error
		if ask, err = ttyApprove(); err != nil {
			return nil, err
		}
	}
	switch e.cfg.Edit {
	case modeAuto:
		ts = append(ts, e.ws.EditTools(nil)...)
	case modeAsk:
		ts = append(ts, e.ws.EditTools(func(c tools.Change) bool {
			return ask(fmt.Sprintf("%s %s?\n%s", c.Tool, c.Path, c.Preview))
		})...)
	}
	switch e.cfg.Shell {
	case modeAuto:
		ts = append(ts, tools.Shell(e.ws.Root(), tools.DefaultShellTimeout, nil))
	case modeAsk:
		ts = append(ts, tools.Shell(e.ws.Root(), tools.DefaultShellTimeout, func(command string) bool {
			return ask("Run shell command in " + e.ws.Root() + "?\n  " + command)
		}))
	}
	return ts, nil
}

func newRunCmd(v *viper.Viper) *cobra.Command {
	var jsonOut, verbose bool
	cmd := &cobra.Command{
		Use:   "run [prompt]",
		Short: "Answer one prompt and exit",
		Long: `Answer one prompt and exit. Piped standard input is appended to the prompt,
so you can run e.g.:

  git diff | senctl-agent run "review this change"

Input too large for the model's context (such as a log) is saved to a file
the model searches with its tools instead:

  journalctl -u myservice | senctl-agent run "why does it keep restarting?"

File editing and shell access are off unless --edit / --shell say otherwise.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			prompt := strings.TrimSpace(strings.Join(args, " "))
			var stdin io.Reader
			if in, ok := cmd.InOrStdin().(*os.File); ok && !isTerminal(in) {
				stdin = in
			}
			if prompt == "" && stdin == nil {
				return fmt.Errorf("give a prompt as an argument or on standard input")
			}
			e, err := setup(cmd, v)
			if err != nil {
				return err
			}
			defer e.close()
			if stdin != nil {
				if prompt, err = e.withInput(prompt, stdin); err != nil {
					return err
				}
			}
			if prompt == "" {
				return fmt.Errorf("give a prompt as an argument or on standard input")
			}
			if e.cfg.Edit == "" {
				e.cfg.Edit = modeOff
			}
			ts, err := e.runTools()
			if err != nil {
				return err
			}
			conv := e.provider.NewConversation(e.opts, e.system, agent.Specs(ts...))
			s := agent.NewSession(conv, ts, e.agentConfig(), recorder(cmd.ErrOrStderr(), verbose))
			out := cmd.OutOrStdout()
			streamed := false
			if !jsonOut && isTerminal(out) {
				s.Stream(func(d string) { streamed = true; fmt.Fprint(out, d) })
			}
			answer, err := s.Send(cmd.Context(), prompt)
			if err != nil {
				return err
			}
			if jsonOut {
				enc := json.NewEncoder(out)
				enc.SetEscapeHTML(false)
				return enc.Encode(map[string]any{"text": answer, "model": e.opts.Model, "provider": e.provider.Name(),
					"usage": map[string]int64{"input_tokens": s.Usage().InputTokens, "output_tokens": s.Usage().OutputTokens}})
			}
			if streamed {
				fmt.Fprintln(out)
				return nil
			}
			fmt.Fprintln(out, strings.TrimSpace(answer))
			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "print the answer and token usage as JSON")
	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "print tool calls to stderr")
	return cmd
}

func newChatCmd(v *viper.Viper) *cobra.Command {
	return &cobra.Command{
		Use:   "chat [prompt]",
		Short: "Start the interactive console (same as running senctl-agent without a command)",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runConsole(cmd, v, strings.TrimSpace(strings.Join(args, " ")))
		},
	}
}

func runConsole(cmd *cobra.Command, v *viper.Viper, initial string) error {
	e, err := setup(cmd, v)
	if err != nil {
		return err
	}
	defer e.close()
	c := &console{env: e, opts: e.opts, shell: orDefault(e.cfg.Shell, modeOff), edit: orDefault(e.cfg.Edit, modeAsk)}
	out := cmd.OutOrStdout()
	interactive := isTerminal(cmd.InOrStdin()) && isTerminal(out)
	c.st = style{on: isTerminal(out) && v.GetString(keyNoColor) == ""}
	rlCfg := &readline.Config{
		Prompt:          "› ",
		Stdin:           cmd.InOrStdin(),
		Stdout:          out,
		Stderr:          cmd.ErrOrStderr(),
		AutoComplete:    c.completer(),
		InterruptPrompt: "^C",
		EOFPrompt:       "",
	}
	if interactive {
		rlCfg.HistoryFile = historyFile(v.GetString(keyStateHome))
	} else {
		rlCfg.FuncIsTerminal = func() bool { return false }
	}
	rl, err := readline.NewFromConfig(rlCfg)
	if err != nil {
		return err
	}
	defer rl.Close()
	c.rl = rl
	c.out = out
	c.newSession()
	return c.run(cmd.Context(), initial)
}

func newModelsCmd(v *viper.Viper) *cobra.Command {
	return &cobra.Command{
		Use:   "models",
		Short: "List the models the endpoint serves",
		RunE: func(cmd *cobra.Command, _ []string) error {
			c, err := load(cmd, v)
			if err != nil {
				return err
			}
			c.Model = "-" // listing needs no model
			p, _, err := c.provider(cmd.Context())
			if err != nil {
				return err
			}
			ml, ok := p.(llm.ModelLister)
			if !ok {
				return fmt.Errorf("%s cannot list models", p.Name())
			}
			ids, err := ml.ListModels(cmd.Context())
			if err != nil {
				return err
			}
			for _, id := range ids {
				fmt.Fprintln(cmd.OutOrStdout(), id)
			}
			return nil
		},
	}
}

func newConfigCmd(v *viper.Viper) *cobra.Command {
	return &cobra.Command{
		Use:   "config",
		Short: "Show the effective configuration (API key masked)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			c, err := load(cmd, v)
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			file := v.ConfigFileUsed()
			if file == "" {
				file = "(none)"
			}
			fmt.Fprintf(out, "config file: %s\nprovider:    %s\nbase_url:    %s\nmodel:       %s\napi_key:     %s\neffort:      %s\nmax_turns:   %d\nmax_tokens:  %d\ndir:         %s\nshell:       %s\nedit:        %s\n",
				file, orDefault(c.Provider, "openai (default)"), c.BaseURL, c.Model, mask(c.APIKey), c.Effort, c.MaxTurns, c.MaxTokens, c.Dir, c.Shell, orDefault(c.Edit, "(ask in the console, off for run)"))
			return nil
		},
	}
}
