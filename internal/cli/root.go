// Package cli is the senctl-agent command-line interface.
package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
	"github.com/eysteinn/senctl-agent/tools"
)

// Version is set at build time with -ldflags "-X github.com/eysteinn/senctl-agent/internal/cli.Version=…".
var Version = "dev"

const defaultSystem = `You are a careful assistant working in a directory on the user's machine (%s).
You can read its files with the tools provided%s. Look things up instead of guessing, quote the files you rely on, and answer concisely. When you are unsure, say so.`

// NewRootCmd builds the senctl-agent command tree.
func NewRootCmd() *cobra.Command {
	v := viper.New()
	root := &cobra.Command{
		Use:   "senctl-agent",
		Short: "A small tool-using LLM agent for the terminal",
		Long: `senctl-agent talks to an LLM (any OpenAI-compatible endpoint such as an LLM
proxy, or the Anthropic API) and lets it read files in a workspace directory
through tools, optionally also run shell commands.

Configuration comes from flags, then environment variables (SENCTL_AGENT_PROVIDER,
SENCTL_AGENT_BASE_URL, SENCTL_AGENT_API_KEY, SENCTL_AGENT_MODEL, …), then a config
file ($XDG_CONFIG_HOME/senctl-agent/config.yaml or ./.senctl-agent.yaml) with the
same keys in snake_case. The API key falls back to OPENAI_API_KEY or
ANTHROPIC_API_KEY for the matching provider.`,
		SilenceUsage: true,
	}
	bindFlags(root, v)
	root.AddCommand(newRunCmd(v), newChatCmd(v), newConfigCmd(v), &cobra.Command{
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
	tools    []agent.Tool
}

func setup(cmd *cobra.Command, v *viper.Viper) (*env, error) {
	cfg, err := load(cmd, v)
	if err != nil {
		return nil, err
	}
	p, opts, err := cfg.provider()
	if err != nil {
		return nil, err
	}
	ws, err := tools.NewWorkspace(cfg.Dir)
	if err != nil {
		return nil, fmt.Errorf("workspace: %w", err)
	}
	e := &env{cfg: cfg, provider: p, opts: opts, tools: ws.Tools()}
	shellNote := ""
	switch cfg.Shell {
	case "auto":
		e.tools = append(e.tools, tools.Shell(ws.Root(), 2*time.Minute, nil))
		shellNote = " and run shell commands there"
	case "ask":
		tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
		if err != nil {
			return nil, fmt.Errorf("--shell=ask needs a terminal to confirm commands: %w", err)
		}
		e.tools = append(e.tools, tools.Shell(ws.Root(), 2*time.Minute, func(command string) bool {
			fmt.Fprintf(tty, "\nRun shell command in %s?\n  %s\n[y/N] ", ws.Root(), command)
			line, _ := bufio.NewReader(tty).ReadString('\n')
			a := strings.ToLower(strings.TrimSpace(line))
			return a == "y" || a == "yes"
		}))
		shellNote = " and run shell commands there (the user confirms each one)"
	}
	e.system = cfg.System
	if e.system == "" {
		e.system = fmt.Sprintf(defaultSystem, ws.Root(), shellNote)
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

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

func newRunCmd(v *viper.Viper) *cobra.Command {
	var jsonOut, verbose bool
	cmd := &cobra.Command{
		Use:   "run [prompt]",
		Short: "Answer one prompt and exit",
		Long: `Answer one prompt and exit. Piped standard input is appended to the prompt,
so you can run e.g.:

  git diff | senctl-agent run "review this change"`,
		RunE: func(cmd *cobra.Command, args []string) error {
			prompt := strings.TrimSpace(strings.Join(args, " "))
			if in, ok := cmd.InOrStdin().(*os.File); ok && !isTerminal(in) {
				data, err := io.ReadAll(io.LimitReader(in, 4<<20))
				if err != nil {
					return err
				}
				if s := strings.TrimSpace(string(data)); s != "" {
					prompt = strings.TrimSpace(prompt + "\n\n<stdin>\n" + s + "\n</stdin>")
				}
			}
			if prompt == "" {
				return fmt.Errorf("give a prompt as an argument or on standard input")
			}
			e, err := setup(cmd, v)
			if err != nil {
				return err
			}
			conv := e.provider.NewConversation(e.opts, e.system, agent.Specs(e.tools...))
			s := agent.NewSession(conv, e.tools, agent.Config{MaxTurns: e.cfg.MaxTurns}, recorder(cmd.ErrOrStderr(), verbose))
			answer, err := s.Send(cmd.Context(), prompt)
			if err != nil {
				return err
			}
			if jsonOut {
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetEscapeHTML(false)
				return enc.Encode(map[string]any{"text": answer, "model": e.opts.Model, "provider": e.provider.Name(),
					"usage": map[string]int64{"input_tokens": s.Usage().InputTokens, "output_tokens": s.Usage().OutputTokens}})
			}
			fmt.Fprintln(cmd.OutOrStdout(), strings.TrimSpace(answer))
			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "print the answer and token usage as JSON")
	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "print tool calls to stderr")
	return cmd
}

func newChatCmd(v *viper.Viper) *cobra.Command {
	var verbose bool
	cmd := &cobra.Command{
		Use:   "chat",
		Short: "Start an interactive session (/exit to quit, /usage, /reset)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			e, err := setup(cmd, v)
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			newSession := func() *agent.Session {
				conv := e.provider.NewConversation(e.opts, e.system, agent.Specs(e.tools...))
				return agent.NewSession(conv, e.tools, agent.Config{MaxTurns: e.cfg.MaxTurns}, recorder(cmd.ErrOrStderr(), verbose))
			}
			s := newSession()
			fmt.Fprintf(out, "senctl-agent %s · %s %s · workspace %s\nType /exit to quit.\n", Version, e.provider.Name(), e.opts.Model, e.cfg.Dir)
			sc := bufio.NewScanner(cmd.InOrStdin())
			sc.Buffer(make([]byte, 64<<10), 1<<20)
			for {
				fmt.Fprint(out, "\n> ")
				if !sc.Scan() {
					fmt.Fprintln(out)
					return sc.Err()
				}
				line := strings.TrimSpace(sc.Text())
				switch line {
				case "":
					continue
				case "/exit", "/quit":
					return nil
				case "/usage":
					u := s.Usage()
					fmt.Fprintf(out, "input tokens %d, output tokens %d\n", u.InputTokens, u.OutputTokens)
					continue
				case "/reset":
					s = newSession()
					fmt.Fprintln(out, "Started a new conversation.")
					continue
				}
				answer, err := s.Send(cmd.Context(), line)
				if err != nil {
					fmt.Fprintf(cmd.ErrOrStderr(), "error: %v\n", err)
					continue
				}
				fmt.Fprintln(out, strings.TrimSpace(answer))
			}
		},
	}
	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "print tool calls to stderr")
	return cmd
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
			fmt.Fprintf(out, "config file: %s\nprovider:    %s\nbase_url:    %s\nmodel:       %s\napi_key:     %s\neffort:      %s\nmax_turns:   %d\nmax_tokens:  %d\ndir:         %s\nshell:       %s\n",
				file, c.Provider, c.BaseURL, c.Model, mask(c.APIKey), c.Effort, c.MaxTurns, c.MaxTokens, c.Dir, c.Shell)
			return nil
		},
	}
}
