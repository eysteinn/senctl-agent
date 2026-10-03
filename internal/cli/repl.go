package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/ergochat/readline"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
	"github.com/eysteinn/senctl-agent/tools"
)

// Modes for the shell and edit tools.
const (
	modeOff  = "off"
	modeAsk  = "ask"
	modeAuto = "auto"
)

var modes = []string{modeOff, modeAsk, modeAuto}

// efforts are the reasoning effort levels /effort accepts, lowest first.
// Providers support different subsets; the request fails on one it lacks.
var efforts = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

// effortDefault clears the effort, leaving the provider's default.
const effortDefault = "default"

// style colours terminal output; zero value prints plain text.
type style struct{ on bool }

func (s style) wrap(code, text string) string {
	if !s.on {
		return text
	}
	return "\x1b[" + code + "m" + text + "\x1b[0m"
}
func (s style) dim(t string) string    { return s.wrap("2", t) }
func (s style) bold(t string) string   { return s.wrap("1", t) }
func (s style) red(t string) string    { return s.wrap("31", t) }
func (s style) green(t string) string  { return s.wrap("32", t) }
func (s style) yellow(t string) string { return s.wrap("33", t) }
func (s style) accent(t string) string { return s.wrap("35", t) }

type turnRecord struct {
	role, text string
}

// console is the interactive session behind `senctl-agent` and `chat`.
type console struct {
	env     *env
	rl      *readline.Instance
	out     io.Writer
	st      style
	session *agent.Session
	opts    llm.Options
	shell   string
	edit    string
	// allowed remembers "always" answers for this session.
	allowShell, allowEdit bool
	history               []turnRecord
	// streamed is set once text has been printed for the current reply.
	streamed bool
}

var commandHelp = [][2]string{
	{"/help", "show this help"},
	{"/model [id]", "list the models, or switch to one; the conversation continues"},
	{"/models", "list the provider's models"},
	{"/effort [level]", "list the reasoning effort levels, or set one (" + effortDefault + " for the provider's)"},
	{"/shell [off|ask|auto]", "show or set shell access"},
	{"/edit [off|ask|auto]", "show or set file editing"},
	{"/tools", "list the tools the model can use"},
	{"/usage", "tokens used in this session"},
	{"/clear", "start a new conversation (keeps settings)"},
	{"/system", "show the system prompt"},
	{"/save [file]", "save the conversation as Markdown"},
	{"/exit", "quit (also Ctrl+D)"},
}

func (c *console) completer() readline.AutoCompleter {
	modeItems := func() []*readline.PrefixCompleter {
		var out []*readline.PrefixCompleter
		for _, m := range modes {
			out = append(out, readline.PcItem(m))
		}
		return out
	}
	var effortItems []*readline.PrefixCompleter
	for _, e := range append([]string{effortDefault}, efforts...) {
		effortItems = append(effortItems, readline.PcItem(e))
	}
	return readline.NewPrefixCompleter(
		readline.PcItem("/help"),
		readline.PcItem("/model", readline.PcItemDynamic(func(string) []string {
			ids, _ := c.models(context.Background())
			return ids
		})),
		readline.PcItem("/models"),
		readline.PcItem("/effort", effortItems...),
		readline.PcItem("/shell", modeItems()...),
		readline.PcItem("/edit", modeItems()...),
		readline.PcItem("/tools"),
		readline.PcItem("/usage"),
		readline.PcItem("/clear"),
		readline.PcItem("/system"),
		readline.PcItem("/save"),
		readline.PcItem("/exit"),
		readline.PcItem("/quit"),
	)
}

// errNoListing: the provider cannot list its models.
var errNoListing = errors.New("this provider does not list its models")

// models lists the provider's models.
func (c *console) models(ctx context.Context) ([]string, error) {
	l, ok := c.env.provider.(llm.ModelLister)
	if !ok {
		return nil, errNoListing
	}
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	return l.ListModels(ctx)
}

// listModels prints the provider's models with the current one marked, or
// why they cannot be listed.
func (c *console) listModels(ctx context.Context) {
	ids, err := c.models(ctx)
	switch {
	case err != nil:
		msg, _, _ := explain(err)
		c.printf("%s\n", c.st.dim("Cannot list models: "+msg))
	case len(ids) == 0:
		c.printf("%s\n", c.st.dim("The provider lists no models."))
	default:
		c.printModels(ids)
	}
}

// switchModel switches to model when the provider lists it. When the
// models cannot be listed it switches anyway: the next message tells.
func (c *console) switchModel(ctx context.Context, model string) {
	if model == c.opts.Model {
		c.printf("Already using %s.\n", c.st.bold(model))
		return
	}
	ids, err := c.models(ctx)
	switch {
	case err == nil && !contains(ids, model):
		c.printf("%s\n", c.st.red("Unknown model "+model+"; the provider offers:"))
		c.printModels(ids)
		return
	case err != nil:
		msg, _, _ := explain(err)
		c.printf("%s\n", c.st.dim("Cannot check the model ("+msg+"); switching anyway."))
	}
	c.opts.Model = model
	c.session.SetOptions(c.opts)
	c.printf("Switched to %s. The conversation continues.\n", c.st.bold(model))
}

// setEffort lists the effort levels, or sets one.
func (c *console) setEffort(level string) {
	cur := orDefault(c.opts.Effort, effortDefault)
	if level == "" {
		c.printf("Effort: %s\n", c.st.bold(cur))
		c.printList(append([]string{effortDefault}, efforts...), cur)
		return
	}
	level = strings.ToLower(level)
	if level != effortDefault && !contains(efforts, level) {
		c.printf("%s\n", c.st.red("Unknown effort "+level+"; use "+effortDefault+", "+strings.Join(efforts, ", ")+"."))
		return
	}
	if level == effortDefault {
		c.opts.Effort = ""
	} else {
		c.opts.Effort = level
	}
	c.session.SetOptions(c.opts)
	c.printf("Effort set to %s.\n", level)
}

func (c *console) printf(format string, a ...any) { fmt.Fprintf(c.out, format, a...) }

// ask reads one answer to a yes / no / always question.
func (c *console) ask(prompt string) (yes, always bool) {
	line, err := c.rl.ReadLineWithConfig(&readline.Config{Prompt: prompt})
	if err != nil {
		return false, false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true, false
	case "a", "always":
		return true, true
	}
	return false, false
}

func (c *console) approveShell(command string) bool {
	if c.shell == modeAuto || c.allowShell {
		c.printf("%s\n", c.st.dim("  $ "+command))
		return true
	}
	c.printf("\n%s\n  %s\n", c.st.yellow("Run shell command?"), c.st.bold(command))
	yes, always := c.ask("[y]es / [n]o / [a]lways this session: ")
	if always {
		c.allowShell = true
	}
	return yes
}

func (c *console) approveEdit(ch tools.Change) bool {
	verb := "Edit"
	if ch.Tool == "write_file" {
		verb = map[bool]string{true: "Create", false: "Overwrite"}[ch.New]
	}
	var b strings.Builder
	for _, l := range strings.Split(strings.TrimSuffix(ch.Preview, "\n"), "\n") {
		switch {
		case strings.HasPrefix(l, "- "):
			b.WriteString("  " + c.st.red(l) + "\n")
		case strings.HasPrefix(l, "+ "):
			b.WriteString("  " + c.st.green(l) + "\n")
		default:
			b.WriteString("  " + c.st.dim(l) + "\n")
		}
	}
	if c.edit == modeAuto || c.allowEdit {
		c.printf("%s %s\n%s", c.st.dim("  "+verb), c.st.dim(ch.Path), b.String())
		return true
	}
	c.printf("\n%s %s\n%s", c.st.yellow(verb+" file?"), c.st.bold(ch.Path), b.String())
	yes, always := c.ask("[y]es / [n]o / [a]lways this session: ")
	if always {
		c.allowEdit = true
	}
	return yes
}

// tools builds the tool set for the current modes.
func (c *console) tools() []agent.Tool {
	ts := c.env.ws.Tools()
	if c.edit != modeOff {
		ts = append(ts, c.env.ws.EditTools(c.approveEdit)...)
	}
	if c.shell != modeOff {
		ts = append(ts, tools.Shell(c.env.ws.Root(), tools.DefaultShellTimeout, c.approveShell))
	}
	return ts
}

func (c *console) recorder() agent.Recorder {
	return func(_ context.Context, e agent.Event) {
		switch e.Kind {
		case agent.EventToolCall:
			c.endStream()
			name, args, _ := strings.Cut(e.Content, " ")
			c.printf("%s %s %s\n", c.st.accent("⏺"), c.st.bold(name), c.st.dim(oneLine(args, 120)))
		case agent.EventToolError:
			c.printf("  %s\n", c.st.red("✗ "+oneLine(e.Content, 200)))
		}
	}
}

func (c *console) endStream() {
	if c.streamed {
		c.printf("\n")
		c.streamed = false
	}
}

func (c *console) newSession() {
	conv := c.env.provider.NewConversation(c.opts, c.env.system, nil)
	ts := c.tools()
	c.session = agent.NewSession(conv, ts, c.env.agentConfig(), c.recorder())
	c.session.SetTools(ts)
	c.session.Stream(func(d string) {
		if !c.streamed {
			c.streamed = true
		}
		c.printf("%s", d)
	})
	c.history = nil
}

func (c *console) retool() {
	c.session.SetTools(c.tools())
}

func (c *console) banner() {
	c.printf("%s %s\n", c.st.accent("✻"), c.st.bold("senctl-agent "+Version))
	c.printf("%s\n", c.st.dim(fmt.Sprintf("  %s · %s · effort %s · workspace %s", c.env.provider.Name(), c.opts.Model,
		orDefault(c.opts.Effort, "default"), c.env.ws.Root())))
	c.printf("%s\n\n", c.st.dim(fmt.Sprintf("  edit %s · shell %s · /help for commands · Ctrl+C cancels a reply", c.edit, c.shell)))
}

func orDefault(s, d string) string {
	if s == "" {
		return d
	}
	return s
}

// command runs a slash command; it reports false when the console should exit.
func (c *console) command(ctx context.Context, line string) bool {
	fields := strings.Fields(line)
	name, arg := fields[0], strings.TrimSpace(strings.TrimPrefix(line, fields[0]))
	switch name {
	case "/exit", "/quit":
		return false
	case "/help":
		for _, h := range commandHelp {
			c.printf("  %-24s %s\n", c.st.bold(h[0]), c.st.dim(h[1]))
		}
		c.printf("  %s\n", c.st.dim("End a line with \\ to continue on the next line."))
	case "/model":
		if arg == "" {
			c.printf("Model: %s\n", c.st.bold(c.opts.Model))
			c.listModels(ctx)
			return true
		}
		c.switchModel(ctx, arg)
	case "/models":
		c.listModels(ctx)
	case "/effort":
		c.setEffort(arg)
	case "/shell", "/edit":
		cur := &c.shell
		if name == "/edit" {
			cur = &c.edit
		}
		if arg == "" {
			c.printf("%s: %s\n", strings.TrimPrefix(name, "/"), *cur)
			return true
		}
		if !contains(modes, arg) {
			c.printf("%s\n", c.st.red("Use off, ask or auto."))
			return true
		}
		if arg == modeAsk && name == "/shell" {
			c.allowShell = false
		} else if arg == modeAsk {
			c.allowEdit = false
		}
		*cur = arg
		c.retool()
		c.printf("%s is now %s.\n", strings.TrimPrefix(name, "/"), arg)
	case "/tools":
		for _, t := range c.tools() {
			c.printf("  %-12s %s\n", c.st.bold(t.Spec.Name), c.st.dim(oneLine(t.Spec.Description, 90)))
		}
		c.printf("  %s\n", c.st.dim(fmt.Sprintf("edit %s · shell %s", c.edit, c.shell)))
	case "/usage":
		u := c.session.Usage()
		c.printf("input tokens %d, output tokens %d\n", u.InputTokens, u.OutputTokens)
	case "/clear":
		c.newSession()
		c.printf("Started a new conversation.\n")
	case "/system":
		c.printf("%s\n", c.env.system)
	case "/save":
		path := arg
		if path == "" {
			path = "senctl-agent-" + time.Now().Format("20060102-150405") + ".md"
		}
		if err := c.save(path); err != nil {
			c.printf("%s\n", c.st.red(err.Error()))
		} else {
			c.printf("Saved to %s.\n", path)
		}
	default:
		c.printf("%s\n", c.st.red("Unknown command "+name+"; try /help."))
	}
	return true
}

// printModels prints the models sorted, marking the current one.
func (c *console) printModels(ids []string) {
	sort.Strings(ids)
	c.printList(ids, c.opts.Model)
}

// printList prints items in order, marking cur.
func (c *console) printList(items []string, cur string) {
	for _, id := range items {
		mark := "  "
		if id == cur {
			mark = c.st.accent("● ")
		}
		c.printf("  %s%s\n", mark, id)
	}
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func (c *console) save(path string) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# senctl-agent conversation\n\n%s · %s · %s\n\n", c.env.provider.Name(), c.opts.Model, time.Now().Format(time.RFC3339))
	for _, t := range c.history {
		fmt.Fprintf(&b, "## %s\n\n%s\n\n", t.role, strings.TrimSpace(t.text))
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// ask sends one message, letting Ctrl+C cancel the reply.
func (c *console) send(ctx context.Context, text string) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	defer signal.Stop(sig)
	go func() {
		select {
		case <-sig:
			cancel()
		case <-ctx.Done():
		}
	}()

	c.history = append(c.history, turnRecord{"You", text})
	c.streamed = false
	answer, err := c.session.Send(ctx, c.env.attachMentions(ctx, text))
	streamed := c.streamed
	c.endStream()
	switch {
	case errors.Is(err, context.Canceled):
		c.printf("%s\n", c.st.dim("(cancelled)"))
		return
	case err != nil:
		msg, hint, _ := explain(err)
		c.printf("%s\n", c.st.red("error: "+msg))
		if hint != "" {
			c.printf("%s\n", c.st.dim(hint))
		}
		return
	}
	if !streamed && strings.TrimSpace(answer) != "" {
		c.printf("%s\n", strings.TrimSpace(answer))
	}
	c.history = append(c.history, turnRecord{"Assistant", answer})
}

// run is the read-eval-print loop. initial, when set, is sent first.
func (c *console) run(ctx context.Context, initial string) error {
	c.banner()
	if initial != "" {
		c.printf("%s %s\n", c.st.accent("›"), initial)
		c.send(ctx, initial)
	}
	var pending []string
	interrupts := 0
	for {
		prompt := c.st.accent("› ")
		if len(pending) > 0 {
			prompt = c.st.dim("… ")
		}
		c.rl.SetPrompt(prompt)
		line, err := c.rl.Readline()
		if errors.Is(err, readline.ErrInterrupt) {
			if len(pending) > 0 || line != "" {
				pending, interrupts = nil, 0
				continue
			}
			interrupts++
			if interrupts >= 2 {
				return nil
			}
			c.printf("%s\n", c.st.dim("(press Ctrl+C again or type /exit to quit)"))
			continue
		}
		if err == io.EOF {
			c.printf("\n")
			return nil
		}
		if err != nil {
			return err
		}
		interrupts = 0
		if strings.HasSuffix(line, "\\") {
			pending = append(pending, strings.TrimSuffix(line, "\\"))
			continue
		}
		text := strings.TrimSpace(strings.Join(append(pending, line), "\n"))
		pending = nil
		if text == "" {
			continue
		}
		if strings.HasPrefix(text, "/") {
			if !c.command(ctx, text) {
				return nil
			}
			continue
		}
		c.send(ctx, text)
		c.printf("\n")
	}
}

// historyFile is the console history under stateHome ($XDG_STATE_HOME,
// default ~/.local/state).
func historyFile(stateHome string) string {
	dir := stateHome
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		dir = filepath.Join(home, ".local", "state")
	}
	dir = filepath.Join(dir, "senctl-agent")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return ""
	}
	return filepath.Join(dir, "history")
}
