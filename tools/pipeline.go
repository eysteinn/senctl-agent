package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
)

const (
	pipelineTimeout   = time.Minute
	maxPipelineOutput = 16 << 20
	maxPipelineStderr = 4000
)

// cmdSpec says how a pipeline command reads its arguments, so that no
// argument can name a file: the input always arrives on stdin, and options
// that read or write files are refused.
type cmdSpec struct {
	short   string          // short options that take a value
	long    map[string]bool // long options that take a separate value
	long2   map[string]bool // long options that take two values
	banned  string          // refused short options
	bannedL map[string]bool // refused long options
	args    int             // positional arguments allowed
	pattern byte            // short option that supplies grep's pattern
}

func set(names ...string) map[string]bool {
	m := map[string]bool{}
	for _, n := range names {
		m[n] = true
	}
	return m
}

// pipelineCommands are the commands pipeline may run.
var pipelineCommands = map[string]cmdSpec{
	"grep": {short: "eABCmdD", long: set("regexp", "max-count", "after-context", "before-context", "context", "label", "binary-files", "devices", "directories"),
		banned: "f", bannedL: set("file", "exclude-from", "include", "exclude", "exclude-dir"), args: 1, pattern: 'e'},
	"head": {short: "nc", long: set("lines", "bytes")},
	"tail": {short: "nc", long: set("lines", "bytes"), banned: "fFs", bannedL: set("follow", "pid", "retry", "sleep-interval")},
	"wc":   {bannedL: set("files0-from")},
	"sort": {short: "ktS", long: set("key", "field-separator", "buffer-size", "parallel", "batch-size", "sort"),
		banned: "oT", bannedL: set("output", "temporary-directory", "compress-program", "files0-from", "random-source")},
	"uniq": {short: "fsw", long: set("skip-fields", "skip-chars", "check-chars")},
	"cut":  {short: "bcdf", long: set("bytes", "characters", "delimiter", "fields", "output-delimiter")},
	"tr":   {args: 2},
	"nl":   {short: "bdfhilnsvw", long: set("body-numbering", "section-delimiter", "footer-numbering", "header-numbering", "line-increment", "join-blank-lines", "number-format", "number-separator", "starting-line-number", "number-width")},
	"tac":  {short: "s", long: set("separator")},
	"jq": {long: set("indent"), long2: set("arg", "argjson"), banned: "fL",
		bannedL: set("from-file", "library-path", "rawfile", "slurpfile", "args", "jsonargs"), args: 1},
}

// pipelineOrder is the order commands are listed to the model.
var pipelineOrder = []string{"grep", "head", "tail", "wc", "sort", "uniq", "cut", "tr", "nl", "tac", "jq"}

var errUnsupported = "pipeline runs simple commands joined by |, with sh-style quoting but without redirection, variables, globbing, command substitution or ;/&&"

// splitPipeline splits a command line into stages of words.
func splitPipeline(s string) ([][]string, error) {
	var stages [][]string
	var words []string
	var cur strings.Builder
	inWord := false
	flush := func() {
		if inWord {
			words = append(words, cur.String())
			cur.Reset()
			inWord = false
		}
	}
	endStage := func() error {
		flush()
		if len(words) == 0 {
			return errors.New("empty command in pipeline")
		}
		stages = append(stages, words)
		words = nil
		return nil
	}
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '\'':
			j := strings.IndexByte(s[i+1:], '\'')
			if j < 0 {
				return nil, errors.New("unterminated ' quote")
			}
			cur.WriteString(s[i+1 : i+1+j])
			i += j + 1
			inWord = true
		case c == '"':
			i++
			for ; i < len(s) && s[i] != '"'; i++ {
				switch {
				case s[i] == '\\' && i+1 < len(s) && strings.IndexByte("\"\\$`", s[i+1]) >= 0:
					i++
				case s[i] == '$' || s[i] == '`':
					return nil, fmt.Errorf("%q is not supported: %s", s[i], errUnsupported)
				}
				cur.WriteByte(s[i])
			}
			if i >= len(s) {
				return nil, errors.New("unterminated \" quote")
			}
			inWord = true
		case c == '\\':
			if i+1 < len(s) {
				i++
				cur.WriteByte(s[i])
				inWord = true
			}
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			flush()
		case c == '|':
			if err := endStage(); err != nil {
				return nil, err
			}
		case strings.IndexByte(";&<>$`(){}*?[", c) >= 0:
			return nil, fmt.Errorf("%q is not supported (quote it if it is part of a pattern): %s", c, errUnsupported)
		default:
			cur.WriteByte(c)
			inWord = true
		}
	}
	if err := endStage(); err != nil {
		return nil, err
	}
	return stages, nil
}

// refuses reports whether a long option is refused. GNU tools accept any
// unambiguous prefix of a long option (--compress for --compress-program),
// so prefixes of refused options are refused too, unless they name an
// allowed option exactly.
func (spec cmdSpec) refuses(opt string) bool {
	if spec.long[opt] || spec.long2[opt] {
		return false
	}
	for banned := range spec.bannedL {
		if strings.HasPrefix(banned, opt) {
			return true
		}
	}
	return false
}

// checkArgs refuses arguments that could name files or change them.
func checkArgs(name string, spec cmdSpec, args []string) error {
	positional, allowed := 0, spec.args
	onlyPositional := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case onlyPositional || a == "-" || !strings.HasPrefix(a, "-"):
			positional++
		case a == "--":
			onlyPositional = true
		case strings.HasPrefix(a, "--"):
			opt, _, hasValue := strings.Cut(a[2:], "=")
			if spec.refuses(opt) {
				return fmt.Errorf("%s --%s is not allowed in pipeline", name, opt)
			}
			switch {
			case spec.long2[opt]:
				i += 2
			case spec.long[opt] && !hasValue:
				i++
			}
			if opt == "regexp" && spec.pattern != 0 {
				allowed = 0
			}
		default:
			for j := 1; j < len(a); j++ {
				c := a[j]
				if strings.IndexByte(spec.banned, c) >= 0 {
					return fmt.Errorf("%s -%c is not allowed in pipeline", name, c)
				}
				if c == spec.pattern {
					allowed = 0
				}
				if strings.IndexByte(spec.short, c) >= 0 {
					if j == len(a)-1 {
						i++
					}
					break
				}
			}
		}
	}
	if positional > allowed {
		return fmt.Errorf("%s: too many arguments. The input file arrives on standard input, so do not name files; options that take a value must be written as documented (e.g. -n 5 or --lines=5)", name)
	}
	return nil
}

// cappedWriter keeps the first max bytes written to it. It is safe for
// concurrent use.
type cappedWriter struct {
	mu   sync.Mutex
	b    bytes.Buffer
	max  int
	lost int
}

func (w *cappedWriter) Write(p []byte) (int, error) {
	n := len(p)
	w.mu.Lock()
	defer w.mu.Unlock()
	room := w.max - w.b.Len()
	if room < len(p) {
		w.lost += len(p) - max(room, 0)
		p = p[:max(room, 0)]
	}
	w.b.Write(p)
	return n, nil
}

// pipelineEnv is a minimal environment: no credentials reach the commands.
func pipelineEnv(home string) []string {
	env := []string{"HOME=" + home}
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if k == "PATH" || k == "LANG" || k == "TZ" || k == "TMPDIR" || strings.HasPrefix(k, "LC_") {
			env = append(env, kv)
		}
	}
	return env
}

// pipelineTool returns the pipeline tool, or false when none of its
// commands are installed.
func (w *Workspace) pipelineTool() (agent.Tool, bool) {
	var have []string
	for _, name := range pipelineOrder {
		if _, err := exec.LookPath(name); err == nil {
			have = append(have, name)
		}
	}
	if len(have) == 0 {
		return agent.Tool{}, false
	}
	return agent.Tool{
		Spec: llm.ToolSpec{Name: "pipeline", Description: "Run a read-only text pipeline over one file, which is fed to the first command on standard input. " +
			"Use it to summarise large files such as logs, e.g. grep -i error | cut -d' ' -f1 | sort | uniq -c | sort -rn | head -20. " +
			"Commands: " + strings.Join(have, ", ") + ". Stages are joined with | and quoted as in sh; there are no file arguments, redirections, variables or globbing. " +
			fmt.Sprintf(".gz input is decompressed. Times out after %s.", pipelineTimeout),
			Schema: object(map[string]any{
				"path":    prop("string", "The input file: relative to the workspace, or an absolute path given by a tool"),
				"command": prop("string", "The pipeline, e.g. grep -c ERROR"),
			}, "path", "command")},
		Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
			in, err := decode[struct{ Path, Command string }](raw)
			if err != nil {
				return "", err
			}
			return w.pipeline(ctx, in.Path, in.Command)
		},
	}, true
}

func (w *Workspace) pipeline(ctx context.Context, p, command string) (string, error) {
	stages, err := splitPipeline(command)
	if err != nil {
		return "", err
	}
	for _, st := range stages {
		spec, ok := pipelineCommands[st[0]]
		if !ok {
			return "", fmt.Errorf("%s is not available in pipeline; use one of %s", st[0], strings.Join(pipelineOrder, ", "))
		}
		if st[0] == "jq" && len(st) > 1 {
			for _, a := range st[1:] {
				if strings.Contains(a, "import") || strings.Contains(a, "include") {
					return "", errors.New("jq modules (import/include) are not allowed in pipeline")
				}
			}
		}
		if err := checkArgs(st[0], spec, st[1:]); err != nil {
			return "", err
		}
	}
	full, _, err := w.resolveFile(p)
	if err != nil {
		return "", err
	}
	input, err := openText(full)
	if err != nil {
		return "", err
	}
	defer input.Close()
	// Commands run in an empty directory, so a stray relative name finds
	// nothing.
	scratch, err := os.MkdirTemp("", "pipeline-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(scratch)

	ctx, cancel := context.WithTimeout(ctx, pipelineTimeout)
	defer cancel()
	stdout := &cappedWriter{max: maxPipelineOutput}
	stderr := &cappedWriter{max: maxPipelineStderr}
	cmds := make([]*exec.Cmd, len(stages))
	// ends are the parent's copies of the pipes between stages.
	var ends []*os.File
	defer func() {
		for _, f := range ends {
			f.Close()
		}
	}()
	var prev *os.File
	for i, st := range stages {
		cmd := exec.CommandContext(ctx, st[0], st[1:]...)
		cmd.Dir, cmd.Env, cmd.Stderr, cmd.WaitDelay = scratch, pipelineEnv(scratch), stderr, time.Second
		if i == 0 {
			cmd.Stdin = input
		} else {
			cmd.Stdin = prev
		}
		if i == len(stages)-1 {
			cmd.Stdout = stdout
		} else {
			r, wr, err := os.Pipe()
			if err != nil {
				return "", err
			}
			ends = append(ends, r, wr)
			cmd.Stdout, prev = wr, r
		}
		cmds[i] = cmd
	}
	for i, cmd := range cmds {
		if err := cmd.Start(); err != nil {
			for _, started := range cmds[:i] {
				_ = started.Process.Kill()
				_ = started.Wait()
			}
			return "", fmt.Errorf("%s: %v", stages[i][0], err)
		}
	}
	// The children hold the pipe ends now; closing ours lets EOF through.
	for _, f := range ends {
		f.Close()
	}
	var codes []string
	for i, cmd := range cmds {
		err := cmd.Wait()
		var ee *exec.ExitError
		switch {
		case err == nil:
		case errors.As(err, &ee):
			if ee.ExitCode() >= 0 {
				codes = append(codes, fmt.Sprintf("%s exited %d", stages[i][0], ee.ExitCode()))
			} else if ctx.Err() == nil && i == len(cmds)-1 {
				codes = append(codes, fmt.Sprintf("%s: %v", stages[i][0], err))
			}
		default:
			if ctx.Err() == nil {
				codes = append(codes, fmt.Sprintf("%s: %v", stages[i][0], err))
			}
		}
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return "", fmt.Errorf("pipeline timed out after %s; narrow it with grep first", pipelineTimeout)
	}
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	var b strings.Builder
	b.Write(stdout.b.Bytes())
	if stdout.b.Len() == 0 {
		b.WriteString("(no output)\n")
	} else if !bytes.HasSuffix(stdout.b.Bytes(), []byte("\n")) {
		b.WriteString("\n")
	}
	if stdout.lost > 0 {
		fmt.Fprintf(&b, "[output stopped at %s; %d bytes more were dropped]\n", humanBytes(maxPipelineOutput), stdout.lost)
	}
	if len(codes) > 0 {
		fmt.Fprintf(&b, "[%s]\n", strings.Join(codes, "; "))
	}
	if stderr.b.Len() > 0 {
		fmt.Fprintf(&b, "[stderr]\n%s", stderr.b.String())
	}
	return b.String(), nil
}
