// Package agent runs bounded tool-using conversations with a model.
//
// Two shapes are supported:
//   - Session: a multi-turn conversation. Each Send serves tool calls until
//     the model answers in plain text (like a chat assistant).
//   - Run: a single task that ends when the model calls a designated
//     submit tool with input the tool accepts (for structured results).
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/eysteinn/senctl-agent/llm"
)

// Tool is a tool the model may call. Run returns the text sent back to the
// model; an error is sent back as a tool error so the model can adjust.
type Tool struct {
	Spec llm.ToolSpec
	Run  func(ctx context.Context, input json.RawMessage) (string, error)
}

// Specs returns the specs of tools, in order.
func Specs(tools ...Tool) []llm.ToolSpec {
	out := make([]llm.ToolSpec, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.Spec)
	}
	return out
}

// Event kinds recorded while the agent runs.
const (
	EventModelText  = "model_text"
	EventToolCall   = "tool_call"
	EventToolResult = "tool_result"
	EventToolError  = "tool_error"
	EventNote       = "note"
)

// Event is one step of an agent run, for observability.
type Event struct {
	Kind    string
	Content string
}

// Recorder receives events as they happen. It must not block for long.
type Recorder func(ctx context.Context, e Event)

// Config bounds a run.
type Config struct {
	// MaxTurns caps model calls per task or per Session.Send (default 30).
	MaxTurns int
	// MaxToolOutput caps the bytes of one tool result sent to the model
	// (default 40000). Longer output is saved with Spill when it is set, and
	// the model gets its first and last lines plus where to find the rest;
	// otherwise it is cut with a marker.
	MaxToolOutput int
	// Spill stores a tool result too large to send whole and returns a
	// reference the model can use with its tools, such as a file path.
	// tools.Cache.Spill fits.
	Spill func(ctx context.Context, tool, output string) (ref string, err error)
}

func (c Config) withDefaults() Config {
	if c.MaxTurns <= 0 {
		c.MaxTurns = 30
	}
	if c.MaxToolOutput <= 0 {
		c.MaxToolOutput = 40000
	}
	return c
}

// ErrNoResult is returned when the model stops without submitting a result
// within the turn budget.
var ErrNoResult = errors.New("agent: no result submitted within the turn budget")

// ErrTurnBudget is returned when a Session turn is still calling tools when
// the turn budget runs out.
var ErrTurnBudget = errors.New("agent: turn budget exhausted while the model was still calling tools")

// maxNudges is how many times a model that stops without submitting is
// reminded to call the submit tool.
const maxNudges = 2

// server executes tool calls and records events.
type server struct {
	tools map[string]Tool
	cfg   Config
	rec   Recorder
}

func newServer(tools []Tool, cfg Config, rec Recorder) *server {
	if rec == nil {
		rec = func(context.Context, Event) {}
	}
	s := &server{tools: map[string]Tool{}, cfg: cfg.withDefaults(), rec: rec}
	for _, t := range tools {
		s.tools[t.Spec.Name] = t
	}
	return s
}

func (s *server) toolError(ctx context.Context, call llm.ToolCall, msg string) llm.ToolResult {
	s.rec(ctx, Event{Kind: EventToolError, Content: call.Name + ": " + msg})
	return llm.ToolResult{CallID: call.ID, Content: msg, IsError: true}
}

// serve runs one tool call. accepted reports that call was the submit tool
// and its input was accepted.
func (s *server) serve(ctx context.Context, call llm.ToolCall, submit string) (res llm.ToolResult, accepted bool, err error) {
	s.rec(ctx, Event{Kind: EventToolCall, Content: call.Name + " " + string(call.Input)})
	tool, ok := s.tools[call.Name]
	if !ok {
		return s.toolError(ctx, call, fmt.Sprintf("unknown tool %q", call.Name)), false, nil
	}
	if !json.Valid(call.Input) {
		return s.toolError(ctx, call, "tool input is not valid JSON; call the tool again"), false, nil
	}
	out, err := tool.Run(ctx, call.Input)
	if err != nil {
		if ctx.Err() != nil {
			return llm.ToolResult{}, false, ctx.Err()
		}
		return s.toolError(ctx, call, err.Error()), false, nil
	}
	if submit != "" && call.Name == submit {
		return llm.ToolResult{CallID: call.ID, Content: out}, true, nil
	}
	out = s.fit(ctx, call.Name, out)
	s.rec(ctx, Event{Kind: EventToolResult, Content: out})
	return llm.ToolResult{CallID: call.ID, Content: out}, false, nil
}

// fit returns out as sent to the model: whole when it is small enough,
// otherwise spilled (or cut) with a note.
func (s *server) fit(ctx context.Context, tool, out string) string {
	limit := s.cfg.MaxToolOutput
	if len(out) <= limit {
		return out
	}
	var ref string
	var err error
	if s.cfg.Spill != nil {
		ref, err = s.cfg.Spill(ctx, tool, out)
	}
	if s.cfg.Spill == nil || err != nil {
		note := fmt.Sprintf("\n[output cut at %d of %d bytes; narrow the request]", limit, len(out))
		if err != nil {
			s.rec(ctx, Event{Kind: EventNote, Content: "saving large tool output failed: " + err.Error()})
		}
		return head(out, limit) + note
	}
	lines := strings.Count(out, "\n")
	if !strings.HasSuffix(out, "\n") {
		lines++
	}
	top := head(out, limit*2/3)
	bottom := tail(out, limit/4)
	omitted := lines - strings.Count(top, "\n") - strings.Count(bottom, "\n")
	return fmt.Sprintf("[This output is %d bytes (%d lines), too much to show whole. All of it is saved at %s: search it and read parts of it with your file tools instead of repeating the call. Its start and end follow.]\n%s\n[… %d lines not shown …]\n%s",
		len(out), lines, ref, strings.TrimSuffix(top, "\n"), max(omitted, 0), bottom)
}

// head returns the start of s, at most n bytes, ending at a line break
// when one is near.
func head(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if i := strings.LastIndexByte(s[:n], '\n'); i >= n/2 {
		return s[:i+1]
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// tail returns the end of s, at most n bytes, starting after a line break
// when one is near.
func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	start := len(s) - n
	if i := strings.IndexByte(s[start:], '\n'); i >= 0 && i < n/2 {
		return s[start+i+1:]
	}
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return s[start:]
}

func addUsage(total *llm.Usage, t *llm.Turn) {
	if t != nil {
		total.InputTokens += t.Usage.InputTokens
		total.OutputTokens += t.Usage.OutputTokens
	}
}

// Run sends prompt and serves tool calls until the model calls submit with
// input that submit.Run accepts. It returns that input and the total usage.
func Run(ctx context.Context, conv llm.Conversation, prompt string, tools []Tool, submit Tool, cfg Config, rec Recorder) (json.RawMessage, llm.Usage, error) {
	srv := newServer(append(append([]Tool{}, tools...), submit), cfg, rec)
	var usage llm.Usage
	text, results := prompt, []llm.ToolResult(nil)
	nudges := 0
	for turnN := 0; turnN < srv.cfg.MaxTurns; turnN++ {
		turn, err := conv.Send(ctx, text, results)
		addUsage(&usage, turn)
		if err != nil {
			return nil, usage, err
		}
		text, results = "", nil
		if strings.TrimSpace(turn.Text) != "" {
			srv.rec(ctx, Event{Kind: EventModelText, Content: turn.Text})
		}
		if len(turn.ToolCalls) == 0 {
			if nudges >= maxNudges {
				return nil, usage, ErrNoResult
			}
			nudges++
			note := fmt.Sprintf("Finish by calling the %s tool with your result.", submit.Spec.Name)
			if turn.Truncated {
				note = "Your reply was cut off at the output limit. Be more concise. " + note
			}
			srv.rec(ctx, Event{Kind: EventNote, Content: note})
			text = note
			continue
		}
		for _, call := range turn.ToolCalls {
			res, accepted, err := srv.serve(ctx, call, submit.Spec.Name)
			if err != nil {
				return nil, usage, err
			}
			if accepted {
				return call.Input, usage, nil
			}
			results = append(results, res)
		}
	}
	return nil, usage, ErrNoResult
}

// Session is a multi-turn conversation with tools.
type Session struct {
	conv   llm.Conversation
	srv    *server
	usage  llm.Usage
	onText func(string)
	// pending holds tool results not yet sent (after an interrupted turn).
	pending []llm.ToolResult
}

// NewSession starts a session over conv with the given tools.
func NewSession(conv llm.Conversation, tools []Tool, cfg Config, rec Recorder) *Session {
	return &Session{conv: conv, srv: newServer(tools, cfg, rec)}
}

// Usage is the total token usage so far.
func (s *Session) Usage() llm.Usage { return s.usage }

// Stream makes Send deliver the model's text to onText as it is generated,
// when the conversation supports streaming (llm.Streamer). nil turns it off.
func (s *Session) Stream(onText func(string)) { s.onText = onText }

// SetOptions changes model options for the following turns. It reports
// false when the conversation cannot change them (llm.Configurable).
func (s *Session) SetOptions(opts llm.Options) bool {
	c, ok := s.conv.(llm.Configurable)
	if ok {
		c.SetOptions(opts)
	}
	return ok
}

// SetTools replaces the tools offered to the model for the following turns.
// It reports false when the conversation cannot change them.
func (s *Session) SetTools(tools []Tool) bool {
	c, ok := s.conv.(llm.Configurable)
	if !ok {
		return false
	}
	c.SetTools(Specs(tools...))
	s.srv = newServer(tools, s.srv.cfg, s.srv.rec)
	return true
}

func (s *Session) send(ctx context.Context, text string, results []llm.ToolResult) (*llm.Turn, error) {
	if st, ok := s.conv.(llm.Streamer); ok && s.onText != nil {
		return st.SendStream(ctx, text, results, s.onText)
	}
	return s.conv.Send(ctx, text, results)
}

// Send adds a user message and serves tool calls until the model answers
// without calling tools, returning that answer. If a request fails midway
// (including a cancelled context), tool results that were not delivered are
// kept and sent with the next message, so the conversation stays valid.
func (s *Session) Send(ctx context.Context, text string) (string, error) {
	results := s.pending
	s.pending = nil
	for turnN := 0; turnN < s.srv.cfg.MaxTurns; turnN++ {
		turn, err := s.send(ctx, text, results)
		addUsage(&s.usage, turn)
		if err != nil {
			if turn == nil {
				s.pending = results
			}
			return "", err
		}
		text, results = "", nil
		if len(turn.ToolCalls) == 0 {
			if turn.Truncated {
				return turn.Text, errors.New("agent: the reply was cut off at the output limit")
			}
			return turn.Text, nil
		}
		if strings.TrimSpace(turn.Text) != "" && s.onText == nil {
			s.srv.rec(ctx, Event{Kind: EventModelText, Content: turn.Text})
		}
		for i, call := range turn.ToolCalls {
			res, _, err := s.srv.serve(ctx, call, "")
			if err != nil {
				// Every call needs an answer before the model can continue.
				for _, rest := range turn.ToolCalls[i:] {
					results = append(results, llm.ToolResult{CallID: rest.ID, Content: "cancelled by the user", IsError: true})
				}
				s.pending = results
				return "", err
			}
			results = append(results, res)
		}
	}
	// Keep the unanswered results so the next Send stays well-formed.
	s.pending = results
	return "", ErrTurnBudget
}
