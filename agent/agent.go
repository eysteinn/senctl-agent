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
	// (default 20000); longer output is cut with a marker.
	MaxToolOutput int
}

func (c Config) withDefaults() Config {
	if c.MaxTurns <= 0 {
		c.MaxTurns = 30
	}
	if c.MaxToolOutput <= 0 {
		c.MaxToolOutput = 20000
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
	if len(out) > s.cfg.MaxToolOutput {
		out = out[:s.cfg.MaxToolOutput] + fmt.Sprintf("\n[output cut at %d bytes; narrow the request]", s.cfg.MaxToolOutput)
	}
	s.rec(ctx, Event{Kind: EventToolResult, Content: out})
	return llm.ToolResult{CallID: call.ID, Content: out}, false, nil
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
	conv  llm.Conversation
	srv   *server
	usage llm.Usage
	// pending holds tool results not yet sent (after an interrupted turn).
	pending []llm.ToolResult
}

// NewSession starts a session over conv with the given tools.
func NewSession(conv llm.Conversation, tools []Tool, cfg Config, rec Recorder) *Session {
	return &Session{conv: conv, srv: newServer(tools, cfg, rec)}
}

// Usage is the total token usage so far.
func (s *Session) Usage() llm.Usage { return s.usage }

// Send adds a user message and serves tool calls until the model answers
// without calling tools, returning that answer.
func (s *Session) Send(ctx context.Context, text string) (string, error) {
	results := s.pending
	s.pending = nil
	for turnN := 0; turnN < s.srv.cfg.MaxTurns; turnN++ {
		turn, err := s.conv.Send(ctx, text, results)
		addUsage(&s.usage, turn)
		if err != nil {
			return "", err
		}
		text, results = "", nil
		if len(turn.ToolCalls) == 0 {
			if turn.Truncated {
				return turn.Text, errors.New("agent: the reply was cut off at the output limit")
			}
			return turn.Text, nil
		}
		if strings.TrimSpace(turn.Text) != "" {
			s.srv.rec(ctx, Event{Kind: EventModelText, Content: turn.Text})
		}
		for _, call := range turn.ToolCalls {
			res, _, err := s.srv.serve(ctx, call, "")
			if err != nil {
				return "", err
			}
			results = append(results, res)
		}
	}
	// Keep the unanswered results so the next Send stays well-formed.
	s.pending = results
	return "", ErrTurnBudget
}
