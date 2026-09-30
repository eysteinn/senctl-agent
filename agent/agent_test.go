package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/eysteinn/senctl-agent/llm"
)

// scripted replays turns and records what the agent sent.
type scripted struct {
	turns []*llm.Turn
	sent  []sent
}

type sent struct {
	text    string
	results []llm.ToolResult
}

func (s *scripted) Send(ctx context.Context, text string, results []llm.ToolResult) (*llm.Turn, error) {
	s.sent = append(s.sent, sent{text, results})
	if len(s.sent) > len(s.turns) {
		return &llm.Turn{}, nil
	}
	return s.turns[len(s.sent)-1], nil
}

func call(id, name, input string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: name, Input: json.RawMessage(input)}
}

var echo = Tool{
	Spec: llm.ToolSpec{Name: "echo"},
	Run:  func(ctx context.Context, in json.RawMessage) (string, error) { return "echo:" + string(in), nil },
}

var submit = Tool{
	Spec: llm.ToolSpec{Name: "submit"},
	Run: func(ctx context.Context, in json.RawMessage) (string, error) {
		var v struct{ Answer string }
		_ = json.Unmarshal(in, &v)
		if v.Answer == "" {
			return "", errors.New("answer is required")
		}
		return "", nil
	},
}

func TestRun(t *testing.T) {
	t.Run("tools then submit", func(t *testing.T) {
		conv := &scripted{turns: []*llm.Turn{
			{ToolCalls: []llm.ToolCall{call("1", "echo", `{"a":1}`), call("2", "nope", `{}`)}, Usage: llm.Usage{InputTokens: 10, OutputTokens: 1}},
			{ToolCalls: []llm.ToolCall{call("3", "submit", `{}`)}},
			{ToolCalls: []llm.ToolCall{call("4", "submit", `{"Answer":"42"}`)}, Usage: llm.Usage{InputTokens: 5}},
		}}
		var events []string
		out, usage, err := Run(context.Background(), conv, "go", []Tool{echo}, submit, Config{}, func(_ context.Context, e Event) { events = append(events, e.Kind) })
		if err != nil || string(out) != `{"Answer":"42"}` || usage.InputTokens != 15 {
			t.Fatalf("out=%s usage=%+v err=%v", out, usage, err)
		}
		r := conv.sent[1].results
		if len(r) != 2 || r[0].Content != `echo:{"a":1}` || !r[1].IsError {
			t.Fatalf("results = %+v", r)
		}
		if r := conv.sent[2].results; len(r) != 1 || !r[0].IsError || !strings.Contains(r[0].Content, "required") {
			t.Fatalf("invalid submit not rejected: %+v", r)
		}
		if strings.Join(events, ",") != "tool_call,tool_result,tool_call,tool_error,tool_call,tool_error,tool_call" {
			t.Fatalf("events = %v", events)
		}
	})

	t.Run("nudged then gives up", func(t *testing.T) {
		conv := &scripted{turns: []*llm.Turn{{Text: "thinking out loud"}, {Text: "still"}, {Text: "no"}}}
		_, _, err := Run(context.Background(), conv, "go", nil, submit, Config{}, nil)
		if !errors.Is(err, ErrNoResult) || len(conv.sent) != 3 || !strings.Contains(conv.sent[1].text, "submit") {
			t.Fatalf("err=%v sent=%+v", err, conv.sent)
		}
	})

	t.Run("turn budget", func(t *testing.T) {
		loop := &llm.Turn{ToolCalls: []llm.ToolCall{call("x", "echo", `{}`)}}
		conv := &scripted{turns: []*llm.Turn{loop, loop, loop, loop}}
		if _, _, err := Run(context.Background(), conv, "go", []Tool{echo}, submit, Config{MaxTurns: 3}, nil); !errors.Is(err, ErrNoResult) || len(conv.sent) != 3 {
			t.Fatalf("err=%v sends=%d", err, len(conv.sent))
		}
	})

	t.Run("long tool output is cut", func(t *testing.T) {
		big := Tool{Spec: llm.ToolSpec{Name: "big"}, Run: func(context.Context, json.RawMessage) (string, error) { return strings.Repeat("x", 100), nil }}
		conv := &scripted{turns: []*llm.Turn{
			{ToolCalls: []llm.ToolCall{call("1", "big", `{}`)}},
			{ToolCalls: []llm.ToolCall{call("2", "submit", `{"Answer":"a"}`)}},
		}}
		if _, _, err := Run(context.Background(), conv, "go", []Tool{big}, submit, Config{MaxToolOutput: 10}, nil); err != nil {
			t.Fatal(err)
		}
		if c := conv.sent[1].results[0].Content; !strings.HasPrefix(c, "xxxxxxxxxx\n[output cut") {
			t.Fatalf("content = %q", c)
		}
	})

	t.Run("long tool output is spilled", func(t *testing.T) {
		var lines []string
		for i := 1; i <= 1000; i++ {
			lines = append(lines, fmt.Sprintf("line %d", i))
		}
		out := strings.Join(lines, "\n") + "\n"
		big := Tool{Spec: llm.ToolSpec{Name: "big"}, Run: func(context.Context, json.RawMessage) (string, error) { return out, nil }}
		conv := &scripted{turns: []*llm.Turn{
			{ToolCalls: []llm.ToolCall{call("1", "big", `{}`)}},
			{ToolCalls: []llm.ToolCall{call("2", "submit", `{"Answer":"a"}`)}},
		}}
		var saved string
		spill := func(_ context.Context, tool, s string) (string, error) {
			saved = s
			return "/cache/" + tool + ".txt", nil
		}
		if _, _, err := Run(context.Background(), conv, "go", []Tool{big}, submit, Config{MaxToolOutput: 900, Spill: spill}, nil); err != nil {
			t.Fatal(err)
		}
		c := conv.sent[1].results[0].Content
		if saved != out || len(c) > 1200 {
			t.Fatalf("saved %d bytes, sent %d", len(saved), len(c))
		}
		for _, want := range []string{"saved at /cache/big.txt", "(1000 lines)", "\nline 1\n", "\nline 1000\n", "lines not shown"} {
			if !strings.Contains(c, want) {
				t.Fatalf("missing %q in %q", want, c)
			}
		}
	})

	t.Run("provider error", func(t *testing.T) {
		conv := &errConv{}
		if _, _, err := Run(context.Background(), conv, "go", nil, submit, Config{}, nil); !errors.Is(err, llm.ErrRefused) {
			t.Fatalf("err = %v", err)
		}
	})
}

type errConv struct{}

func (errConv) Send(context.Context, string, []llm.ToolResult) (*llm.Turn, error) {
	return &llm.Turn{}, llm.ErrRefused
}

func TestSession(t *testing.T) {
	conv := &scripted{turns: []*llm.Turn{
		{Text: "let me look", ToolCalls: []llm.ToolCall{call("1", "echo", `{"x":1}`)}},
		{Text: "The answer is 1."},
		{Text: "Second answer."},
	}}
	var events []string
	s := NewSession(conv, []Tool{echo}, Config{}, func(_ context.Context, e Event) { events = append(events, e.Kind) })
	got, err := s.Send(context.Background(), "q1")
	if err != nil || got != "The answer is 1." {
		t.Fatalf("send 1 = %q, %v", got, err)
	}
	if r := conv.sent[1].results; len(r) != 1 || r[0].Content != `echo:{"x":1}` {
		t.Fatalf("results = %+v", r)
	}
	if got, err := s.Send(context.Background(), "q2"); err != nil || got != "Second answer." || conv.sent[2].text != "q2" {
		t.Fatalf("send 2 = %q, %v", got, err)
	}
	if strings.Join(events, ",") != "model_text,tool_call,tool_result" {
		t.Fatalf("events = %v", events)
	}
}

func TestSessionTurnBudget(t *testing.T) {
	loop := &llm.Turn{ToolCalls: []llm.ToolCall{call("x", "echo", `{}`)}}
	conv := &scripted{turns: []*llm.Turn{loop, loop, {Text: "done"}}}
	s := NewSession(conv, []Tool{echo}, Config{MaxTurns: 2}, nil)
	if _, err := s.Send(context.Background(), "go"); !errors.Is(err, ErrTurnBudget) {
		t.Fatalf("err = %v", err)
	}
	// The pending tool results go out with the next message.
	if got, err := s.Send(context.Background(), "continue"); err != nil || got != "done" || len(conv.sent[2].results) != 1 {
		t.Fatalf("resume = %q, %v, %+v", got, err, conv.sent[2])
	}
}

// streamConv streams each turn's text in two pieces.
type streamConv struct{ scripted }

func (s *streamConv) SendStream(ctx context.Context, text string, results []llm.ToolResult, onText func(string)) (*llm.Turn, error) {
	t, err := s.Send(ctx, text, results)
	if t != nil && t.Text != "" {
		onText(t.Text[:len(t.Text)/2])
		onText(t.Text[len(t.Text)/2:])
	}
	return t, err
}

func TestSessionStream(t *testing.T) {
	conv := &streamConv{scripted{turns: []*llm.Turn{{Text: "streamed answer"}}}}
	s := NewSession(conv, nil, Config{}, nil)
	var got []string
	s.Stream(func(d string) { got = append(got, d) })
	answer, err := s.Send(context.Background(), "q")
	if err != nil || answer != "streamed answer" || strings.Join(got, "|") != "streame|d answer" {
		t.Fatalf("answer %q pieces %v err %v", answer, got, err)
	}
}

// cancelOnSecond fails the second request, like a cancelled context.
type cancelOnSecond struct {
	scripted
	calls int
}

func (c *cancelOnSecond) Send(ctx context.Context, text string, results []llm.ToolResult) (*llm.Turn, error) {
	c.calls++
	if c.calls == 2 {
		return nil, context.Canceled
	}
	return c.scripted.Send(ctx, text, results)
}

func TestSessionKeepsToolResultsAfterCancel(t *testing.T) {
	conv := &cancelOnSecond{scripted: scripted{turns: []*llm.Turn{
		{ToolCalls: []llm.ToolCall{call("1", "echo", `{}`)}},
		{Text: "after resume"},
	}}}
	s := NewSession(conv, []Tool{echo}, Config{}, nil)
	if _, err := s.Send(context.Background(), "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	// The tool result that never reached the model goes with the next message.
	answer, err := s.Send(context.Background(), "go on")
	if err != nil || answer != "after resume" {
		t.Fatalf("answer %q err %v", answer, err)
	}
	last := conv.sent[len(conv.sent)-1]
	if last.text != "go on" || len(last.results) != 1 || last.results[0].CallID != "1" {
		t.Fatalf("resumed request = %+v", last)
	}
}

func TestSessionSetToolsAndOptions(t *testing.T) {
	conv := &scripted{}
	s := NewSession(conv, nil, Config{}, nil)
	if s.SetTools([]Tool{echo}) || s.SetOptions(llm.Options{Model: "x"}) {
		t.Fatal("a non-configurable conversation reported success")
	}
}
