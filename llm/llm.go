// Package llm is a small provider-neutral interface over chat models with
// tool use, used by the optional analysis worker. Each Conversation keeps
// its history in the provider's own wire format, so provider-specific
// content (for example reasoning blocks that must be sent back unchanged)
// survives across turns.
package llm

import (
	"context"
	"encoding/json"
	"errors"
)

// ErrRefused is returned when the model declines to answer.
var ErrRefused = errors.New("llm: the model declined the request")

// ToolSpec describes a tool the model may call. Schema is a JSON Schema
// object for the tool input.
type ToolSpec struct {
	Name        string
	Description string
	Schema      map[string]any
}

// ToolCall is one tool invocation requested by the model.
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult answers one ToolCall.
type ToolResult struct {
	CallID  string
	Content string
	IsError bool
}

// Usage counts tokens for one model call.
type Usage struct {
	InputTokens  int64
	OutputTokens int64
}

// Turn is the model's reply to one Send.
type Turn struct {
	Text      string
	ToolCalls []ToolCall
	// Truncated reports that the reply hit the output token limit.
	Truncated bool
	Usage     Usage
}

// Options configure a conversation.
type Options struct {
	Model string
	// Effort is a provider-neutral reasoning effort hint: low, medium, high
	// (providers may accept more). Empty leaves the provider default.
	Effort    string
	MaxTokens int64
}

// Provider starts conversations with one configured model endpoint.
type Provider interface {
	// Name identifies the provider ("anthropic", "openai").
	Name() string
	NewConversation(opts Options, system string, tools []ToolSpec) Conversation
}

// Conversation is one multi-turn exchange. Send appends a user turn made of
// tool results (answering the previous turn's calls, if any) followed by
// optional text, and returns the model's next turn.
type Conversation interface {
	Send(ctx context.Context, text string, results []ToolResult) (*Turn, error)
}

// Streamer is implemented by conversations that can stream the model's
// text while it is generated. onText receives each piece of text as it
// arrives; the returned Turn is the same as Send's.
type Streamer interface {
	SendStream(ctx context.Context, text string, results []ToolResult, onText func(string)) (*Turn, error)
}

// Configurable is implemented by conversations whose options and tools can
// change between turns, e.g. to switch model mid-conversation.
type Configurable interface {
	SetOptions(opts Options)
	SetTools(tools []ToolSpec)
}

// ModelLister is implemented by providers that can list their models.
type ModelLister interface {
	ListModels(ctx context.Context) ([]string, error)
}
