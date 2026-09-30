package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
	"github.com/eysteinn/senctl-agent/tools"
)

// A chat-style session with read-only file tools, against an
// OpenAI-compatible LLM proxy.
func Example_session() {
	provider, err := llm.New(llm.Config{Provider: "openai", BaseURL: "https://llm-proxy.example.com/v1", APIKey: "…"})
	if err != nil {
		log.Fatal(err)
	}
	ws, err := tools.NewWorkspace(".")
	if err != nil {
		log.Fatal(err)
	}
	conv := provider.NewConversation(llm.Options{Model: "my-model"}, "You answer questions about this repository.", agent.Specs(ws.Tools()...))
	session := agent.NewSession(conv, ws.Tools(), agent.Config{MaxTurns: 20}, nil)
	_ = session // answer, err := session.Send(ctx, "Where is the HTTP server started?")
}

// A task that must end in a structured result, submitted through a tool
// whose Run validates it.
func Example_run() {
	provider, _ := llm.New(llm.Config{Provider: "anthropic"}) // uses ANTHROPIC_API_KEY
	type verdict struct {
		Label  string `json:"label"`
		Reason string `json:"reason"`
	}
	submit := agent.Tool{
		Spec: llm.ToolSpec{Name: "submit", Description: "Submit the classification.", Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"label":  map[string]any{"type": "string", "enum": []string{"bug", "flake"}},
				"reason": map[string]any{"type": "string"},
			},
			"required": []string{"label", "reason"},
		}},
		Run: func(ctx context.Context, in json.RawMessage) (string, error) {
			var v verdict
			if err := json.Unmarshal(in, &v); err != nil || v.Label == "" {
				return "", fmt.Errorf("label and reason are required")
			}
			return "accepted", nil
		},
	}
	conv := provider.NewConversation(llm.Options{Model: llm.DefaultAnthropicModel}, "Classify CI failures.", []llm.ToolSpec{submit.Spec})
	_ = conv // raw, usage, err := agent.Run(ctx, conv, "Test X failed with …", nil, submit, agent.Config{}, nil)
}
