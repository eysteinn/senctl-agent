# senctl-agent

A small Go harness for tool-using LLM agents, usable as a **library** or as a
**command-line tool**.

- Talks to any **OpenAI-compatible** chat-completions endpoint (OpenAI, Azure
  OpenAI, vLLM, Ollama, LiteLLM and other LLM proxies) or the **Anthropic** API
  (official Go SDK).
- A bounded **agent loop** with tools: multi-turn sessions, or single tasks
  that end in a validated structured result.
- **Evidence checks** to verify that text an agent quotes really appears in what
  it read.
- Ready-made **workspace tools** (read-only files, optional shell).

## Command line

```sh
go install github.com/eysteinn/senctl-agent/cmd/senctl-agent@latest

export SENCTL_AGENT_PROVIDER=openai
export SENCTL_AGENT_BASE_URL=https://llm-proxy.example.com/v1
export SENCTL_AGENT_API_KEY=…            # or OPENAI_API_KEY / ANTHROPIC_API_KEY
export SENCTL_AGENT_MODEL=my-model       # anthropic defaults to claude-opus-5-5

senctl-agent run "where is the HTTP server started?"
git diff | senctl-agent run "review this change"
senctl-agent chat                        # interactive; /usage, /reset, /exit
senctl-agent config                      # effective settings, key masked
```

The model can read files under `--dir` (default: the current directory)
through `read_file`, `list_dir`, `glob` and `grep`; paths cannot escape it,
including through symlinks. Shell access is off unless you pass
`--shell ask` (you confirm every command on the terminal) or `--shell auto`.

Settings come from flags, then `SENCTL_AGENT_*` environment variables, then a
YAML file (`$XDG_CONFIG_HOME/senctl-agent/config.yaml` or `./.senctl-agent.yaml`):

```yaml
provider: openai
base_url: https://llm-proxy.example.com/v1
model: my-model
effort: high          # reasoning effort hint
max_turns: 30         # model calls per prompt
max_tokens: 16000     # output tokens per model call
send_effort: false    # openai: forward effort as reasoning_effort
fallbacks: ""         # anthropic: server-side refusal fallback (default on for the first-party API)
```

## Library

```go
import (
    "github.com/eysteinn/senctl-agent/agent"
    "github.com/eysteinn/senctl-agent/llm"
)

provider, _ := llm.New(llm.Config{Provider: "openai", BaseURL: proxyURL, APIKey: key})
conv := provider.NewConversation(llm.Options{Model: "my-model"}, systemPrompt, agent.Specs(myTools...))

// Chat-style: each Send serves tool calls until the model answers in text.
session := agent.NewSession(conv, myTools, agent.Config{MaxTurns: 20}, nil)
answer, err := session.Send(ctx, "…")

// Task-style: runs until the model calls `submit` with input it accepts.
result, usage, err := agent.Run(ctx, conv, prompt, myTools, submit, agent.Config{}, recorder)
```

Packages:

| Package | What it is |
|---|---|
| `llm` | `Provider` / `Conversation` interface, Anthropic and OpenAI-compatible adapters, `New(Config)` |
| `agent` | `Session` and `Run`, tool definitions, event recording, turn and output limits |
| `evidence` | Verbatim-quote checking against the text an agent was shown |
| `tools` | Read-only workspace file tools and an opt-in shell tool |

Each conversation keeps its history in the provider's own wire format, so
provider-specific content (such as reasoning blocks that must be sent back
unchanged) survives across turns.
