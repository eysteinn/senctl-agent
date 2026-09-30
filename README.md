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

Download a binary for Linux, macOS or Windows (amd64 / arm64) from the
[releases page](https://github.com/eysteinn/senctl-agent/releases) and put it on your `PATH`,
or build it with Go 1.26+:

```sh
go install github.com/eysteinn/senctl-agent/cmd/senctl-agent@latest   # or download a release binary

export SENCTL_AGENT_PROVIDER=openai
export SENCTL_AGENT_BASE_URL=https://llm-proxy.example.com/v1
export SENCTL_AGENT_API_KEY=…            # or OPENAI_API_KEY / ANTHROPIC_API_KEY
export SENCTL_AGENT_MODEL=my-model       # anthropic defaults to claude-opus-5-5

senctl-agent                             # interactive console
senctl-agent "why does the build fail?"  # console, starting with that message
senctl-agent run "where is the HTTP server started?"   # one-shot answer
git diff | senctl-agent run "review this change"
senctl-agent config                      # effective settings, key masked
```

### Interactive console

Replies stream as they are generated; tool use shows as `⏺ tool args` lines. Line editing,
history (`$XDG_STATE_HOME/senctl-agent/history`) and Tab completion of commands work as in
a shell. End a line with `\` to continue on the next one. Ctrl+C cancels the current reply
(press it twice at an empty prompt to quit), Ctrl+D quits.

| Command | |
|---|---|
| `/model [id]` | show the model (and the provider's models), or switch; the conversation continues |
| `/models` | list the provider's models |
| `/effort [level]` | show or set reasoning effort (`low` … `max`) |
| `/edit [off\|ask\|auto]` | file editing; `ask` shows a diff and asks `[y]es / [n]o / [a]lways` |
| `/shell [off\|ask\|auto]` | shell commands; `ask` confirms each one |
| `/tools` | tools the model can use right now |
| `/usage` | tokens used in this session |
| `/clear` | start a new conversation, keeping settings |
| `/system` | show the system prompt |
| `/save [file]` | save the conversation as Markdown |
| `/help`, `/exit` | |

### Tools and permissions

The model always has read-only file tools under `--dir` (default: the current directory):
`read_file`, `list_dir`, `glob`, `grep`. With `--edit` it can also `write_file` and
`edit_file` (exact-text replacement); with `--shell` it can run `shell` commands. Paths can
never leave the workspace, including through symlinks.

| Setting | Console default | `run` default |
|---|---|---|
| `--edit off\|ask\|auto` | `ask` | `off` |
| `--shell off\|ask\|auto` | `off` | `off` |

In `run`, `ask` confirms on the terminal (`/dev/tty`), so it also works with piped input.

### Configuration

Settings come from flags, then `SENCTL_AGENT_*` environment variables, then a YAML file
(`$XDG_CONFIG_HOME/senctl-agent/config.yaml` or `./.senctl-agent.yaml`):

```yaml
provider: openai
base_url: https://llm-proxy.example.com/v1
model: my-model
effort: high          # reasoning effort hint
max_turns: 30         # model calls per prompt
max_tokens: 16000     # output tokens per model call
edit: ask             # off | ask | auto
shell: off            # off | ask | auto
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
session.Stream(func(text string) { fmt.Print(text) }) // optional live output
answer, err := session.Send(ctx, "…")
session.SetOptions(llm.Options{Model: "other-model"}) // switch model mid-conversation

// Task-style: runs until the model calls `submit` with input it accepts.
result, usage, err := agent.Run(ctx, conv, prompt, myTools, submit, agent.Config{}, recorder)
```

Packages:

| Package | What it is |
|---|---|
| `llm` | `Provider` / `Conversation` interface, Anthropic and OpenAI-compatible adapters, `New(Config)`; optional `Streamer`, `Configurable`, `ModelLister` |
| `agent` | `Session` and `Run`, tool definitions, event recording, turn and output limits |
| `evidence` | Verbatim-quote checking against the text an agent was shown |
| `tools` | Workspace file tools (read-only, plus write/edit with an approval hook) and an opt-in shell tool |

Each conversation keeps its history in the provider's own wire format, so
provider-specific content (such as reasoning blocks that must be sent back
unchanged) survives across turns.

## Releasing

Push a tag like `v0.2.0`. The Release workflow runs GoReleaser (`.goreleaser.yaml`): it runs the
tests, builds the CLI for Linux, macOS and Windows on amd64 and arm64 with the version stamped
in (`senctl-agent version`), and publishes archives plus `checksums.txt` as a GitHub release.
`goreleaser release --snapshot --clean` builds the same archives locally without publishing.
