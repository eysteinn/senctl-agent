# senctl-agent

A small Go harness for tool-using LLM agents, usable as a **library** or as a
**command-line tool**.

- Point it at an **LLM proxy** and it works: it speaks the standard OpenAI chat
  completions API that proxies and gateways (LiteLLM, OpenRouter, vLLM, Ollama,
  Azure OpenAI, …) expose. It can also call the **Anthropic** API directly
  (official Go SDK).
- A bounded **agent loop** with tools: multi-turn sessions, or single tasks
  that end in a validated structured result.
- **Evidence checks** to verify that text an agent quotes really appears in what
  it read.
- Ready-made **workspace tools** (read-only files, optional shell) that cope with
  files far larger than a model's context, such as multi-gigabyte logs.

## Command line

Download a binary for Linux, macOS or Windows (amd64 / arm64) from the
[releases page](https://github.com/eysteinn/senctl-agent/releases) and put it on your `PATH`,
or build it with Go 1.26+:

```sh
go install github.com/eysteinn/senctl-agent/cmd/senctl-agent@latest   # or download a release binary

export SENCTL_AGENT_BASE_URL=https://llm-proxy.example.com   # with or without /v1
export SENCTL_AGENT_API_KEY=…            # or OPENAI_API_KEY
export SENCTL_AGENT_MODEL=my-model       # optional if the proxy serves one model

senctl-agent models                      # what the proxy serves

senctl-agent                             # interactive console
senctl-agent "why does the build fail?"  # console, starting with that message
senctl-agent run "where is the HTTP server started?"   # one-shot answer
git diff | senctl-agent run "review this change"
journalctl -u api | senctl-agent run "why does it restart?"  # large input is searched, not pasted
senctl-agent config                      # effective settings, key masked
```

### Interactive console

Replies stream as they are generated; tool use shows as `⏺ tool args` lines. Line editing,
history (`$XDG_STATE_HOME/senctl-agent/history`) and Tab completion of commands work as in
a shell. End a line with `\` to continue on the next one. Ctrl+C cancels the current reply
(press it twice at an empty prompt to quit), Ctrl+D quits. Mention a file as `@path` to attach
it: small text files are included, larger ones are described so the model can search them.

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
`read_file`, `file_info`, `list_dir`, `glob`, `grep` and `pipeline`. With `--edit` it can also
`write_file` and `edit_file` (exact-text replacement); with `--shell` it can run `shell`
commands. Paths can never leave the workspace, including through symlinks.

### Large files and output

Nothing is ever read whole into the model's context:

- `read_file` pages through a file (2000 lines or ~32 KB per call, long lines cut) and says
  where the next page starts; a negative `offset` reads from the end (`-100` = last 100 lines).
  A line index built on first use makes any page of a multi-gigabyte file instant.
- `file_info` gives size, line count, longest line and the first and last lines, so the model
  can decide how to approach a file before reading it.
- `grep` streams through files of any size, with `content` / `files` / `count` modes, context
  lines, paging (`offset`, `limit`) and totals. Lines that cannot match are skipped cheaply.
- `pipeline` runs a read-only text pipeline over one file, e.g.
  `grep -i error | cut -d' ' -f3 | sort | uniq -c | sort -rn | head`. It can use `grep`, `head`,
  `tail`, `wc`, `sort`, `uniq`, `cut`, `tr`, `nl`, `tac` and `jq` when installed, with the file on
  standard input. Redirection, variables, globbing, file arguments and options that read or
  write files are refused, and commands run with a minimal environment in an empty directory.
- `.gz` files are decompressed by all of these.
- Tool output too long for the context (over 40 KB) is saved in full to a session cache
  directory (`$XDG_CACHE_HOME/senctl-agent/sessions/…`, removed on exit). The model sees the
  first and last lines plus the file's path, and searches it with the same tools. So does
  large piped input to `run`.

| Setting | Console default | `run` default |
|---|---|---|
| `--edit off\|ask\|auto` | `ask` | `off` |
| `--shell off\|ask\|auto` | `off` | `off` |

In `run`, `ask` confirms on the terminal (`/dev/tty`), so it also works with piped input.

### Configuration

Settings come from flags, then `SENCTL_AGENT_*` environment variables, then a YAML file
(`$XDG_CONFIG_HOME/senctl-agent/config.yaml` or `./.senctl-agent.yaml`):

```yaml
base_url: https://llm-proxy.example.com
api_key: …
model: my-model       # optional if the proxy serves one model
effort: high          # reasoning effort hint
max_turns: 30         # model calls per prompt
max_tokens: 16000     # output tokens per model call
edit: ask             # off | ask | auto
shell: off            # off | ask | auto
send_effort: false    # openai: forward effort as reasoning_effort
provider: openai      # default; anthropic calls the Anthropic API directly
fallbacks: ""         # anthropic: server-side refusal fallback (default on for the first-party API)
```

Only `base_url` and `api_key` are needed for a proxy. If the URL answers 404, `/v1` is
tried and kept, so either form of the address works. Without a model, the proxy's only
model is used; if it serves several, the error lists them.

## Library

```go
import (
    "github.com/eysteinn/senctl-agent/agent"
    "github.com/eysteinn/senctl-agent/llm"
    "github.com/eysteinn/senctl-agent/tools"
)

provider, _ := llm.New(llm.Config{BaseURL: proxyURL, APIKey: key}) // OpenAI-compatible by default
conv := provider.NewConversation(llm.Options{Model: "my-model"}, systemPrompt, agent.Specs(myTools...))

// Chat-style: each Send serves tool calls until the model answers in text.
session := agent.NewSession(conv, myTools, agent.Config{MaxTurns: 20}, nil)
session.Stream(func(text string) { fmt.Print(text) }) // optional live output
answer, err := session.Send(ctx, "…")
session.SetOptions(llm.Options{Model: "other-model"}) // switch model mid-conversation

// Keep oversized tool output out of the context: save it and let the tools read it.
cache, _ := tools.NewCache("")
defer cache.Close()
ws, _ := tools.NewWorkspace(dir)
ws.AllowRead(cache.Dir())
session = agent.NewSession(conv, ws.Tools(), agent.Config{Spill: cache.Spill}, nil)

// Task-style: runs until the model calls `submit` with input it accepts.
result, usage, err := agent.Run(ctx, conv, prompt, myTools, submit, agent.Config{}, recorder)
```

Packages:

| Package | What it is |
|---|---|
| `llm` | `Provider` / `Conversation` interface, Anthropic and OpenAI-compatible adapters, `New(Config)`; optional `Streamer`, `Configurable`, `ModelLister` |
| `agent` | `Session` and `Run`, tool definitions, event recording, turn and output limits, spilling oversized tool output (`Config.Spill`) |
| `evidence` | Verbatim-quote checking against the text an agent was shown |
| `tools` | Workspace file tools for files of any size (read-only, plus write/edit with an approval hook), the read-only `pipeline` tool, a `Cache` for oversized output, and an opt-in shell tool |

Each conversation keeps its history in the provider's own wire format, so
provider-specific content (such as reasoning blocks that must be sent back
unchanged) survives across turns.

## Releasing

Push a tag like `v0.2.0`. The Release workflow runs GoReleaser (`.goreleaser.yaml`): it runs the
tests, builds the CLI for Linux, macOS and Windows on amd64 and arm64 with the version stamped
in (`senctl-agent version`), and publishes archives plus `checksums.txt` as a GitHub release.
`goreleaser release --snapshot --clean` builds the same archives locally without publishing.
