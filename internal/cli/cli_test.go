package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeProxy is an OpenAI-compatible endpoint. It lists two models; when
// the conversation has no tool result yet it asks to read notes.txt (or to
// write hello.txt when asked to create a file), otherwise it answers with
// the tool result's first line. It streams when asked to.
type fakeProxy struct {
	mu       sync.Mutex
	requests []map[string]any
	auth     []string
}

func (f *fakeProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/models") {
		_, _ = io.WriteString(w, `{"data":[{"id":"beta"},{"id":"alpha"}]}`)
		return
	}
	b, _ := io.ReadAll(r.Body)
	var req map[string]any
	_ = json.Unmarshal(b, &req)
	f.mu.Lock()
	f.requests = append(f.requests, req)
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	f.mu.Unlock()
	msgs := req["messages"].([]any)
	last := msgs[len(msgs)-1].(map[string]any)
	var text string
	var calls []any
	call := func(name, args string) {
		calls = append(calls, map[string]any{"id": "c1", "type": "function", "function": map[string]any{"name": name, "arguments": args}})
	}
	switch {
	case last["role"] == "tool":
		text = "The tool says: " + strings.SplitN(last["content"].(string), "\n", 2)[0]
	case strings.Contains(last["content"].(string), "notes"):
		call("read_file", `{"path":"notes.txt"}`)
	case strings.Contains(last["content"].(string), "create"):
		call("write_file", `{"path":"hello.txt","content":"hi\n"}`)
	default:
		text = "You said: " + last["content"].(string)
	}
	if req["stream"] == true {
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(v any) { b, _ := json.Marshal(v); fmt.Fprintf(w, "data: %s\n\n", b) }
		if text != "" {
			half := len(text) / 2
			send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": text[:half]}}}})
			send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": text[half:]}}}})
		}
		for i, c := range calls {
			c.(map[string]any)["index"] = i
			send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{c}}}}})
		}
		send(map[string]any{"choices": []any{}, "usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 2}})
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		return
	}
	msg := map[string]any{"role": "assistant", "content": nil}
	if text != "" {
		msg["content"] = text
	}
	if len(calls) > 0 {
		msg["tool_calls"] = calls
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"finish_reason": "stop", "message": msg}},
		"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 2}})
}

func run(t *testing.T, stdin io.Reader, args ...string) (string, string, error) {
	t.Helper()
	cmd := NewRootCmd()
	var out, errb bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	if stdin != nil {
		cmd.SetIn(stdin)
	} else {
		cmd.SetIn(strings.NewReader(""))
	}
	cmd.SetArgs(args)
	err := cmd.Execute()
	return out.String(), errb.String(), err
}

func workspace(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("deploy on fridays\nsecond line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestRunWithTool(t *testing.T) {
	proxy := &fakeProxy{}
	srv := httptest.NewServer(proxy)
	defer srv.Close()
	t.Setenv("SENCTL_AGENT_PROVIDER", "openai")
	t.Setenv("SENCTL_AGENT_BASE_URL", srv.URL)
	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	dir := workspace(t)

	out, stderr, err := run(t, nil, "run", "what do the notes say?", "--model", "proxy-model", "--dir", dir, "-v")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, stderr)
	}
	if strings.TrimSpace(out) != "The tool says: 1: deploy on fridays" {
		t.Fatalf("out = %q", out)
	}
	if !strings.Contains(stderr, "tool_call: read_file") {
		t.Fatalf("verbose stderr = %q", stderr)
	}
	if proxy.requests[0]["model"] != "proxy-model" || proxy.auth[0] != "Bearer sk-test" {
		t.Fatalf("request = %v auth = %v", proxy.requests[0]["model"], proxy.auth[0])
	}
	tools := proxy.requests[0]["tools"].([]any)
	if len(tools) != 4 {
		t.Fatalf("tools offered = %d (shell must be off by default)", len(tools))
	}

	out, _, err = run(t, nil, "run", "hello", "--model", "m", "--dir", dir, "--json")
	var res struct {
		Text  string
		Usage map[string]int64
	}
	if err != nil || json.Unmarshal([]byte(out), &res) != nil || res.Text != "You said: hello" || res.Usage["input_tokens"] != 10 {
		t.Fatalf("json out = %q, %v", out, err)
	}

	// Piped stdin is appended to the prompt.
	r, w, _ := os.Pipe()
	_, _ = w.WriteString("some piped text")
	w.Close()
	out, _, err = run(t, r, "run", "summarize", "--model", "m", "--dir", dir)
	if err != nil || !strings.Contains(out, "<stdin>") || !strings.Contains(out, "some piped text") {
		t.Fatalf("stdin out = %q, %v", out, err)
	}

	if _, _, err := run(t, nil, "run", "x", "--dir", dir); err == nil || !strings.Contains(err.Error(), "no model") {
		t.Fatalf("missing model err = %v", err)
	}
	if _, _, err := run(t, nil, "run", "x", "--model", "m", "--shell", "sometimes"); err == nil {
		t.Fatal("invalid --shell accepted")
	}
}

func TestConsole(t *testing.T) {
	proxy := &fakeProxy{}
	srv := httptest.NewServer(proxy)
	defer srv.Close()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir := workspace(t)
	saved := filepath.Join(t.TempDir(), "conv.md")
	script := strings.Join([]string{
		"/help",
		"/model",
		"/model alpha",
		"hello \\",
		"there",
		"please create a file",
		"y",
		"read the notes",
		"/edit off",
		"/tools",
		"/effort high",
		"/usage",
		"/save " + saved,
		"/clear",
		"/bogus",
		"/exit",
	}, "\n") + "\n"
	out, stderr, err := run(t, strings.NewReader(script), "--provider", "openai", "--base-url", srv.URL, "--model", "m", "--dir", dir)
	if err != nil {
		t.Fatalf("console: %v %s", err, stderr)
	}
	for _, want := range []string{
		"/model [id]",               // help
		"Model: m", "alpha", "beta", // model listing
		"Switched to alpha",              // model switch
		"You said: hello \nthere",        // continued line, streamed reply
		"Create file? hello.txt", "+ hi", // edit approval with preview
		"The tool says: Created hello.txt (1 line).",
		"The tool says: 1: deploy on fridays",
		"edit is now off",
		"Effort set to high.",
		"input tokens 50, output tokens 10", // 5 model calls
		"Saved to " + saved,
		"Started a new conversation.",
		"Unknown command /bogus",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("console output missing %q:\n%s", want, out)
		}
	}
	if b, err := os.ReadFile(filepath.Join(dir, "hello.txt")); err != nil || string(b) != "hi\n" {
		t.Fatalf("hello.txt = %q, %v", b, err)
	}
	// Requests after /model alpha use the new model and stream.
	if proxy.requests[0]["model"] != "alpha" || proxy.requests[0]["stream"] != true {
		t.Fatalf("first request = %v %v", proxy.requests[0]["model"], proxy.requests[0]["stream"])
	}
	toolNames := func(req map[string]any) string {
		var names []string
		for _, tl := range req["tools"].([]any) {
			names = append(names, tl.(map[string]any)["function"].(map[string]any)["name"].(string))
		}
		return strings.Join(names, ",")
	}
	if got := toolNames(proxy.requests[0]); !strings.Contains(got, "edit_file") || strings.Contains(got, "shell") {
		t.Fatalf("console tools = %s (edit asks by default, shell off)", got)
	}
	toolsSection := out[strings.Index(out, "edit is now off"):]
	if strings.Contains(toolsSection[:strings.Index(toolsSection, "Effort set")], "write_file") {
		t.Fatal("/tools still lists write_file after /edit off")
	}
	if md, _ := os.ReadFile(saved); !strings.Contains(string(md), "## You") || !strings.Contains(string(md), "## Assistant") {
		t.Fatalf("saved transcript = %s", md)
	}
}

func TestConsoleDeclinedEdit(t *testing.T) {
	srv := httptest.NewServer(&fakeProxy{})
	defer srv.Close()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	dir := workspace(t)
	out, _, err := run(t, strings.NewReader("create it\nn\n/exit\n"), "--provider", "openai", "--base-url", srv.URL, "--model", "m", "--dir", dir)
	if err != nil || !strings.Contains(out, "declined") {
		t.Fatalf("out = %s, err = %v", out, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "hello.txt")); err == nil {
		t.Fatal("declined edit was written")
	}
}

func TestConfigPrecedence(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	if err := os.MkdirAll(filepath.Join(cfgDir, "senctl-agent"), 0o755); err != nil {
		t.Fatal(err)
	}
	yaml := "provider: anthropic\nmodel: from-file\nbase_url: http://file/v1\napi_key: file-key-123456\nmax_turns: 7\n"
	if err := os.WriteFile(filepath.Join(cfgDir, "senctl-agent", "config.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SENCTL_AGENT_MODEL", "from-env")
	out, _, err := run(t, nil, "config", "--base-url", "http://flag/v1")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"provider:    anthropic", "model:       from-env", "base_url:    http://flag/v1", "api_key:     file…3456", "max_turns:   7", "config.yaml"} {
		if !strings.Contains(out, want) {
			t.Fatalf("config output missing %q:\n%s", want, out)
		}
	}
}
