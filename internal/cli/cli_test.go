package cli

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeProxy is an OpenAI-compatible endpoint: when the conversation has no
// tool result yet it asks to read notes.txt, otherwise it answers with the
// tool result's first line.
type fakeProxy struct {
	mu       sync.Mutex
	requests []map[string]any
	auth     []string
}

func (f *fakeProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var req map[string]any
	_ = json.Unmarshal(b, &req)
	f.mu.Lock()
	f.requests = append(f.requests, req)
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	f.mu.Unlock()
	msgs := req["messages"].([]any)
	last := msgs[len(msgs)-1].(map[string]any)
	var msg map[string]any
	switch {
	case last["role"] == "tool":
		first := strings.SplitN(last["content"].(string), "\n", 2)[0]
		msg = map[string]any{"role": "assistant", "content": "The notes say: " + first}
	case strings.Contains(last["content"].(string), "notes"):
		msg = map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{
			"id": "c1", "type": "function", "function": map[string]any{"name": "read_file", "arguments": `{"path":"notes.txt"}`}}}}
	default:
		msg = map[string]any{"role": "assistant", "content": "You said: " + last["content"].(string)}
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
	if strings.TrimSpace(out) != "The notes say: 1: deploy on fridays" {
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

func TestChat(t *testing.T) {
	srv := httptest.NewServer(&fakeProxy{})
	defer srv.Close()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	dir := workspace(t)
	in := strings.NewReader("first\n/usage\nread the notes\n/reset\n/exit\n")
	out, stderr, err := run(t, in, "chat", "--provider", "openai", "--base-url", srv.URL, "--model", "m", "--dir", dir)
	if err != nil {
		t.Fatalf("chat: %v %s", err, stderr)
	}
	for _, want := range []string{"You said: first", "input tokens 10, output tokens 2", "The notes say: 1: deploy on fridays", "Started a new conversation."} {
		if !strings.Contains(out, want) {
			t.Fatalf("chat output missing %q:\n%s", want, out)
		}
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
