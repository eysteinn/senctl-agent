package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/eysteinn/senctl-agent/llm"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeProxy is a Responses API endpoint. It lists two models; when
// the conversation has no tool result yet it asks to read notes.txt (or to
// write hello.txt when asked to create a file), otherwise it answers with
// the tool result's first line. It streams when asked to.
type fakeProxy struct {
	models   []string // default beta, alpha
	mu       sync.Mutex
	requests []map[string]any
	auth     []string
}

func (f *fakeProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/models") {
		f.mu.Lock()
		models := f.models
		f.mu.Unlock()
		if models == nil {
			models = []string{"beta", "alpha"}
		}
		var data []map[string]string
		for _, m := range models {
			data = append(data, map[string]string{"id": m})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
		return
	}
	b, _ := io.ReadAll(r.Body)
	var req map[string]any
	_ = json.Unmarshal(b, &req)
	f.mu.Lock()
	f.requests = append(f.requests, req)
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	f.mu.Unlock()
	items := req["input"].([]any)
	last := items[len(items)-1].(map[string]any)
	var text string
	var output []any
	call := func(name, args string) {
		output = append(output, map[string]any{"type": "function_call", "id": "fc1", "call_id": "c1", "name": name, "arguments": args})
	}
	switch {
	case last["type"] == "function_call_output":
		text = "The tool says: " + strings.SplitN(last["output"].(string), "\n", 2)[0]
	case strings.Contains(last["content"].(string), "notes"):
		call("read_file", `{"path":"notes.txt"}`)
	case strings.Contains(last["content"].(string), "everything"):
		call("pipeline", `{"path":"big.txt","command":"grep line"}`)
	case strings.Contains(last["content"].(string), "create"):
		call("write_file", `{"path":"hello.txt","content":"hi\n"}`)
	default:
		text = "You said: " + last["content"].(string)
	}
	if text != "" {
		output = append([]any{map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": text}}}}, output...)
	}
	resp := map[string]any{"status": "completed", "output": output, "usage": map[string]any{"input_tokens": 10, "output_tokens": 2}}
	if req["stream"] == true {
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(v map[string]any) {
			b, _ := json.Marshal(v)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", v["type"], b)
		}
		if text != "" {
			half := len(text) / 2
			send(map[string]any{"type": "response.output_text.delta", "delta": text[:half]})
			send(map[string]any{"type": "response.output_text.delta", "delta": text[half:]})
		}
		send(map[string]any{"type": "response.completed", "response": resp})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
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
	for _, tl := range proxy.requests[0]["tools"].([]any) {
		if name := tl.(map[string]any)["name"]; name == "shell" || name == "write_file" {
			t.Fatalf("run offered %s by default", name)
		}
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

	// Large piped input is saved to the session cache instead, and the cache
	// is gone when the run ends.
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	r, w, _ = os.Pipe()
	go func() {
		for i := 1; i <= 5000; i++ {
			fmt.Fprintf(w, "log line %d\n", i)
		}
		w.Close()
	}()
	out, _, err = run(t, r, "run", "summarize", "--model", "m", "--dir", dir)
	if err != nil || !strings.Contains(out, "too large to include (68893 bytes, 5000 lines)") || !strings.Contains(out, "saved at "+cacheHome) ||
		!strings.Contains(out, "It begins:\nlog line 1\n") || strings.Contains(out, "log line 5000") {
		t.Fatalf("large stdin out = %q, %v", out, err)
	}
	if left, _ := os.ReadDir(filepath.Join(cacheHome, "senctl-agent", "sessions")); len(left) != 0 {
		t.Fatalf("session cache left behind: %v", left)
	}

	if _, _, err := run(t, nil, "run", "x", "--dir", dir); err == nil || !strings.Contains(err.Error(), "the endpoint offers alpha, beta") {
		t.Fatalf("missing model err = %v", err)
	}
	if out, _, err := run(t, nil, "models"); err != nil || out != "alpha\nbeta\n" {
		t.Fatalf("models = %q, %v", out, err)
	}

	// With only a proxy URL and key, its one model is used.
	proxy.mu.Lock()
	proxy.models = []string{"only-model"}
	proxy.mu.Unlock()
	if out, _, err := run(t, nil, "run", "hello", "--dir", dir); err != nil || strings.TrimSpace(out) != "You said: hello" {
		t.Fatalf("out = %q, %v", out, err)
	}
	if last := proxy.requests[len(proxy.requests)-1]["model"]; last != "only-model" {
		t.Fatalf("model = %v", last)
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
	if err := os.WriteFile(filepath.Join(dir, "facts.md"), []byte("the sky is blue\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	saved := filepath.Join(t.TempDir(), "conv.md")
	script := strings.Join([]string{
		"/help",
		"/model",
		"/model nope",
		"/model alpha",
		"hello \\",
		"there",
		"please create a file",
		"y",
		"read the notes",
		"/edit off",
		"/tools",
		"/effort",
		"/effort extreme",
		"/effort high",
		"/usage",
		"what is in @facts.md?",
		"/save " + saved,
		"/clear",
		"/bogus",
		"/exit",
	}, "\n") + "\n"
	out, stderr, err := run(t, strings.NewReader(script), "--base-url", srv.URL, "--model", "m", "--dir", dir)
	if err != nil {
		t.Fatalf("console: %v %s", err, stderr)
	}
	for _, want := range []string{
		"/model [id]",               // help
		"Model: m", "alpha", "beta", // model listing
		"Unknown model nope; the provider offers:", // unlisted model refused
		"Switched to alpha",                        // model switch
		"You said: hello \nthere",                  // continued line, streamed reply
		"Create file? hello.txt", "+ hi",           // edit approval with preview
		"The tool says: Created hello.txt (1 line).",
		"The tool says: 1: deploy on fridays",
		"edit is now off",
		"Effort: default", "minimal", "xhigh", // effort listing
		"Unknown effort extreme",
		"Effort set to high.",
		"input tokens 50, output tokens 10", // 5 model calls
		"You said: what is in @facts.md?\n\n<file path=\"facts.md\">\nthe sky is blue\n</file>", // @mention
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
	// Requests after /effort high carry it.
	last := proxy.requests[len(proxy.requests)-1]
	if r, _ := last["reasoning"].(map[string]any); r["effort"] != "high" {
		t.Fatalf("last request reasoning = %v", last["reasoning"])
	}
	toolNames := func(req map[string]any) string {
		var names []string
		for _, tl := range req["tools"].([]any) {
			names = append(names, tl.(map[string]any)["name"].(string))
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
	out, _, err := run(t, strings.NewReader("create it\nn\n/exit\n"), "--base-url", srv.URL, "--model", "m", "--dir", dir)
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
	yaml := "provider: anthropic\nmodel: from-file\nbase_url: http://file/v1\napi_key: file-key-123456\nmax_turns: 7\neffort: low\nshell: ask\n"
	if err := os.WriteFile(filepath.Join(cfgDir, "senctl-agent", "config.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	config := func(args ...string) string {
		t.Helper()
		out, _, err := run(t, nil, append([]string{"config"}, args...)...)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	expect := func(out string, want ...string) {
		t.Helper()
		for _, w := range want {
			if !strings.Contains(out, w) {
				t.Fatalf("config output missing %q:\n%s", w, out)
			}
		}
	}
	// Each setting: flag, then environment, then file, then default.
	t.Setenv("SENCTL_AGENT_MODEL", "from-env")
	t.Setenv("SENCTL_AGENT_EFFORT", "high")
	t.Setenv("SENCTL_AGENT_BASE_URL", "http://env/v1")
	t.Setenv("OPENAI_API_KEY", "generic-key-000000")
	expect(config("--base-url", "http://flag/v1", "--effort", "medium"),
		"provider:    anthropic", "model:       from-env", "base_url:    http://flag/v1", "effort:      medium",
		"api_key:     file…3456", "max_turns:   7", "max_tokens:  16000", "shell:       ask", "config.yaml")
	expect(config("--api-key", "flag-key-abcdef", "--shell", "off"), "api_key:     flag…cdef", "effort:      high", "shell:       off", "base_url:    http://env/v1")
	t.Setenv("SENCTL_AGENT_API_KEY", "env-key-987654")
	expect(config(), "api_key:     env-…7654")

	// OPENAI_API_KEY / ANTHROPIC_API_KEY stand in only when no api_key is set
	// anywhere, matching the provider.
	empty := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", empty)
	t.Setenv("SENCTL_AGENT_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "anthropic-key-111111")
	expect(config(), "provider:    openai (default)", "api_key:     gene…0000")
	expect(config("--provider", "anthropic"), "api_key:     anth…1111")
}

func TestNoWorkingDirectoryConfig(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.WriteFile(localConfig, []byte("base_url: http://attacker.example/v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, stderr, err := run(t, nil, "config")
	if err != nil || strings.Contains(out, "attacker") || !strings.Contains(stderr, "ignoring ./.senctl-agent.yaml") {
		t.Fatalf("out = %q, stderr = %q, err = %v", out, stderr, err)
	}
	// Named explicitly, by flag or environment, it is used.
	if out, _, err := run(t, nil, "config", "--config", localConfig); err != nil || !strings.Contains(out, "base_url:    http://attacker.example/v1") {
		t.Fatalf("--config: out = %q, err = %v", out, err)
	}
	t.Setenv("SENCTL_AGENT_CONFIG", filepath.Join(dir, localConfig))
	if out, _, err := run(t, nil, "config"); err != nil || !strings.Contains(out, "base_url:    http://attacker.example/v1") {
		t.Fatalf("SENCTL_AGENT_CONFIG: out = %q, err = %v", out, err)
	}
}

func TestErrorReport(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, "{\n  \"error\": {\n    \"message\": \"Missing bearer authentication in header\",\n    \"type\": \"invalid_request_error\"\n  }\n}")
	}))
	defer srv.Close()
	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	models := httptest.NewServer(&fakeProxy{})
	defer models.Close()
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error": {"message": "The model `+"`gpt-nope`"+` does not exist or you do not have access to it.", "type": "invalid_request_error", "code": "model_not_found"}}`)
	}))
	defer api.Close()
	web := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, "<!DOCTYPE html>\n<html><head><title>404 Not Found</title></head><body>nope</body></html>")
	}))
	defer web.Close()
	tmp := t.TempDir()
	file := filepath.Join(tmp, "file.txt")
	badType := filepath.Join(tmp, "bad.yaml")
	for name, content := range map[string]string{file: "hi\n", badType: "max_turns: lots\n"} {
		if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("SENCTL_AGENT_API_KEY", "")

	for _, tt := range []struct {
		name string
		args []string
		code int
		want string
	}{
		{"no key", []string{"run", "x", "--base-url", srv.URL}, exitConfig,
			"senctl-agent: no API key set\n  set SENCTL_AGENT_API_KEY, api_key: in "},
		{"no key, no base URL", []string{"run", "x"}, exitConfig, "senctl-agent: no API key set\n"},
		{"rejected key", []string{"run", "x", "--base-url", srv.URL, "--api-key", "bad"}, exitConfig,
			"senctl-agent: the endpoint rejected the API key (HTTP 401: Missing bearer authentication in header)\n  check SENCTL_AGENT_API_KEY"},
		{"unreachable", []string{"run", "x", "--base-url", closed.URL, "--api-key", "k"}, exitConfig,
			"senctl-agent: cannot reach " + closed.URL + ": dial tcp"},
		{"several models", []string{"run", "x", "--base-url", models.URL, "--api-key", "k"}, exitConfig,
			"senctl-agent: no model given; the endpoint offers alpha, beta\n  set --model, SENCTL_AGENT_MODEL or model: in "},
		{"unknown model", []string{"run", "x", "--base-url", api.URL, "--api-key", "k", "--model", "gpt-nope"}, exitConfig,
			"senctl-agent: the endpoint does not serve this model (HTTP 404: The model `gpt-nope` does not exist or you do not have access to it.)\n  check --model, SENCTL_AGENT_MODEL or model: in "},
		{"web page", []string{"run", "x", "--base-url", web.URL, "--api-key", "k", "--model", "m"}, exitConfig,
			"senctl-agent: the endpoint did not answer like an LLM API (HTTP 404: a web page (\"404 Not Found\"), not an API response)\n  check --base-url, "},
		{"base URL without scheme", []string{"run", "x", "--base-url", "proxy.example.com", "--model", "m"}, exitConfig,
			"senctl-agent: base URL \"proxy.example.com\" must be an http:// or https:// address\n  check --base-url, SENCTL_AGENT_BASE_URL or base_url: in "},
		{"unknown provider", []string{"run", "x", "--provider", "gemini", "--api-key", "k"}, exitConfig,
			"senctl-agent: unknown provider \"gemini\"; use openai or anthropic\n  check --provider, "},
		{"bad fallbacks", []string{"run", "x", "--fallbacks", "maybe"}, exitConfig,
			"senctl-agent: fallbacks must be true or false, not \"maybe\"\n  check --fallbacks, "},
		{"missing workspace", []string{"run", "x", "--dir", filepath.Join(tmp, "missing"), "--model", "m"}, exitConfig,
			"senctl-agent: workspace " + filepath.Join(tmp, "missing") + " does not exist\n  check --dir, "},
		{"workspace is a file", []string{"run", "x", "--dir", file, "--model", "m"}, exitConfig,
			"senctl-agent: workspace " + file + " is not a directory\n  check --dir, "},
		{"missing config file", []string{"run", "x", "--config", filepath.Join(tmp, "nope.yaml")}, exitConfig,
			"senctl-agent: config file " + filepath.Join(tmp, "nope.yaml") + " does not exist\n  check --config or SENCTL_AGENT_CONFIG\n"},
		{"wrong type in config file", []string{"run", "x", "--config", badType}, exitConfig,
			"senctl-agent: max_turns must be a whole number, not \"lots\"\n  check --max-turns, SENCTL_AGENT_MAX_TURNS or max_turns: in "},
		{"bad flag value", []string{"run", "x", "--max-turns", "lots"}, exitUsage,
			"senctl-agent: invalid argument \"lots\" for \"--max-turns\" flag: not a whole number\n  run 'senctl-agent run --help' for usage\n"},
		{"unknown flag", []string{"run", "x", "--nope"}, exitUsage,
			"senctl-agent: unknown flag: --nope\n  run 'senctl-agent run --help' for usage\n"},
		{"bad setting", []string{"run", "x", "--shell", "sometimes"}, exitConfig,
			"senctl-agent: shell must be off, ask or auto, not \"sometimes\"\n  check --shell, SENCTL_AGENT_SHELL or shell: in "},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, stderr, err := run(t, nil, tt.args...)
			if err == nil {
				t.Fatal("no error")
			}
			if stderr != "" {
				t.Errorf("cobra printed %q itself", stderr)
			}
			var b bytes.Buffer
			if code := report(&b, err, false); code != tt.code || !strings.HasPrefix(b.String(), tt.want) {
				t.Errorf("report = %d\n%s\nwant %d\n%s", code, b.String(), tt.code, tt.want)
			}
		})
	}

	_, _, err := run(t, nil, "run", "x", "--base-url", srv.URL)
	var b bytes.Buffer
	report(&b, err, true)
	if !strings.Contains(b.String(), "\n  error: llm: no model given, and listing the endpoint's models failed: openai: list models: HTTP 401: ") {
		t.Errorf("debug report lacks the full error:\n%s", b.String())
	}
}

// When the provider cannot list its models, /model says why and switches
// without checking; /effort default goes back to the provider's default.
func TestConsoleUnlistedModels(t *testing.T) {
	proxy := &fakeProxy{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/models") {
			http.NotFound(w, r)
			return
		}
		proxy.ServeHTTP(w, r)
	}))
	defer srv.Close()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	script := "/model\n/model m\n/model other\n/effort low\n/effort default\nhi\n/exit\n"
	out, stderr, err := run(t, strings.NewReader(script), "--base-url", srv.URL, "--model", "m", "--dir", workspace(t))
	if err != nil {
		t.Fatalf("console: %v %s", err, stderr)
	}
	for _, want := range []string{
		"Model: m\nCannot list models: openai: list models: HTTP 404: 404 page not found",
		"Already using m.",
		"Cannot check the model (openai: list models: HTTP 404: 404 page not found); switching anyway.\nSwitched to other.",
		"Effort set to default.",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("console output missing %q:\n%s", want, out)
		}
	}
	if len(proxy.requests) != 1 || proxy.requests[0]["model"] != "other" || proxy.requests[0]["reasoning"] != nil {
		t.Fatalf("requests = %v", proxy.requests)
	}
}

// Tool output too large for the model is saved, and the console says so.
func TestConsoleSpillNotice(t *testing.T) {
	srv := httptest.NewServer(&fakeProxy{})
	defer srv.Close()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	dir := workspace(t)
	var b strings.Builder
	for i := 1; i <= 5000; i++ {
		fmt.Fprintf(&b, "line %d of a big file\n", i)
	}
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	out, stderr, err := run(t, strings.NewReader("show everything\n/exit\n"), "--base-url", srv.URL, "--model", "m", "--dir", dir)
	if err != nil {
		t.Fatalf("console: %v %s", err, stderr)
	}
	for _, want := range []string{
		"⏺ pipeline",
		"↳ pipeline output is 117 KB (5000 lines), too large to send whole: saved to ",
		"-output.txt; the model sees its start and end",
		"The tool says: [This output is", // the model got the preview, not the whole output
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("console output missing %q:\n%s", want, out)
		}
	}
}

func TestRequestTimingSettings(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	if err := os.MkdirAll(filepath.Join(cfgDir, "senctl-agent"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfgDir, "senctl-agent", "config.yaml"), []byte("max_retries: 3\nrequest_timeout: 2m\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SENCTL_AGENT_REQUEST_TIMEOUT", "90s")
	out, _, err := run(t, nil, "config")
	if err != nil || !strings.Contains(out, "request_timeout: 1m30s") || !strings.Contains(out, "max_retries: 3") {
		t.Fatalf("config: %v\n%s", err, out)
	}
	if out, _, _ := run(t, nil, "config", "--max-retries", "0", "--request-timeout", "10m"); !strings.Contains(out, "max_retries: 0") || !strings.Contains(out, "request_timeout: 10m0s") {
		t.Fatalf("flags: %s", out)
	}
	for _, args := range [][]string{{"--max-retries", "-1"}, {"--request-timeout", "0s"}} {
		if _, _, err := run(t, nil, append([]string{"config"}, args...)...); err == nil {
			t.Errorf("%v accepted", args)
		}
	}
}

func TestTimeoutIsExplained(t *testing.T) {
	msg, hint, code := explain(fmt.Errorf("openai: %w: no complete response within 5m0s", llm.ErrTimeout))
	if code != exitError || !strings.Contains(msg, "did not answer in time") || !strings.Contains(hint, "--request-timeout") {
		t.Fatalf("msg %q hint %q code %d", msg, hint, code)
	}
}
