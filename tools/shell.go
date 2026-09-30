package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"

	"github.com/eysteinn/senctl-agent/agent"
	"github.com/eysteinn/senctl-agent/llm"
)

const maxShellOutput = 30000

// Shell returns a tool that runs shell commands in dir with a timeout. It
// can change files and reach the network, so only offer it when the user
// asked for it. Approve, when set, is asked before each command; returning
// false refuses it.
func Shell(dir string, timeout time.Duration, approve func(command string) bool) agent.Tool {
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	return agent.Tool{
		Spec: llm.ToolSpec{
			Name:        "shell",
			Description: fmt.Sprintf("Run a shell command (sh -c) in the workspace. Times out after %s. Returns exit code, stdout and stderr.", timeout),
			Schema:      object(map[string]any{"command": prop("string", "The command line")}, "command"),
		},
		Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
			in, err := decode[struct{ Command string }](raw)
			if err != nil {
				return "", err
			}
			if in.Command == "" {
				return "", fmt.Errorf("command is required")
			}
			if approve != nil && !approve(in.Command) {
				return "", fmt.Errorf("the user declined to run this command")
			}
			ctx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			cmd := exec.CommandContext(ctx, "sh", "-c", in.Command)
			cmd.Dir = dir
			// Background children can hold the pipes open after sh is
			// killed; stop waiting for them shortly after.
			cmd.WaitDelay = time.Second
			var out, errb bytes.Buffer
			cmd.Stdout, cmd.Stderr = &out, &errb
			runErr := cmd.Run()
			code := 0
			if runErr != nil {
				if ee, ok := runErr.(*exec.ExitError); ok {
					code = ee.ExitCode()
				} else {
					return "", runErr
				}
			}
			if ctx.Err() == context.DeadlineExceeded {
				return "", fmt.Errorf("command timed out after %s", timeout)
			}
			res := fmt.Sprintf("exit code %d\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
			if len(res) > maxShellOutput {
				res = res[:maxShellOutput] + "\n[output cut]"
			}
			return res, nil
		},
	}
}
