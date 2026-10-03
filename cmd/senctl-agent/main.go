// Command senctl-agent is a small tool-using LLM agent for the terminal.
package main

import (
	"os"

	"github.com/eysteinn/senctl-agent/internal/cli"
)

func main() {
	os.Exit(cli.Execute())
}
