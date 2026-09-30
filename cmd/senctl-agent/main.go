// Command senctl-agent is a small tool-using LLM agent for the terminal.
package main

import (
	"os"

	"github.com/eysteinn/senctl-agent/internal/cli"
)

func main() {
	if err := cli.NewRootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}
