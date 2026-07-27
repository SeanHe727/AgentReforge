# mini-agent demo target

This deliberately weak rule-based command agent is a reproducible AgentReforge
target. It has a real CLI and obvious reusable improvement opportunities.

```bash
python3 -m mini_agent "echo hello" "add 2 3" "reverse abc"
```

Initial commands:

- `echo <text>`
- `add <a> <b>`
- `reverse <text>`

Known gaps include malformed-input crashes, no help/discovery command, terse
unknown-command output, and no persistent command registry.
