# AgentReforge

AgentReforge is a controlled recursive coding-agent workflow that improves an
existing agent system from its source code, execution trajectory, and a user
goal.

It is not an unrestricted “LLM edits itself forever” loop. Each improvement is
planned, implemented, reviewed, tested, isolated in Git, and either committed or
rolled back.

## What is implemented

- A general ReAct coding agent with local tools, SQLite memory/code index, MCP,
  and a FastAPI runtime.
- One evidence-grounded improvement Orchestrator.
- Two governance modes: `autonomous` and `supervised`.
- Recursive Runs with one branch/worktree for their full lifetime.
- One commit per delivered Loop.
- One coherent capability Task per Loop, with a validated single-node Task plan.
- Typed, output-repairable Orchestrator, Reviewer, and Deliverer hand-offs;
  Writer hands off its authoritative Git diff with only an optional text note.
- Two minimal hard-gate families: hand-off validity and delivery/commit safety.
- Task-level code review and Loop-level runnable Delivery with frozen usage scenarios.
- A demo-agent runtime adapter that normalizes real target tool calls into Delivery
  trajectory evidence.
- Persistent run/loop/component records for audit and next-loop feedback.

## Workflow

```text
User intent + target trajectory + current source
    -> Orchestrator diagnoses and ranks Candidates
    -> exclude completed achievements and Delivery-only reward hacks
    -> select one unsolved Candidate and create one bounded Task
    -> validate proposal schema, IDs, references, and the Task plan
    -> optional human approval
    -> Writer implements the Task
    -> Reviewer returns typed blocking/non-blocking findings
    -> malformed hand-offs are rewritten by their producing output module
    -> Deliverer Runner executes frozen smoke/safety commands and target-agent scenarios
    -> capture scenario output, generated artifacts, and target trajectory when required
    -> Delivery Judge assesses actual run evidence against the Loop goal
    -> only observed implementation defects may return to Writer for repair
    -> verification/plan/environment failures end the Loop and inform the next one
    -> reject repeated Candidate + verification strategies via the Negative-Attempt Ledger
    -> verify the candidate and committed Git trees match the verified tree
    -> one Loop commit
    -> feed LoopOutcome and delivered Scenario trajectory to the next Loop
    -> distinguish current-commit evidence from stale baseline evidence, or converge
```

Terminology and detailed invariants are documented in
[the unified workflow](docs/unified-agent-workflow.md).

## Quick start

```bash
uv sync --extra dev
export OPENAI_API_KEY=...

# Run the ordinary coding agent.
uv run agent-reforge -p "inspect this repository and summarize the architecture"

# Improve another Git repository.
uv run agent-reforge improve \
  --cwd /path/to/target-agent \
  --intent "Improve repository inspection and verification behavior" \
  --model gpt-5.4-mini \
  --mode autonomous \
  --loops 3 \
  --keep
```

AgentReforge does not modify the target's active branch during the run. It creates
an isolated `improve/...` branch/worktree and keeps it for inspection unless
`--merge` is explicitly requested.

## Inspect an actual improvement

The separate
[AgentReforge demo agent](https://github.com/SeanHe727/AgentReforge-demo)
keeps the original coding agent on `main` and the accepted AgentReforge output
on `improve/agentreforge`.

Review the real branch diff instead of relying on a benchmark score:

```bash
git clone https://github.com/SeanHe727/AgentReforge-demo.git
cd AgentReforge-demo
git diff main...improve/agentreforge
```

The comparison shows the bounded capability change, including the generated
prompt and tool-safety changes. It is an inspectable demonstration, not a claim
of universal coding-capability improvement.

## Repository map

```text
agentreforge/agent/          ordinary ReAct/plan execution components
agentreforge/improve/        AgentReforge recursive improvement workflow
                            including target adapters and system-owned safety probes
agentreforge/orchestration/  shared Task executor
                            and output-only handoff repair
agentreforge/tools/          local tool contracts and execution
agentreforge/policy/         command/path safety
agentreforge/rag/            local code index
agentreforge/memory/         SQLite memory
agentreforge/runtime/        FastAPI runtime
eval/mini/                  deterministic early-version smoke benchmark
eval/                       larger experimental coding evaluation
tests/                      Workflow and component regression tests
docs/                       architecture and design documentation
scripts/                    local project utilities
```

See [docs/architecture.md](docs/architecture.md) for module ownership and the
recommended reading order.

## Evaluation scope

The deterministic mini suite currently establishes a clean non-regression result,
not a universal coding-capability improvement claim. Agent workflow evaluation is
high variance and existing model benchmarks do not isolate orchestration quality.
For this early version, the primary artifact is therefore the inspectable recursive
demo: proposals, Task reviews, acceptance evidence, diffs, and one commit per Loop.

## Development

```bash
uv run pytest
uv run ruff check agentreforge eval tests --exclude eval/solutions
```

## License

MIT
