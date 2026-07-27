# AgentReforge

AgentReforge is a controlled recursive coding-agent workflow that improves an
existing agent system from its source code, execution trajectory, and a user
goal.

It is not an unrestricted “LLM edits itself forever” loop. Each improvement is
planned, implemented, reviewed, tested, isolated in Git, and either committed or
rolled back.

> The installable Python package and CLI still use the historical name
> `metaimprove` / `meta-improve`. AgentReforge is the public project name.

## What is implemented

- A general ReAct coding agent with local tools, SQLite memory/code index, MCP,
  and a FastAPI runtime.
- One evidence-grounded improvement Orchestrator.
- Two governance modes: `autonomous` and `supervised`.
- Recursive Runs with one branch/worktree for their full lifetime.
- One commit per delivered Loop.
- Bounded Improvement Batches containing one large Candidate or up to three
  small compatible Candidates.
- Candidate-owned Tasks with a frozen Writer/Reviewer contract.
- Deterministic proposal, DAG, policy, write-scope, acceptance, command, and
  delivery-integrity gates.
- Task-level agentic review and Batch-level Delivery.
- Persistent run/loop/component records for audit and next-loop feedback.

## Workflow

```text
User intent + target trajectory + current source
    -> Orchestrator diagnoses and ranks Candidates
    -> pack one Improvement Batch
    -> validate proposal, Task DAG, safety, and acceptance contract
    -> policy gate / optional human approval
    -> Writer implements each Task
    -> Reviewer checks each frozen Task contract
    -> DeliveryCoordinator runs deterministic checks and reviews the full diff
    -> verify the candidate Git tree is unchanged
    -> one Loop commit
    -> feed LoopOutcome to the next Loop, or converge
```

Terminology and detailed invariants are documented in
[the unified workflow](docs/unified-agent-workflow.md).

## Quick start

```bash
uv sync --extra dev
export OPENAI_API_KEY=...

# Run the ordinary coding agent.
uv run meta-improve -p "inspect this repository and summarize the architecture"

# Improve another Git repository.
uv run meta-improve improve \
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
metaimprove/agent/          ordinary ReAct/plan execution components
metaimprove/improve/        AgentReforge recursive improvement workflow
metaimprove/orchestration/  shared Task executor
metaimprove/tools/          local tool contracts and execution
metaimprove/policy/         command/path safety
metaimprove/rag/            local code index
metaimprove/memory/         SQLite memory
metaimprove/runtime/        FastAPI runtime
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
uv run ruff check metaimprove eval tests --exclude eval/solutions
```

## License

MIT
