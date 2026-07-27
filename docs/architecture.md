# AgentReforge architecture and reading guide

## Recommended reading order

1. `README.md` — product scope and quick start.
2. `docs/unified-agent-workflow.md` — terminology, lifecycle, and invariants.
3. `metaimprove/improve/pipeline.py` — authoritative state transitions.
4. `metaimprove/improve/models.py` — shared proposal and Task contracts.
5. `metaimprove/improve/orchestrator.py` — diagnosis and Batch planning.
6. `metaimprove/improve/writer_reviewer.py` and `reviewer.py` — Task execution.
7. `delivery_coordinator.py`, `acceptance_runner.py`, and `deliverer.py` —
   Batch-level delivery.

`docs/self-improving-agent-design.md` is an earlier design document. Use the
unified workflow and current code when the two differ.

## Package ownership

### `metaimprove/improve/`

The AgentReforge product workflow:

- `pipeline.py` owns Recursive Run and Loop state transitions.
- `orchestrator.py` produces evidence-grounded Improvement Batches.
- `models.py` defines the frozen proposal, Candidate, Task, and acceptance schema.
- `plan_validator.py` validates Task dependency graphs.
- `policy_gate.py` authorizes proposed and actual write scope.
- `writer_reviewer.py` schedules Tasks and bounded Review Rounds.
- `reviewer.py` performs independent Task-level review.
- `acceptance.py` validates acceptance contracts before implementation.
- `acceptance_runner.py` executes frozen deterministic checks.
- `deliverer.py` judges full-Batch goal realization from the complete diff.
- `delivery_coordinator.py` combines deterministic and LLM delivery evidence.
- `worktree.py` owns Git isolation, snapshots, commits, and rollback.
- `context.py`, `records.py`, and `history_index.py` provide current-run feedback
  and durable audit history.
- `trajectory.py` stores target-agent execution evidence separately from
  AgentReforge's own workflow records.

### General agent infrastructure

- `metaimprove/agent/` — ReAct query loop and older plan/reviewer adapters.
- `metaimprove/orchestration/` — shared dependency-aware Task executor.
- `metaimprove/tools/` — tool registry, built-ins, and execution.
- `metaimprove/llm/` — provider-neutral LLM interfaces and structured parsing.
- `metaimprove/policy/` — command/path guards and audit logging.
- `metaimprove/rag/` and `memory/` — local code retrieval and SQLite memory.
- `metaimprove/mcp/` — external MCP tool discovery.
- `metaimprove/runtime/` — FastAPI thread/turn/event API.

## Runtime artifacts

AgentReforge writes target-specific runtime data under the target repository's
`.meta-improve/` directory:

```text
.meta-improve/
  records/<run_id>/
    run.json
    loops/loop_<n>/
      record.json
      diff.patch
  reports/
  worktrees/
  improvement_history.db
  orchestrator_code_index.db
```

These are generated audit artifacts, not source files, and should normally remain
Git-ignored.

## Version topology

```text
target main at V0
  \
   improve/<run> branch + one worktree
      Loop 0 -> delivered commit V1
      Loop 1 -> delivered commit V2
      ...
      converge / stop / fail
```

The target's active branch remains unchanged unless the user explicitly chooses
`--merge`. Keeping the baseline branch unchanged is useful for before/after demos.

## Evaluation directories

- `eval/mini/` is a deterministic four-case brownfield smoke suite.
- `eval/` contains the larger experimental greenfield suite and optional LLM
  grading.
- Neither evaluation is part of the Delivery hard gate for an improved target.
