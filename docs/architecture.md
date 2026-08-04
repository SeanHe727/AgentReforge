# AgentReforge architecture and reading guide

## Recommended reading order

1. `README.md` — product scope and quick start.
2. `docs/unified-agent-workflow.md` — terminology, lifecycle, and invariants.
3. `docs/agent-interface-contracts.md` — every agent input/output schema.
4. `agentreforge/improve/pipeline.py` — authoritative state transitions.
5. `agentreforge/improve/models.py` — shared proposal and Task contracts.
6. `agentreforge/improve/orchestrator.py` — diagnosis and single-Candidate planning.
7. `agentreforge/improve/writer_reviewer.py` and `reviewer.py` — Task execution.
8. `delivery_coordinator.py`, `acceptance_runner.py`, and `deliverer.py` —
   the Loop-level Deliverer facade, deterministic Runner, and runtime Judge.

`docs/self-improving-agent-design.md` is an earlier design document. Use the
unified workflow and current code when the two differ.

## Package ownership

### `agentreforge/improve/`

The AgentReforge product workflow:

- `pipeline.py` owns Recursive Run and Loop state transitions.
- `orchestrator.py` selects one evidence-grounded capability Candidate per Loop.
- `models.py` defines the frozen proposal, Candidate, Task, and acceptance schema.
- `plan_validator.py` validates Task dependency graphs.
- `policy_gate.py` routes explicit proceed/abstain/needs-human intent and rejects
  generated runtime artifacts; suggested file scope is not an authorization gate.
- `writer_reviewer.py` schedules Tasks and bounded Review Rounds.
- `reviewer.py` performs independent Task-level review. Read operations inspect the
  candidate; shell verification runs in a fresh disposable copy with a writable
  scratch directory, and any candidate/eval mutation invalidates that command.
- `acceptance.py` validates minimal delivery commands and hand-off references.
- `acceptance_runner.py` is the Deliverer's deterministic Runner for smoke and
  explicit safety commands plus Orchestrator-frozen target-agent scenarios in
  isolated fixture workspaces. It records process output, changed artifacts, and
  a JSONL target trajectory when the frozen scenario requires process evidence.
- `demo_agent_adapter.py` runs the demo target in a fresh subprocess, wraps its
  actual model/tool dispatcher, and emits normalized ordered component inputs,
  outputs, tool calls/results, and separate total-turn/action-budget facts. The same
  adapter owns a deterministic path-confinement probe over an outside sentinel.
- `deliverer.py` owns four bounded phases: Scenario Readiness, Runner invocation,
  per-Scenario Evidence Analysis, and final Goal Realization over the evidence card
  plus Loop diff.
- `delivery_coordinator.py` is the Deliverer facade and combines both decisions;
  deterministic Runner failures cannot be overridden by the Judge. Its classified
  root cause controls whether the Writer may receive a bounded repair attempt.
- `worktree.py` owns Git isolation, snapshots, commits, and rollback.
- `context.py`, `records.py`, and `history_index.py` provide current-run feedback
  and durable audit history, including delivered achievements and failed-attempt
  fingerprints. Target runs carry their observed commit and source so the next
  Orchestrator can distinguish current delivered evidence from stale baselines.
- `trajectory.py` stores the target agent's full observable JSONL action history
  (bounded component inputs/outputs, linked tool calls/results, usage, errors,
  budgets, and completion) outside the target-visible Task Workspace
  separately from AgentReforge's own workflow records.

### General agent infrastructure

- `agentreforge/agent/` — ReAct query loop and older plan/reviewer adapters.
- `agentreforge/orchestration/` — shared dependency-aware Task executor and
  output-only hand-off repair.
- `agentreforge/tools/` — tool registry, built-ins, and execution.
- `agentreforge/llm/` — provider-neutral LLM interfaces and structured parsing.
- `agentreforge/policy/` — command/path guards and audit logging.
- `agentreforge/rag/` and `memory/` — local code retrieval and SQLite memory.
- `agentreforge/mcp/` — external MCP tool discovery.
- `agentreforge/runtime/` — FastAPI thread/turn/event API.

## Runtime artifacts

AgentReforge writes target-specific runtime data under the target repository's
`.agentreforge/` directory:

```text
.agentreforge/
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

## Structured agent hand-offs

System prompts remain readable instructions, but agent-to-agent requests and
decisions are typed JSON rather than prose protocols. No control decision depends
on finding words such as `APPROVED`, `finished`, or `verified` in free text.

| Producer -> consumer | Structured input | Structured output / authoritative artifact |
| --- | --- | --- |
| User -> Planner | `PlannerRequest` JSON | `PlanSpec` JSON |
| Planner -> ordinary Worker | task/dependency JSON envelope | `WorkerHandoff` JSON; tool effects are execution evidence |
| ordinary Worker -> Reviewer | Writer handoff + task-scoped diff JSON envelope | `ReviewerDecision` JSON |
| Workflow Analysis -> coordinator | prior Reforge Loop records only | `WorkflowDiagnosisBoard` JSON |
| Target run -> Case Analysis | one run summary + read-only complete target trajectory | `CaseAnalysis` JSON |
| Case coordinator -> Target Analysis | `CaseAnalysisBoard` + filtered backlog/source | `DiagnosisBoard` JSON |
| Selection -> Contract Expansion | frozen winner/context JSON | `ContractExpansion` JSON |
| Contract Expansion -> Scenario Definition | filtered semantic contract + evidence; no repository paths | `DeliveryCaseDefinition` JSON |
| Scenario Definition -> Scenario Materialization | frozen case + target repository launch context | `DeliveryExecutionDesign` JSON |
| Coordinator -> Writer Planning | compact `WriterTaskContract` + read-only repository | `WriterPlan` JSON |
| Writer -> Reviewer | compact contract + parsed handoff + authoritative task diff | `ReviewerOutput` / structured `ReviewResult` |
| Reviewer -> Writer repair | stable-ID findings + retained worktree state | `FindingResolution` records plus updated authoritative diff |
| Orchestrator -> Scenario Readiness | frozen capability, Scenario, fixture and environment JSON | `ScenarioReadinessOutput` JSON |
| Runner -> Scenario Evidence Analysis | execution, artifacts, environment and trajectory JSON | `ScenarioEvidenceCard` JSON |
| Evidence Analysis -> Goal Realization | proposal, contract, diff, run evidence and evidence card JSON | `DeliveryReviewOutput` JSON |
| Any malformed producer -> handoff repair | validation error, output contract, immutable context and prior output JSON | repaired instance of the same producer schema |

The Scenario Task Contract is itself Pydantic data. Its Outcome Contract separates:

- `primary_success`: positive results that all need supporting evidence;
- `guardrails`: prohibited regressions that can veto delivery but cannot prove it;
- `inconclusive`: observations showing that the run cannot establish delivery.

Every outcome condition carries an ID, description, rationale, and
`evidence_direction`. The target agent receives a stable fielded rendering of this
data; its action stream is captured separately as JSONL. Git diffs and command output
remain opaque evidence fields rather than being mistaken for control messages.

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
- Neither evaluation is part of the delivery/commit gate for an improved target.
