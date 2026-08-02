# Unified AgentReforge Agent Workflow

## 1. Purpose

This document proposes a unified workflow architecture for AgentReforge.

The central idea is:

> AgentReforge should have one deterministic orchestrator, use the general coding
> agent as its implementation worker, support autonomous and supervised governance,
> and improve code through bounded, evidence-driven recursive cycles.

The general coding agent should not need to understand the complete self-improvement
process. It receives a concrete coding task and works inside an isolated Git
worktree. The AgentReforge orchestrator owns analysis, policy, approval, versioning,
verification, and the decision to continue or stop.

## 2. Design Goals

The unified workflow should:

1. Replace the separate top-level orchestration loops with one orchestration model.
2. Reuse the existing ReAct coding agent as the implementation worker.
3. Support exactly two governance modes:
   - autonomous;
   - supervised, with a human in the loop.
4. Permit automatic recursive improvement, subject to explicit budgets and stop
   conditions.
5. Validate every LLM-generated task graph deterministically before execution.
6. Evaluate actual artifacts such as diffs and test results, rather than relying on
   a worker's textual summary.
7. Use Git commits, branches, and worktrees as the versioning and isolation system.
8. Keep final acceptance deterministic wherever executable checks are available.

## 3. High-Level Workflow

```text
User intent
    -> Analyze source, trajectories, evaluations, and current version
    -> Produce ImprovementProposal and acceptance contract
    -> Validate proposal and task DAG
    -> Apply policy and, in supervised mode, request human approval
    -> Create a Git branch and worktree from the pinned base commit
    -> Design and freeze the evaluation artifact
    -> Give concrete tasks to the general coding agent
    -> Evaluate the actual diff and run frozen checks
    -> Accept or reject the candidate version
    -> Decide whether new evidence justifies another improvement cycle
```

The corresponding control flow is:

```mermaid
flowchart TD
    U["User intent"] --> O["AgentReforge Orchestrator"]
    O --> A["Analyze"]
    A --> P["Proposal and acceptance contract"]
    P --> PV["Validate proposal and task DAG"]
    PV --> G{"Policy and approval"}
    G -->|deny| S["Abstain or stop"]
    G -->|allow| W["Create Git worktree"]
    W --> D["Design and freeze evaluation"]
    D --> E["General coding agent executes"]
    E --> V["Verify diff and frozen checks"]
    V -->|fail| R{"Retry budget remains?"}
    R -->|yes| E
    R -->|no| S
    V -->|pass| C["Record verified version"]
    C --> N{"New evidence justifies another cycle?"}
    N -->|yes| A
    N -->|no| F["Finish"]
```

### Workflow terminology

- **Recursive Run** is the largest execution unit. It owns one branch and one
  worktree for its full lifetime and may contain multiple Loops.
- **Loop** is one Orchestrator -> Writer/Reviewer -> Delivery cycle. It selects
  exactly one Candidate, executes exactly one coherent capability Task, and ends
  with exactly one accepted commit or a recorded failure/abstention.
- **Candidate** is one ranked potential improvement identified by the
  Orchestrator. Each Loop selects one evidence-backed, currently unsolved Candidate.
- **Task** is the Loop's one independently implemented and reviewed unit. It may
  touch multiple files for a coherent capability but keeps one objective.
- **Review Round** is one bounded Writer -> Reviewer attempt for a Task.
- **Delivery** is the Loop-level runnable gate. It runs frozen checks, judges
  whether the selected Candidate is realized, verifies the candidate Git tree
  is unchanged by delivery checks, and accepts the one Loop commit.

## 4. One Orchestrator

"One orchestrator" means one component owns workflow state transitions, task
scheduling, failure propagation, approval checkpoints, version transitions, and
recursive-cycle decisions.

It does not mean one unconstrained LLM should act as planner, writer, reviewer,
policy engine, and final verifier. Deterministic control and LLM reasoning remain
separate.

### Orchestrator information contract

The Orchestrator receives two histories that must never be conflated:

- **Target-agent trajectory:** the original task prompt, observable target-agent
  tool calls/results, final response, errors, and outcome. This is diagnostic
  evidence for deciding what capability to improve. Each run records its
  `target_commit`, `evidence_source`, and whether it describes the current Loop base.
- **AgentReforge trajectory:** one record per improvement loop, grouped under a
  stable recursive-run id. It contains the Orchestrator diagnosis/proposal,
  Writer task attempts, Reviewer findings, policy decisions, Deliverer result,
  loop diff, and commit. This is the audit trail and feedback for later loops.

A deterministic `OrchestratorContextBuilder` always supplies current target
runs, current repository structure, the recursive-run manifest, and prior loops
from the same run. Historical retrieval is deliberately secondary: SQLite FTS
may find analogous older loops, but it never replaces direct current-run facts.

The initial target trajectories are bound to the Recursive Run's base commit.
After a Loop is delivered, its frozen Scenario runs are normalized into new
target-agent runs with `evidence_source=delivered_scenario` and the delivered commit,
then appended to the next Loop's context. Older baseline runs remain inspectable but
become `is_current=false`; they cannot prove that a newly delivered capability is
still ineffective.

Capability-level achievement de-duplication remains an Orchestrator reasoning
responsibility. It compares mechanisms and expected capability deltas rather than
Candidate-name equality. A renamed refinement of a delivered capability requires new
current-commit evidence; stale baseline behavior is not sufficient justification.

The Orchestrator is one logical component but does not force all reasoning into
one model call. Its three typed stages use clean contexts and communicate through
durable artifacts:

```text
Deterministic Context Compiler
  -> Evidence Triage
       input: current alerts, compact run summaries, achievements, repository map
       output: DiagnosisBoard (alert dispositions + agent-level ProblemCases)
  -> Candidate Selection
       input: DiagnosisBoard, backlog, one-Loop budget, repository summary
       output: SelectionDecision (Candidates + scorecards + Top-2 + frozen winner)
  -> Contract Expansion
       input: frozen winner, relevant evidence references, repository map
       output: ContractExpansion (one bounded contract + frozen scenarios)
  -> deterministic coordinator joins SelectionDecision + ContractExpansion
       into the final ImprovementProposal
```

The stages do not share an accumulating chat history. They exchange typed artifacts,
and detailed target events remain behind a read-only drill-down tool. Triage must
account for every current terminal alert, but it is free to defer one with an
evidence-based reason. The task workspace's missing domain feature is a symptom:
ProblemCases and Candidates must describe reusable target-agent capability and may
only name target-repository components. Schema or completeness failures are repaired
inside the producing stage rather than terminating the Loop.

Selection cannot be changed during Contract Expansion or later acceptance repair.
The coordinator reattaches the frozen `SelectionDecision` deterministically, so
contract formatting and scenario revisions cannot switch to an easier Candidate.
Partial Diagnosis/Selection artifacts are persisted even if a later stage fails.

The scorecards are structured LLM deliberation, not a deterministic weighted
score or policy gate. Problem severity and evidence are assessed before
intervention cost so a cheap speculative change does not outrank a current
terminal failure merely because it is easy to implement. Evaluability distinguishes
mechanism compliance from a real outcome delta: the Orchestrator predicts baseline
and candidate behavior, identifies the observable difference and confounders, and
uses those facts in a qualitative Top-2 review. A highly observable but trivial
change must not win solely because it is easy to demonstrate.

A possible top-level interface is:

```python
class AgentReforgeOrchestrator:
    analyzer: Analyzer
    code_agent: CodeAgent
    evaluator: EvaluationSystem
    policy: PolicyEngine
    approval: ApprovalProvider
    versions: GitVersionManager
    recursion: RecursionPolicy

    async def run(self, request: ImproveRequest) -> ImproveRun:
        ...
```

The existing `plan_execute`, multi-agent orchestrator, and improvement pipeline
should eventually become strategies or internal services of this orchestrator,
rather than independent top-level workflow loops.

## 5. General Coding Agent as the Worker

The existing ReAct coding agent should be the implementation engine used by
AgentReforge.

AgentReforge adds analysis and governance before it, and verification after it:

```text
AgentReforge = Analyze + Govern + General Code Agent + Verify + Recurse
```

The coding agent receives a normalized task such as:

```python
class CodeTask(BaseModel):
    id: str
    goal: str
    cwd: str
    dependencies: list[str]
    constraints: list[str]
    acceptance_criteria: list[AcceptanceCriterion]
    evidence: list[Evidence]
    editable_paths: list[str]
    denied_paths: list[str]
```

Its responsibility is limited to concrete implementation work:

```text
read -> reason -> call tools -> edit -> run checks -> report evidence
```

The coding agent does not decide whether its own version should be accepted or
whether AgentReforge should recurse again.

### 5.1 Shared Writer/Reviewer task contract

The Orchestrator freezes one task contract for both implementation and review. It
must not generate separate Writer and Reviewer interpretations of the same goal.

```python
class ContractClause(BaseModel):
    id: str
    description: str


class ImprovementTask(BaseModel):
    id: str
    candidate: str
    description: str
    rationale: str
    capability_change: str
    required_behaviors: list[ContractClause]
    implementation_constraints: list[ContractClause]
    invariants: list[ContractClause]
    prohibited_shortcuts: list[ContractClause]
    affected_components: list[str]
    reviewer_focus: list[str]
    required_safety_properties: list[Literal["path_confinement"]]
    dependencies: list[str]
    acceptance_criteria_ids: list[str]
```

The Writer's implementation hand-off is the task-scoped Git diff and current
repository state. Writer prose is an optional note with no schema and is never
treated as evidence. Changed files come from Git; verification comes from the
Reviewer and AcceptanceRunner rather than self-reported commands or clause claims.

The Reviewer judges the Loop's frozen Task contract and its authoritative diff,
including cross-file integration within that one capability. It must not demand a
pristine worktree or confuse expected product changes with runtime artifacts.
Runnable delivery and pre/post tree immutability belong to Delivery.

```text
Orchestrator -> frozen task contract -> Writer -> task-scoped diff
                         \----------------------> Reviewer
Reviewer needs_fix -> evidenced blocking findings -> Writer
```

Only task id, objective, and valid dependencies are required. Candidate ownership,
clauses, affected components, acceptance hints, and safety-property declarations
remain available when they clarify the work but are not mandatory schema gates.
Suggested components do not prevent the Writer from making a necessary integration
change elsewhere inside the repository worktree.

## 6. Two Governance Modes

The workflow supports exactly two user-facing governance modes.

```python
class GovernanceMode(StrEnum):
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"
```

### 6.1 Autonomous mode

The deterministic policy engine returns `allow` or `deny`. Allowed actions proceed
without asking a human. Denied actions stop, abstain, or fail according to the
workflow state.

Autonomous does not mean unrestricted. Path restrictions, command restrictions,
budgets, frozen evaluations, and acceptance gates still apply.

### 6.2 Supervised mode

The same policy engine runs first. Hard-denied actions remain denied. At configured
checkpoints, otherwise permitted actions require human approval.

Useful checkpoints include:

- approval of the improvement proposal;
- approval of protected-path changes;
- approval of unusually risky commands;
- approval before accepting or merging a verified version;
- approval before starting another recursive cycle.

### 6.3 Explicit approval providers

Governance must be explicit rather than inferred from whether an approval callback
happens to be present.

```python
class ApprovalProvider(Protocol):
    async def authorize(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


class AutoApprovalProvider:
    """Uses policy decisions without prompting a human."""


class HumanApprovalProvider:
    """Uses policy decisions and prompts at supervised checkpoints."""
```

The same provider should be available throughout the run so that proposal approval,
tool execution, protected paths, merge decisions, and recursion follow consistent
rules.

## 7. Bounded Recursive Improvement

Automatic recursion should be modeled as a sequence of improvement cycles, not as
an unlimited LLM loop.

```python
class RecursionPolicy(BaseModel):
    max_cycles: int = 3
    max_failed_cycles: int = 1
    max_total_tokens: int = 200_000
    max_wall_time_seconds: int = 3600
    require_measurable_gain: bool = True
    stop_on_no_new_evidence: bool = True
```

Another cycle may begin only when new evidence shows that the original intent is
not yet fully satisfied. Examples include:

- a benchmark still misses its target;
- a trajectory exposes a remaining error;
- a required metric has improved but has not reached its threshold;
- a verified change exposes a new, directly related blocker;
- the previous cycle intentionally deferred a task required by the original goal.

The workflow must stop when:

- the original intent and required criteria are satisfied;
- there is no new evidence supporting another change;
- the expected gain is below policy thresholds;
- a hard policy rule denies continuation;
- the failure, token, time, or cycle budget is exhausted;
- supervised mode does not receive approval to continue.

The LLM should not be asked to invent another improvement merely to keep the loop
running.

## 8. Task DAG Validation and Execution

Every LLM-generated task graph must be validated before any task runs.

```python
class PlanValidationResult(BaseModel):
    valid: bool
    duplicate_ids: list[str]
    unknown_dependencies: list[str]
    cycles: list[list[str]]
    unreachable_tasks: list[str]
    criteria_without_tasks: list[str]
    tasks_without_criteria: list[str]
```

Validation must check at least:

- task IDs are non-empty and unique;
- every dependency references an existing task;
- the graph is acyclic;
- the graph contains at least one runnable root task;

Acceptance assignment and suggested file scope are agent planning concerns rather
than Task-plan validity conditions.

Runtime task states should be explicit:

```text
pending -> running -> completed
                   -> failed
pending            -> blocked
pending            -> skipped
```

If a task fails, dependent tasks should be marked `blocked` with the failed
dependency recorded. An empty runnable batch is successful only when every task is
completed or intentionally skipped. Otherwise it is a deadlock or blocked plan.

### 8.1 Concurrency

Read-only independent tasks may run concurrently. Tasks that write to the same
worktree should run serially unless the orchestrator can prove their declared write
sets do not overlap.

A future implementation may isolate parallel writers in separate worktrees and use
an integration step, but shared-worktree concurrent writing should not be the
default.

## 9. Evaluation Instead of a Summary-Only Reviewer

The current Reviewer and Test Agent overlap because both translate natural-language
requirements into quality judgments. They should be unified under an evaluation
subsystem, while preserving separation between evaluation design and final
acceptance.

```python
class EvaluationSystem:
    designer: EvaluationAgent
    verifier: DeterministicVerifier
```

### 9.1 Evaluation design before implementation

The Evaluation Agent should:

- translate acceptance criteria into generated tests, benchmarks, invariants, or
  manual checks;
- inspect real source interfaces before generating tests;
- reject criteria that are ambiguous or not verifiable;
- create an evaluation artifact;
- freeze and hash that artifact before product implementation begins.

### 9.2 Change review after implementation

The Evaluation Agent may also inspect:

- the actual diff;
- changed paths;
- tool and command results;
- test coverage gaps;
- structural risks not represented by generated tests.

It may produce structured findings and request another bounded implementation
round. It must not be the sole final acceptance authority.

### 9.3 Minimal deterministic verification

The deterministic Delivery/Commit Gate records only:

- safe execution of one minimal package/CLI smoke command;
- explicit system-owned safety-property commands, when declared;
- denial of destructive commands and generated runtime artifacts;
- confirmation that verification did not mutate the candidate;
- equality of the verified and committed Git trees.

Orchestrator-generated functional criteria, exact natural-language output markers,
lint, benchmarks, and optional regression checks are Reviewer evidence by default.
They do not become commit-blocking merely because an LLM marked them `required`.

### 9.4 Deliverer boundary

The Deliverer is one conceptual component with two internal parts:

1. **Runner:** executes minimal end-to-end/smoke commands, explicit safety checks,
   and capability-specific scenarios frozen by the Orchestrator before the Writer
   runs. Each scenario supplies a target-agent prompt, safe argv command, isolated
   fixture workspace, expected/forbidden behaviors, and optional typed executable
   conditions declared by the Orchestrator. The trusted Runner materializes those
   conditions without accepting arbitrary environment variables or setup scripts.
   It records command, exit code, bounded output, changed artifacts, observed
   environment facts, and a JSONL target-agent trajectory when
   `requires_trajectory` is frozen. The Pipeline separately compares pre/post
   candidate tree snapshots.
2. **Delivery Judge:** receives the Runner records, frozen Orchestrator goal,
   selected Candidate and causal mechanism, high-level requirements, and complete
   Loop diff. It decides whether the observed runnable behavior realizes the goal
   and classifies a rejection as `implementation_defect`, `verification_gap`,
   `plan_gap`, or `environment_failure`.

The Delivery Judge does not repeat function-level code review or targeted unit
tests and cannot override a deterministic Runner failure. It may interpret actual
output only as bounded evidence for goal realization; broader generated-code
quality remains a post-run capability evaluation.

The physical composition is:

```text
Pipeline
  -> DeliveryCoordinator
       -> AcceptanceRunner.run(proposal, cwd) -> AcceptanceRun
       -> Deliverer.review(proposal, loop_diff, AcceptanceRun) -> typed GoalReview
       -> Delivery(passed = acceptance.passed and goal.accepted)
```

`DeliveryCoordinator` is the Deliverer facade: the Runner directly serves the
Delivery Judge, while retaining an independently enforceable deterministic verdict.
Worktree mutation/integrity remains a Pipeline responsibility because it owns
candidate snapshots and Git isolation; it is not a Writer Task or Reviewer judgment.

Failure routing follows the diagnosed cause rather than a generic retry:

| Delivery classification | Next action |
| --- | --- |
| `implementation_defect` | Return the concrete scenario evidence to Writer for one bounded repair attempt. |
| `verification_gap` | Do not edit product code; end the Loop and expose the missing evidence to the next Orchestrator decision. |
| `plan_gap` | End the Loop; the next Orchestrator must choose a different cause or intervention. |
| `environment_failure` | End the Loop without pretending that product behavior was judged. |

The first integration surface is a generic CLI command with `{prompt}` and
`{workspace}` placeholders. Targets may emit richer trajectory evidence to the
path in `AGENTREFORGE_TRAJECTORY_PATH`; a broader target-adapter layer can be added
without changing the Orchestrator/Deliverer contract.

For the demo target, Runner automatically resolves `python -m demo_agent` to a
system-owned adapter. The adapter executes the candidate worktree's real
`run_task`, wraps its real tool dispatcher, and emits ordered `tool_result` and
`done` events. Baseline diagnosis and post-change Delivery therefore use the same
trajectory semantics; baseline events are never reused as proof of improved behavior.

Scenario evidence is intentionally typed by observability. Final output and changed
artifacts can prove externally visible outcomes, but a target's statement that it
called a tool or verified its work is only self-report. A scenario that judges tool
use, ordering, inspect-before-edit, or verify-after-edit sets
`requires_trajectory=true`; missing trajectory then deterministically produces
`verification_gap` before the LLM Judge runs.

When a capability depends on runtime availability, the Orchestrator may freeze
`executable_conditions` with only a command name and `available|unavailable`
state. Runner constructs a scenario-local `PATH`, verifies the resulting facts,
and includes them in Delivery evidence. Such scenarios require trajectory so the
Judge can connect the declared environment to the target agent's actual tool path.
If Runner cannot materialize the environment, the result is
`environment_failure`, never a request for Writer to change product code.

Plain `delivery_run` does not support Scenario placeholders. A safety criterion uses
`mode=invariant` but supplies no generated command: safety probes are system-owned.
For demo-agent path confinement, Runner creates an outside sentinel, asks the target
tool dispatcher to read it through `../`, and succeeds only when the real tool blocks
the traversal without exposing the sentinel.

Safety requirements are opt-in rather than global invariants. A Task may declare
`path_confinement` only when its selected Candidate implements or changes a
path-taking/filesystem boundary, and it must reference a matching invariant safety
criterion. An unrelated Candidate cannot be rejected because the baseline lacks a
different safety capability; that capability belongs in its own future Loop. If the
Delivery Judge accepts the selected goal while an incompatible safety probe fails,
the result is a `plan_gap`, not an environment failure or an instruction to Writer.

Every failed Loop records `failure_kind` plus an `attempt_fingerprint` covering the
selected Candidate and frozen verification strategy. These form the
Negative-Attempt Ledger. Proposal validation rejects an unchanged failed strategy,
and after a missing-trajectory `verification_gap` it rejects the same Candidate with
the same unavailable trajectory requirement. The Orchestrator must change the
evidence mechanism, select a different supported Candidate, or abstain.
It also rejects the same Candidate carrying a safety requirement that already failed,
even if the model cosmetically rewrites its Scenario.

Reviewer and Deliverer schema failures are returned to the same producer for
output-only repair. They must never be converted into product-code repair
instructions for another Agent. Writer has no output schema: its authoritative
handoff is the Git diff.

### 9.5 Avoid correlated self-approval

It is acceptable to reuse the same implementation class or model provider for test
design and change review, but they should use separate phase contexts and frozen
artifacts. The component that designed a test must not be able to silently rewrite
that test after seeing the implementation.

## 10. Git-Based Versioning and Isolation

AgentReforge uses Git as the version system. A separate content-addressed snapshot
system is not required for the core workflow.

Git provides:

- a pinned base commit;
- an isolated branch;
- a separate worktree;
- stable checkpoints through commits;
- diffs and changed-path inspection;
- rollback and version comparison;
- optional merge into the user's branch.

Each delivered Loop should record:

```python
class ImprovementVersion(BaseModel):
    loop: int
    base_commit: str
    branch: str
    verified_commit: str | None
    proposal_hash: str
    evaluation_hash: str
```

The version sequence should look like:

```text
Recursive Run branch at V0
  -> Loop 0: frozen proposal -> Task changes -> Delivery -> commit V1
  -> Loop 1 analyzes V1 plus LoopOutcome 0
  -> Loop 1: frozen proposal -> Task changes -> Delivery -> commit V2
```

One Recursive Run creates one branch/worktree from a pinned base commit. All of
its Loops run serially there, and every delivered Loop contributes exactly one
commit. A failed Loop is reset to its own base commit without discarding earlier
delivered Loop commits.

Only a delivered commit may become the base of the next Loop. Whether the final
Recursive Run branch is merged into the user's active branch is a separate
governance decision.

## 11. Runtime, Threads, and Runs

Conversation state and workflow state should be separate.

```python
class Thread(BaseModel):
    id: str
    messages: list[Message]
    events: list[Event]
    run_ids: list[str]


class ImproveRun(BaseModel):
    id: str
    thread_id: str | None
    intent: str
    governance_mode: GovernanceMode
    status: RunStatus
    current_cycle: int
    artifacts: list[ArtifactReference]
    versions: list[ImprovementVersion]
```

`Thread.messages` supplies model history for later turns. `Thread.events` is the UI
and audit stream. `ImproveRun` stores resumable workflow state, approvals, artifacts,
cycles, and Git versions.

A service restart should not erase active runs. Persistence may initially use
SQLite, with append-only events plus materialized run state.

## 12. Shared Run Context

Dependencies currently passed individually through many functions should be grouped
into a shared context.

```python
class RunContext:
    run_id: str
    cycle: int
    cwd: str
    client: LlmClient
    registry: ToolRegistry
    governance_mode: GovernanceMode
    policy: PolicyEngine
    approval: ApprovalProvider
    memory: MemoryManager | None
    code_index: CodeIndex | None
    event_sink: EventSink
    artifact_store: ArtifactStore
    budget: RunBudget
    cancellation: CancellationToken
```

This prevents plan, team, runtime, and improvement paths from accidentally omitting
approval, trajectory logging, memory, event collection, or budgets.

## 13. Suggested Orchestrator Algorithm

```python
async def run(request: ImproveRequest, ctx: RunContext) -> ImproveRun:
    base_commit = await versions.resolve_head(request.target)
    version = await versions.create_recursive_run_worktree(
        run_id=ctx.run_id,
        base_commit=base_commit,
    )

    for loop in recursion.iter_loops(ctx.budget):
        loop_base = await version.head()
        loop_ctx = ctx.for_loop(loop, loop_base)

        evidence = await analyzer.collect_evidence(
            intent=request.intent,
            commit=loop_base,
            previous_loop_outcomes=run.loop_outcomes,
            ctx=loop_ctx,
        )
        diagnosis = await analyzer.triage(request.intent, evidence, loop_ctx)
        selection = await analyzer.select(diagnosis, loop_ctx)
        if selection.decision == "abstain":
            return run.converged(selection.decision_reason)
        contract = await analyzer.expand_contract(selection, loop_ctx)
        proposal = assemble_proposal(diagnosis, selection, contract)

        proposal_validation = validate_proposal(proposal)
        plan_validation = validate_task_dag(proposal.tasks)
        if not proposal_validation.valid or not plan_validation.valid:
            return run.blocked("invalid proposal or task graph")

        authorization = await authorize_proposal(proposal, loop_ctx)
        if not authorization.allowed:
            return run.abstained(authorization.reasons)

        evaluation = await evaluator.design(proposal, version, loop_ctx)
        frozen = freeze(proposal, evaluation, loop_base)

        execution = await code_agent.execute_plan(
            proposal.tasks,
            frozen=frozen,
            cwd=version.worktree_path,
            ctx=loop_ctx,
        )

        verification = await evaluator.verify(
            frozen=frozen,
            execution=execution,
            version=version,
            ctx=loop_ctx,
        )

        if not verification.passed:
            retry = recursion.retry_decision(verification, loop_ctx.budget)
            if retry.allowed:
                execution = await code_agent.repair(retry.findings, loop_ctx)
                verification = await evaluator.verify(
                    frozen=frozen,
                    execution=execution,
                    version=version,
                    ctx=loop_ctx,
                )
            if not verification.passed:
                await version.reset(loop_base)
                return run.failed(verification)

        loop_commit = await versions.commit_delivered_loop(version, verification)
        run.record_loop_outcome(loop, loop_commit, verification)

        continuation = recursion.should_continue(
            original_intent=request.intent,
            previous_evidence=evidence,
            verification=verification,
            budget=loop_ctx.budget,
        )
        if not continuation.allowed:
            return run.completed(loop_commit, continuation.reason)

        if ctx.governance_mode == GovernanceMode.SUPERVISED:
            approval = await ctx.approval.authorize(continuation.as_request())
            if not approval.allowed:
                return run.completed(loop_commit, "human stopped recursion")

    return run.completed(await version.head(), "recursion budget exhausted")
```

## 14. Proposed Module Boundaries

One possible package structure is:

```text
agentreforge/
  orchestration/
    orchestrator.py
    state_machine.py
    run_context.py
    task_executor.py
    plan_validator.py

  agent/
    code_agent.py
    query.py

  analysis/
    analyzer.py
    evidence.py
    proposal.py

  evaluation/
    designer.py
    verifier.py
    models.py
    runner.py

  governance/
    policy.py
    approval.py
    permissions.py
    budgets.py

  versioning/
    git_versions.py
    worktree.py

  runtime/
    threads.py
    runs.py
    events.py
    persistence.py

  tools/
  llm/
  memory/
  rag/
  mcp/
  entrypoints/
```

This is a target structure, not a requirement for a single large refactor. Existing
modules can be migrated incrementally.

## 15. Incremental Migration Plan

### Phase 1: Correctness foundations

1. Add deterministic task-DAG validation.
2. Add `blocked` propagation for failed dependencies and deadlock detection.
3. Make governance mode and approval provider explicit.
4. Ensure all tool execution paths receive the same policy and approval context.
5. Persist real thread message history across Runtime API turns.

### Phase 2: Unified execution kernel

1. Introduce `RunContext`.
2. Extract a reusable task-graph executor.
3. Configure serial or parallel scheduling as a policy.
4. Convert plan execution and team execution into executor configurations.
5. Make the improvement workflow use the same coding-agent worker interface.

### Phase 3: Unified evaluation

1. Replace summary-only review with artifact-based evaluation.
2. Unify Test Agent and Reviewer under `EvaluationSystem`.
3. Freeze the evaluation artifact before implementation.
4. Implement a deterministic final verifier.
5. Prohibit the writer from changing frozen evaluation files.

### Phase 4: Versioned recursive runs

1. Add `ImprovementVersion` records.
2. Create one branch/worktree for the complete Recursive Run.
3. Add explicit recursion budgets and stop conditions.
4. Produce exactly one commit per delivered Loop and use it to seed the next Loop.
5. Persist runs so recursive work can be inspected and resumed.

### Phase 5: Remove obsolete orchestration paths

After behavior is covered by tests and migrated to the unified orchestrator:

1. remove duplicated plan/team loops;
2. remove implicit callback-based governance behavior;
3. retire unused snapshot code if Git versioning fully covers the intended use case;
4. update the README and design documentation to describe the implemented workflow.

## 16. Core Architectural Invariant

The intended boundary is:

> The general coding agent changes code. The AgentReforge orchestrator decides why a
> change is justified, whether it is allowed, how it must be verified, whether the
> candidate version is accepted, and whether another bounded improvement cycle is
> warranted.

This preserves autonomous recursive improvement without making the workflow an
unbounded self-approval loop.
