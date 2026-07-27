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
- **Loop**, also called an **Improvement Batch**, is one
  Orchestrator -> Writer/Reviewer -> Delivery cycle. It selects a bounded set of
  one to three Candidates and ends with exactly one accepted commit or a recorded
  failure/abstention.
- **Candidate** is one ranked potential improvement identified by the
  Orchestrator. A large Candidate normally occupies a Batch alone; two or three
  small compatible Candidates may share a Batch even when their objectives are
  independent.
- **Task** is an independently implemented and reviewed unit owned by exactly one
  selected Candidate. One large Candidate may be split into several Tasks.
- **Review Round** is one bounded Writer -> Reviewer attempt for a Task.
- **Delivery** is the Batch-level integration gate. It runs frozen checks, judges
  whether every selected Candidate is realized, verifies the candidate Git tree
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
  evidence for deciding what capability to improve.
- **AgentReforge trajectory:** one record per improvement loop, grouped under a
  stable recursive-run id. It contains the Orchestrator diagnosis/proposal,
  Writer task attempts, Reviewer findings, policy decisions, Deliverer result,
  loop diff, and commit. This is the audit trail and feedback for later loops.

A deterministic `OrchestratorContextBuilder` always supplies current target
runs, current repository structure, the recursive-run manifest, and prior loops
from the same run. Historical retrieval is deliberately secondary: SQLite FTS
may find analogous older loops, but it never replaces direct current-run facts.

The Orchestrator follows a fixed reasoning workflow:

```text
Orient -> Diagnose symptom/root cause/capability gap
       -> Generate interventions at multiple leverage levels
       -> Rank Candidates by evidence/benefit/risk/effort
       -> Pack 1-3 compatible Candidates into an Improvement Batch
       -> Explain each causal mechanism and the packing decision
       -> Plan tasks and validate the DAG/acceptance contract
```

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

The Writer implements every clause and returns a validated `WriterReport` containing
the task id, changed files, per-clause implementation evidence, commands and results,
known limitations, and deviations. A malformed report or one that omits a contract
clause is rejected before LLM review. The Reviewer receives that same contract,
validated report, and task-scoped diff, then audits every required behavior and
invariant with code or runtime evidence. A rejection cites the failed clause id and
a concrete required fix.

The Reviewer is Task-scoped: it judges the frozen Task contract and that Task's
diff. It must not demand a pristine worktree or confuse expected product changes
with runtime artifacts. Cross-Task compatibility, full-Batch goal realization,
and pre/post tree immutability belong to Delivery.

```text
Orchestrator -> frozen task contract -> Writer -> task-scoped diff
                         \----------------------> Reviewer
Reviewer REVISE -> clause-specific feedback -> Writer
```

The proposal validator rejects a task before writing begins when its rationale,
capability change, required behaviors, invariants, reviewer focus, or traceable
clause ids are missing. This prevents a strategically sound proposal from degrading
into a vague implementation hand-off.

As a lightweight safety floor, a Task that changes path-taking tools must declare
`path_confinement`. One assigned executable acceptance criterion must carry the
same property, attempt a relative `..` traversal, and assert a stable rejection
marker. The Task Reviewer repeats that negative check rather than accepting a
“read-only” or “confined” comment as evidence. This is intentionally a minimum
guard, not a comprehensive filesystem-security suite.

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
- required acceptance criteria are assigned to tasks;
- tasks that can modify code have an allowed scope.

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

### 9.3 Deterministic verification

The final verifier should execute and record:

- frozen generated tests;
- existing regression tests;
- lint and type checks defined by the target;
- before/after benchmarks;
- permission and changed-path checks;
- confirmation that the writer did not edit the frozen evaluation artifact;
- required acceptance-criterion outcomes.

Final acceptance should be derived from these results according to explicit rules.
An LLM review is supporting evidence, not the only hard gate.

### 9.4 Deliverer boundary

The delivery stage has two distinct parts:

1. A deterministic contract runner executes the frozen full-Batch commands,
   evaluates exit codes and declared output assertions, and the Pipeline compares
   the pre/post candidate tree snapshots.
2. The Deliverer LLM receives the frozen Orchestrator goals, all selected
   Candidates and causal mechanisms, high-level requirements, and the complete
   Loop diff. It decides whether every Candidate is realized and whether the
   combined Batch integrates successfully.

The Deliverer does not re-run per-task review, interpret command output into
unsupported PASS claims, or grade code generated by the target agent. Those
belong to the task Reviewer, deterministic runner, and post-run capability
evaluation respectively.

The physical composition is:

```text
Pipeline
  -> DeliveryCoordinator
       -> AcceptanceRunner.run(proposal, cwd) -> AcceptanceRun
       -> Deliverer.review(proposal, loop_diff) -> GoalReview
       -> Delivery(passed = acceptance.passed and goal.accepted)
```

Neither leaf component depends on the other. `DeliveryCoordinator` is the only
component that combines their decisions. Worktree mutation/integrity remains a
Pipeline responsibility because it owns candidate snapshots and Git isolation;
it is not a Writer Task or Reviewer judgment.

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
        proposal = await analyzer.propose(request.intent, evidence, loop_ctx)

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
