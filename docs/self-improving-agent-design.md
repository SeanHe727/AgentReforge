# Evidence-Grounded Agent Improvement Pipeline

## 1. Purpose

This document defines a reliability-first pipeline for improving Python agent
systems. The initial target is AgentReforge itself, but the architecture must support
other runnable Python agent projects through an explicit target contract.

The system is not designed to maximize autonomous self-modification. Its goal is
to make agent improvement evidence-grounded, reviewable, measurable, reversible,
and capable of abstaining when the evidence or expected value is insufficient.

The core promise is:

> Convert source code, execution trajectories, evaluation results, and user
> intent into a frozen improvement proposal; implement it in isolation; then
> accept the change only when pre-approved before/after checks demonstrate the
> intended improvement without unacceptable regression.

## 2. Product Positioning

The project should be presented as a controlled agent-improvement pipeline, not
as an unrestricted recursively self-improving agent.

Its differentiators are:

- Evidence-grounded proposals rather than ungrounded reflection.
- Evaluation criteria defined and approved before implementation.
- Explicit benefit, risk, effort, confidence, and abstention decisions.
- A bounded Writer-Critic collaboration loop during implementation.
- Lightweight isolation through Git worktrees, with stronger sandboxing left as
  an optional deployment choice.
- Before/after validation against a frozen evaluation artifact.
- Deterministic policy gates, with LLM judgments used as evidence rather than as
  the sole authority.
- Support for both self-improvement and improvement of external Python agents.

## 3. System Inputs

A trustworthy proposal should use four input categories:

1. **Source**: project structure, implementation, prompts, tools, configuration,
   tests, and documentation.
2. **Trajectory**: model calls, tool calls, errors, retries, latency, token use,
   and other runtime behavior.
3. **Evaluation**: existing tests, benchmarks, scenario results, and operational
   metrics.
4. **User intent**: the requested improvement, scope, constraints, priorities,
   and acceptable risk.

Source code explains how a system is built. Trajectories show how it actually
behaves. Evaluations provide an external signal about whether that behavior is
acceptable. None of the three is sufficient by itself.

## 4. Supported Target Boundary

The first version targets complete, runnable Python agent pipelines. A target is
ready for analysis when it has:

- A reproducible Python environment.
- A documented agent entry point.
- Readable source code.
- At least one readable trajectory source.
- A command or adapter that can run the agent.

An existing benchmark is not required for analysis. A project without one can
still receive a proposed evaluation contract. However, a frozen evaluation
artifact is required before the system may claim that an applied change is an
improvement.

Initially out of scope:

- Multi-language monorepos.
- Agents that can only run against inaccessible production services.
- Projects with no reproducible execution environment.
- Components that cannot be invoked or observed independently.
- Fully autonomous merge or deployment to production.

Later versions may add adapters for partial systems such as tools, prompts, RAG
pipelines, planners, and evaluators.

## 5. Target Project Contract

Each external target should provide an `improve.yaml` contract. The contract
defines how the improvement engine may inspect, execute, modify, and evaluate
the project.

```yaml
project:
  name: example-agent
  root: .

inputs:
  trajectories:
    format: openinference
    paths:
      - traces/*.jsonl

environment:
  python: "3.12"
  install: uv sync

agent:
  entrypoint: uv run example-agent run

commands:
  test: uv run pytest
  lint: uv run ruff check .
  typecheck: uv run mypy src

permissions:
  default: readable
  editable:
    - src/**
    - tests/**
  read_only:
    - README.md
    - docs/**
  approval_required:
    - pyproject.toml
    - src/security/**
    - .github/workflows/**
  denied:
    - .env
    - "**/*.pem"
    - "**/*secret*"
    - production_data/**
  hidden:
    - .git/**
    - .venv/**

execution:
  network: deny
  timeout_seconds: 300
  approval_required_commands:
    - uv add *
    - git commit *
  denied_commands:
    - git push *
    - curl *
```

Permission precedence is:

```text
denied > approval_required > editable > readable > default
```

`editable` implies read access. `hidden` is an optional discovery rule: hidden
paths are omitted from project context, while `denied` remains the enforcement
rule at execution time.

Before analysis, a deterministic preflight check validates paths, commands,
permission conflicts, environment setup, trajectory parsing, and repository
state. The LLM must not guess missing contract information.

## 6. Core Artifacts

### 6.1 Improvement Proposal

An `ImprovementProposal` is an RFC-style, evidence-grounded change proposal. It
combines the intent of a specification, the implementation outline of a plan,
and the success conditions of an evaluation design.

Recommended sections:

1. Summary.
2. Problem and evidence.
3. Goals and non-goals.
4. Proposed change.
5. Affected components and dependencies.
6. Benefit, risk, effort, and confidence.
7. Implementation tasks and dependencies.
8. Acceptance criteria.
9. Evaluation plan.
10. Rollback plan.
11. Alternatives considered.
12. Approval decision.

Evidence must point to inspectable sources such as a trajectory span, test
result, benchmark run, code location, or log record.

```python
class Evidence(BaseModel):
    source_type: Literal["trajectory", "test", "benchmark", "code", "log"]
    reference: str
    observation: str


class ImprovementProposal(BaseModel):
    summary: str
    problem_statement: str
    evidence: list[Evidence]
    goals: list[str]
    non_goals: list[str]
    affected_components: list[str]
    dependencies: list[str]
    tasks: list[ImprovementTask]
    benefit: int
    risk: int
    effort: int
    confidence: float
    decision: Literal["proceed", "abstain", "needs_human"]
    decision_reason: str
    acceptance_criteria: list[AcceptanceCriterion]
    evaluation_plan: EvaluationPlan
    rollback_plan: str
    alternatives_considered: list[str]
```

LLM scores are decision inputs, not final policy. Deterministic rules may force
abstention or human review based on evidence quality, risk, permissions, affected
components, and the availability of executable acceptance criteria.

### 6.2 Frozen Proposal

After approval, the proposal is immutable and records:

```text
proposal_id
proposal_version
proposal_hash
approved_by
approved_at
target_commit
baseline_run_id
```

Goals, non-goals, evidence, permissions, acceptance criteria, and evaluation
rules must never change silently after approval. A substantial change creates a
new proposal version and requires approval and baseline evaluation again.

### 6.3 Evaluation Artifact

The evaluation artifact is generated and approved before implementation. It
contains executable checks, scenario definitions, metrics, expected outcomes,
and guardrail thresholds. It is hashed and frozen separately from the proposal.

Acceptance criteria declare their validation mode:

```python
class AcceptanceCriterion(BaseModel):
    id: str
    description: str
    mode: Literal[
        "red_green",
        "invariant",
        "metric_improvement",
        "non_regression",
        "manual",
    ]
    verification_method: Literal[
        "existing_test",
        "generated_test",
        "static_check",
        "benchmark",
        "manual",
    ]
    required: bool = True
```

Expected before/after behavior depends on the mode:

| Mode | Before | After |
| --- | --- | --- |
| `red_green` | Fails or is unsupported | Passes |
| `invariant` | Passes | Still passes |
| `metric_improvement` | Establishes baseline | Meets target improvement |
| `non_regression` | Establishes baseline | Does not exceed allowed regression |
| `manual` | Captured for comparison | Human-reviewed |

Metric-based proposals define a primary metric and guardrail metrics. An
efficiency improvement, for example, must not reduce task success below the
approved guardrail.

### 6.4 Execution Journal and Blockers

The approved intent is immutable, but runtime state is mutable. The execution
journal records task status, attempts, critic feedback, partial results,
commands, test results, and blockers.

```python
class ExecutionBlocker(BaseModel):
    task_id: str
    review_rounds: int
    unresolved_findings: list[Finding]
    attempted_fixes: list[str]
    affected_tasks: list[str]
    recommendation: Literal[
        "abstain_task",
        "amend_plan",
        "request_human",
        "abort_proposal",
    ]
```

When a task is blocked, dependency status is propagated. Independent tasks may
continue; dependent tasks become `dependency_blocked`. Required acceptance
criteria determine whether partial completion can be retained or the entire
proposal must be rejected.

## 7. Roles and Responsibilities

### Orchestrator

- Reads source, trajectory, evaluation, and user intent.
- Designs the Improvement Proposal.
- Defines acceptance criteria and the Evaluation Plan.
- Produces the implementation task DAG.
- Coordinates runtime state and responds to blockers.
- Does not write implementation code.
- Does not unilaterally declare the improvement successful.

### Test Planner and Test Agent

- The Test Planner translates approved acceptance criteria into a structured
  test plan.
- The Test Agent writes concrete tests or benchmark scripts from that plan.
- Generated tests must not modify the implementation under test.
- Test changes are reviewed separately from product-code changes.

### Implementation Writer

- Implements only the approved task and scope.
- Produces a checkpoint after each logical change unit.
- Maps every change to a task and acceptance criterion.
- Does not change the frozen proposal or evaluation artifact.

### Implementation Critic

- Reviews each logical change unit while implementation is in progress.
- Checks correctness, scope, maintainability, and proposal alignment.
- Returns structured `accept`, `revise`, or `escalate` feedback.
- Does not edit code and is not the final acceptance gate.

### Evaluation Runner

- Deterministically executes before/after tests, regression suites, static
  checks, and benchmarks.
- Records raw results without interpreting them through an LLM.

### Verifier and Policy Gate

- Maps evaluation evidence back to acceptance criteria.
- Applies deterministic acceptance, rejection, escalation, and abstention rules.
- Treats LLM review as a soft signal unless a human explicitly delegates a
  decision.

## 8. Writer-Critic Collaboration

Review occurs after a logical change unit, not after every line or token. A unit
may be a function, module, proposal task, or independently explainable diff.

Each Writer checkpoint includes:

- Task ID.
- Summary of the change.
- Current diff.
- Acceptance criteria addressed.
- Local checks already run.
- Known limitations or uncertainty.

The Critic returns structured findings with severity, location, evidence,
required fix, and related criterion. Review is bounded, for example to three
rounds per task.

If the limit is exceeded, the system creates an `ExecutionBlocker` rather than
continuing indefinitely. The Orchestrator may skip an optional task, continue
independent tasks, request human input, propose a plan amendment, or abort.

A small implementation-path adjustment that does not change scope, risk,
permissions, or evaluation may be represented as a `PlanAmendment`. Any change
to approved intent or success criteria requires a new proposal version.

## 9. Evaluation Bootstrap

An existing benchmark is optional for onboarding but an evaluation oracle is
mandatory before accepting an improvement.

Projects without tests or benchmarks use a separate bootstrap phase:

```text
User intent + source + trajectory
                ↓
Orchestrator proposes Evaluation Plan
                ↓
Test Agent creates tests/scenarios
                ↓
Human or independent review
                ↓
Run against original target to establish baseline
                ↓
Freeze Evaluation Artifact
                ↓
Begin implementation
```

The evaluator and target implementation must not be changed in the same
improvement cycle. Otherwise the system can make itself appear better by moving
the goalposts or weakening tests.

For nondeterministic agent behavior, before and after evaluation should use
multiple runs and report success rate, cost, latency, and variance. A small
initial run count is acceptable, provided the experiment structure supports
larger samples later.

## 10. Isolation and Safety

The first version uses a Git worktree and temporary branch:

```text
Stable host improvement engine
            ↓
Temporary target worktree and branch
            ↓
Writer changes only the candidate worktree
            ↓
Evaluation produces diff and report
            ↓
Accept, reject, or request human review
```

Worktrees provide change isolation and clean diffs, but they are not security
sandboxes. Commands can still access the host and network. The minimum local
safety baseline therefore includes:

- Path enforcement from the target contract.
- Command policy and HITL approval for side effects.
- Default denial of network access where enforceable.
- Secret and sensitive-content redaction from trajectories and logs.
- Timeouts and bounded retries.
- Snapshot or Git rollback.
- No automatic push, merge, or deployment.
- Structured audit logging.

Docker or another sandbox may be added as an optional stronger runtime without
changing the pipeline interfaces.

## 11. End-to-End Scenario

```text
1. User selects a target project and supplies improve.yaml and improvement intent.
2. Preflight validates environment, permissions, entry point, and trajectory inputs.
3. The target is run at a fixed commit to collect source, trajectory, and baseline evidence.
4. The Orchestrator diagnoses a candidate structural weakness.
5. It produces an evidence-grounded Improvement Proposal.
6. Policy evaluates evidence, benefit, risk, effort, confidence, and protected scope.
7. The system abstains, requests human review, or allows proposal review to continue.
8. The user or delegated reviewer approves the Proposal and Evaluation Plan.
9. Proposal, target commit, and Evaluation Artifact are frozen and hashed.
10. The Evaluation Runner executes the Before suite and stores immutable results.
11. A temporary Git worktree is created from the approved target commit.
12. The Writer implements one logical change unit.
13. The Critic reviews the checkpoint and accepts, requests revision, or escalates.
14. Bounded Writer-Critic rounds continue task by task.
15. A blocker is reported to the Orchestrator if a task cannot converge.
16. The Orchestrator propagates dependency status and chooses continue, amend,
    abstain, escalate, or abort without changing frozen success criteria.
17. After implementation, the Evaluation Runner executes the same frozen suite.
18. Existing regression tests, static checks, generated tests, and fixed scenarios
    produce an Evaluation Report.
19. The Verifier maps code and test evidence to every acceptance criterion.
20. The Policy Gate accepts only if all required criteria and guardrails pass.
21. The user receives the Proposal, execution journal, diff, Evaluation Report,
    critic findings, cost, and final decision.
22. Accepted changes remain available for explicit merge; rejected changes are
    discarded or retained only as an auditable failed candidate.
23. Failures and blockers become trajectory evidence for a future improvement
    cycle rather than causing an unbounded repair loop in the current proposal.
```

## 12. Operating Modes

### Analyze

Requires a runnable project and readable trajectory. Produces grounded findings
and an Improvement Proposal, but does not modify code.

### Assisted Improve

Uses user-approved acceptance criteria and a generated evaluation artifact.
Implements in an isolated worktree and returns a diff and report.

### Benchmarked Improve

Uses an existing regression suite and fixed benchmark in addition to proposal-
specific generated tests. Supports the strongest automated acceptance policy.

The same engine supports two target relationships:

- **Self mode**: a stable AgentReforge instance improves a candidate AgentReforge worktree.
- **External mode**: AgentReforge improves another compatible Python agent project.

The host improvement engine and target worktree remain separate in both modes.

## 13. Initial Benchmark Scope

The first benchmark should verify the pipeline rather than claim broad coding
capability. Initial scenarios may cover:

- Detecting repeated or inefficient tool calls from a trajectory.
- Grounding a proposal in a concrete tool error.
- Abstaining when evidence is insufficient.
- Escalating a protected or high-risk modification.
- Producing executable acceptance criteria.
- Demonstrating a red-green before/after fix.
- Preserving an existing regression suite.
- Propagating a blocked task through its dependency graph.

Later evaluations should measure proposal precision, evidence validity,
abstention calibration, regression rate, accepted-change success rate, net
improvement, cost efficiency, and rollback rate. Ablations should compare a
baseline agent, reflection-only behavior, proposal plus evaluation, and the full
pipeline with abstention and policy gates.

## 14. Delivery Sequence

1. Target contract and deterministic preflight validation.
2. Structured trajectory store with redaction and adapters.
3. Improvement Proposal, evidence, acceptance criteria, and policy models.
4. Analyze-only Reflection/Orchestrator flow.
5. Evaluation bootstrap and frozen before/after artifacts.
6. Git worktree lifecycle and audit journal.
7. Bounded Writer-Critic implementation loop.
8. Evaluation Runner, Verifier, and deterministic Policy Gate.
9. End-to-end assisted improvement demo on AgentReforge.
10. External Python agent adapter and reproducible example project.
11. Benchmark suite, ablations, documentation, and stronger optional sandboxing.

## 15. Design Invariants

The implementation must preserve these invariants:

- No accepted improvement without inspectable evidence.
- No implementation before proposal and evaluation approval.
- No silent mutation of frozen goals, permissions, or success criteria.
- No evaluator weakening in the same cycle as target modification.
- No unbounded Writer-Critic or self-repair loop.
- No LLM judgment as the sole hard acceptance signal.
- No modification of the stable host process in place.
- No automatic merge, push, or deployment in the initial release.
- Every final decision must be explainable from proposal, policy, code diff, and
  evaluation evidence.
