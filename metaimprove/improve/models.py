"""Data contract for the evidence-grounded improvement pipeline.

These Pydantic models are the shared vocabulary every stage speaks (design doc
sections 6.1-6.4). They encode the reliability-first decisions:
  - proposals must cite inspectable Evidence,
  - acceptance criteria declare HOW they're verified (mode + method),
  - benefit/risk/effort/confidence + an explicit proceed/abstain/needs_human
    decision (LLM scores are inputs; a deterministic policy gate decides),
  - a frozen+hashed proposal so approved intent can't silently drift.

Models are ordered so dependencies come before the models that use them (no
forward references needed).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- primitives -----------------------------------------------------------------


class Evidence(BaseModel):
    # every claim must point at something inspectable.
    source_type: Literal["trajectory", "test", "benchmark", "code", "log"]
    reference: str  # e.g. "ours/agent/query.py:52", "traces/run1.jsonl#span3"
    observation: str


class Finding(BaseModel):
    # a verdict's structured note about a change unit (design doc section 8).
    severity: Literal["info", "minor", "major", "blocker"]
    location: str
    description: str
    evidence: str = ""
    required_fix: str = ""
    related_criterion: str = ""  # AcceptanceCriterion.id


class Metric(BaseModel):
    # a measurable target for metric-based acceptance.
    name: str
    direction: Literal["increase", "decrease"]
    min_delta: float = 0.0  # primary: required improvement; guardrail: allowed regression


class ImprovementTask(BaseModel):
    # one implementation task in the proposal's DAG.
    id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    acceptance_criteria_ids: list[str] = Field(default_factory=list)


class AcceptanceCriterion(BaseModel):
    # A traceable, executable definition of done.  `command` checks are the hard
    # acceptance signal; `review` and `manual` checks may add judgement but cannot
    # make an autonomous proposal auto-verifiable on their own.
    id: str
    description: str
    mode: Literal["red_green", "invariant", "metric_improvement", "non_regression", "manual"]
    verification: Literal["command", "review", "manual"] = "command"
    command: str = ""
    expected_exit_code: int = 0
    required_output_contains: list[str] = Field(default_factory=list)
    forbidden_output_contains: list[str] = Field(default_factory=list)
    # rigor of the functional test the Test Agent generates:
    #   full    -> core/framework: happy path + edges + error handling + invariants
    #   focused -> important logic: unit test asserting key values + intermediates
    #   basic   -> simple: one unit test on the main behavior
    # Ignored for metric modes (verified via the runner's external benchmark hook)
    # and manual mode (human-reviewed); the Test Agent does not auto-generate those.
    test_level: Literal["full", "focused", "basic"] = "focused"
    required: bool = True


class EvaluationPlan(BaseModel):
    # the frozen evaluation design: what to run + the metric gates.
    primary_metric: Metric | None = None
    guardrail_metrics: list[Metric] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)  # test-agent turns these into runs
    run_count: int = 1  # repeat runs for nondeterministic behavior


class ReviewResult(BaseModel):
    # the Reviewer's verdict on a change unit.
    verdict: Literal["accept", "revise", "escalate"]
    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""


class ExecutionBlocker(BaseModel):
    # raised when a task can't converge within bounded Writer-Reviewer rounds.
    task_id: str
    review_rounds: int
    unresolved_findings: list[Finding] = Field(default_factory=list)
    attempted_fixes: list[str] = Field(default_factory=list)
    affected_tasks: list[str] = Field(default_factory=list)
    recommendation: Literal["abstain_task", "amend_plan", "request_human", "abort_proposal"]


# --- the central artifact -------------------------------------------------------


class ImprovementProposal(BaseModel):
    # an RFC-style, evidence-grounded change proposal (design doc section 6.1).
    summary: str
    problem_statement: str
    evidence: list[Evidence] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    # Explicit repo-relative file paths, directory prefixes ending in "/", or
    # glob patterns the Writer is authorized to modify.
    allowed_write_paths: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tasks: list[ImprovementTask] = Field(default_factory=list)
    # decision inputs (1-5 scores from the LLM; confidence in [0,1]).
    benefit: int
    risk: int
    effort: int
    confidence: float
    # the LLM's proposed decision; the deterministic policy gate has final say.
    decision: Literal["proceed", "abstain", "needs_human"]
    decision_reason: str
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    # ONE run of the candidate: commands whose combined output the Deliverer judges.
    delivery_run: list[str] = Field(default_factory=list)
    # what the Deliverer verifies against that single run's output, point by point.
    delivery_checklist: list[str] = Field(default_factory=list)
    evaluation_plan: EvaluationPlan | None = None
    rollback_plan: str = ""
    alternatives_considered: list[str] = Field(default_factory=list)


class FrozenProposal(BaseModel):
    # an approved proposal, made immutable + hashed so intent can't drift
    # (design doc section 6.2). A substantial change requires a new version.
    proposal_id: str
    proposal_version: int
    proposal_hash: str
    approved_by: str
    approved_at: str
    target_commit: str
    baseline_run_id: str
    proposal: ImprovementProposal
