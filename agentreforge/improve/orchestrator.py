"""Improvement Orchestrator: the analyst that produces an ImprovementProposal.

Reuses the ReAct loop (query) with a READ-ONLY tool subset so the model can
investigate the actual source, grounded by the execution trajectory and the
user's intent. It does NOT write code (that's the Writer). Its final message is
the ImprovementProposal as JSON, validated via Pydantic — with bounded
self-correcting repair retries if the JSON doesn't satisfy the schema.

Kept separate from Planner on purpose: Planner decomposes a plain goal for the
common /plan and /team execution modes; the Orchestrator is the richer,
evidence-grounded planner for the self-improvement pipeline.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ..agent.query import query
from ..llm.base import LlmClient
from ..llm.collect import collect_text
from ..llm.parse import parse_json_model
from ..observability import traceable
from ..tools.base import Tool, ToolContext, ToolResult, object_schema
from ..tools.registry import ToolRegistry
from ..types import Message
from .context import OrchestratorContext, OrchestratorContextBuilder
from .history_index import ImprovementHistoryIndex
from .models import (
    ContractExpansion,
    DiagnosisBoard,
    ImprovementProposal,
    OrchestratorArtifacts,
    SelectionDecision,
)
from .plan_validator import validate_plan_tool
from .records import ReforgeLoopRecord

ORCHESTRATOR_PROMPT = """You are the improvement Orchestrator: an analyst, not an implementer.

You receive two strictly separated histories:
1. `target_agent_runs`: what the TARGET AGENT was asked to do and how it behaved.
   Use these runs to diagnose what capability should improve. Every run is tagged
   with `target_commit`, `evidence_source`, and `is_current`. Read its structured
   `outcome`, `stopped_early`, `step_budget_exhausted`, `evaluation_passed`, and
   `evaluation_summary` fields before interpreting free-form response text.
   `current_run_alerts` is a compact attention index of unresolved outcomes on the
   current target commit. It is not a ranking gate, but every alert requires an
   explicit diagnosis or evidence-based disposition before selection.
   `current_target_commit`, `current_run_ids`, and `non_current_run_ids` are
   deterministic Git-identity facts. Never relabel a listed current run as another
   agent version, and never call a listed non-current run current evidence.
2. `previous_reforge_loops`: what AgentReforge itself planned, wrote, reviewed,
   delivered, and committed earlier in THIS recursive run. Use these to avoid
   repetition and plan the next improvement. Never treat Reforge workflow events as
   target-agent behavior.

For a coding agent, distinguish the TARGET AGENT REPOSITORY from a TASK WORKSPACE.
The target repository contains the agent system being improved; each target-agent run
may ask that same agent version to edit a different disposable repository. Different
task files do not make the run irrelevant or turn it into another agent's history.
`target_commit` identifies the agent version whose capability the run measures. A
failure in an idempotency, algorithm, CLI, or other task workspace is current evidence
about that target agent when its `target_commit` is current. Never describe generated
task artifacts as delivered improvements to another repository.

You also receive a deterministic repository map and READ-ONLY code tools. Produce an
evidence-grounded ImprovementProposal only after completing this workflow:

PHASE 1 — ORIENT
- State internally what the target agent was asked to do, its observable outcome,
  the current agent architecture, and what previous Reforge loops already changed.
- If the target task or outcome is missing, reduce confidence; do not invent it.
- Treat `is_current=true` runs as evidence about the current candidate commit.
  Baseline or older-commit runs remain useful historical evidence, but they cannot
  prove that a capability is still broken after a delivered Loop. Delivery Scenario
  trajectories from a successful Loop are fed back as `delivered_scenario` runs for
  the new commit and are the freshest behavioral evidence in the next Loop.
- The context contains bounded summaries. Use `get_target_evidence` or
  `get_reforge_loop` only when a cited detail needs inspection.
- Build an Achievement Ledger from `previous_reforge_loops` before diagnosing:
  delivered achievements, rolled-back attempts, remaining gaps, and current source
  evidence. A delivered record is a claim to verify against the current tree, not an
  excuse to repeat the same intervention. If the capability is present, move on. If it
  is present but ineffective, diagnose the next direct cause instead of rewording or
  reimplementing the recorded solution.
- Treat `improvement_backlog` as a DYNAMIC HYPOTHESIS BACKLOG, not a fixed roadmap.
  Re-assess it against the current tree and latest trajectory every Loop: re-rank useful
  items, refine an imprecise item, add newly discovered direct causes, and retire stale
  hypotheses. `open`, `deferred`, and `attempt_failed` items are not obligations, but
  they must be considered before inventing a new Candidate or abstaining.
- A `behavior_verified` backlog item closes only its recorded tuple of capability gap,
  mechanism, expected delta, and verification scope. It does NOT close a broad field
  such as "tool use", and it does not by itself prove a causal pre/post improvement.
  Work in the same field is allowed when it has a different direct cause, mechanism, or
  observable delta.
- Treat every non-completed Loop's `failure_kind` and `attempt_fingerprint` as a
  Negative-Attempt Ledger. After `verification_gap`, do not repeat the same Candidate
  with the same unavailable evidence requirement. Change the verification mechanism,
  choose a different evidenced Candidate, or abstain. Rewording the same scenario is
  not a new attempt.

PHASE 2 — DIAGNOSE
- Perform PROBLEM TRIAGE before considering solution cost. Identify 1-3 capability gaps
  from the prior dynamic backlog, whole current architecture, and latest evidence rather
  than examining only the most recently delivered Scenario. For each, distinguish observed
  symptom, likely root cause, missing capability, evidence references, and uncertainty.
- Assess the problem itself separately from its proposed solution: evidence strength,
  failure severity, recurrence, cross-task impact, evidence freshness, and user relevance.
  A cheap intervention must not make a weak problem outrank a current terminal failure.
- Account for every current `failed_verification`, incomplete, step-budget, or error outcome.
  It need not be selected, but its direct cause must become a Candidate hypothesis or receive
  an explicit evidence-based disposition explaining why it is non-actionable or lower value.
  Begin this audit from `current_run_alerts`; do not let a newer run from a different
  `target_commit` silently displace a current-commit alert.
- A missing action in one trajectory is not automatically a missing system feature.
  Check the source to learn whether the failure is instruction, workflow, tool, state,
  or architecture related.
- Current-run facts are authoritative and already supplied directly. Use
  `search_improvement_history` only after an initial diagnosis, to find analogous
  OLD runs; retrieval results are supporting experience, not proof of current behavior.

PHASE 3 — GENERATE CANDIDATES
- Compare viable interventions at different leverage levels: prompt/instruction,
  workflow/control flow, and tool/module/architecture where supported by evidence.
- For each Candidate fill four qualitative scorecards before ranking:
  * `problem`: evidence strength, failure severity, recurrence, cross-task impact,
    evidence freshness, and user relevance;
  * `causal`: root-cause confidence, intervention fit, competing hypotheses, and a
    falsification condition;
  * `impact`: expected outcome impact, generality, one-Loop feasibility, regression
    risk, effort, and the predicted capability delta;
  * `evaluability`: mechanism and outcome observability, baseline/candidate
    discriminability, attribution confidence, noise robustness, evaluation cost,
    baseline prediction, candidate prediction, observable difference, and confounders.
- Scores structure your judgment; NEVER add them into a mechanical total. Use anchored
  reasoning and evidence references. High observability alone must not make a trivial
  low-impact task win. For a high-impact but weakly observable problem, prefer an
  independently useful observable vertical slice or defer it with the measurement gap.
- All scorecard numbers use 1=low and 5=high. Higher regression risk, effort, and
  evaluation cost are disadvantages; the other dimensions are advantages. Keep
  assessments concise enough to compare 1-3 grounded Candidates within the output budget.
- Calibrate scores against observable anchors:
  * failure severity 5 means terminal `failed_verification`, incomplete work, or an
    unmet user outcome; 3 means recoverable degradation; 1 means a theoretical weakness;
  * outcome observability 5 means an executable user outcome changes directly; 1 means
    only prompt wording or process compliance is visible;
  * discriminability 5 means baseline is expected to fail while Candidate is expected
    to pass under the same Scenario; 1 means both likely pass;
  * noise robustness 5 means deterministic or repeated evidence; 1 means a single
    stochastic run could easily reverse the apparent result.
- Do not default to a prompt tweak merely because it is cheap. Do not force a new
  module when the evidence supports a smaller fix.
- Reject changes that only patch the observed example without a reusable mechanism.
- Exclude every Candidate whose capability delta is already in the verified Achievement
  Ledger at the SAME completion scope unless new trajectory, test, or source evidence
  proves that exact capability remains ineffective. In that case target the newly
  diagnosed direct cause; do not select a cosmetic refinement of the completed
  intervention.
- Capability-level de-duplication is your responsibility, not string equality. Treat
  renamed variants such as "fallback guidance" and "strengthen fallback guidance" as
  the same delivered capability when their mechanism and expected delta are equivalent.
  Reopen a delivered capability only with CURRENT-commit evidence that it remains
  ineffective; stale baseline evidence is insufficient.
- A Delivery command, smoke marker, expected string, report, or documentation change is
  verification infrastructure—not a target-agent capability. Never select work whose
  purpose is merely to make AgentReforge's gate or evaluator pass. It is eligible only
  when independent evidence shows the target agent itself has the corresponding runtime
  or usability defect.
- Represent the current todo set as `candidate_backlog`, a dictionary keyed by a stable
  Candidate id. Every item must carry its diagnosis, intervention, priority, scope,
  dependencies/conflicts, and history/disposition. Do not make the todo list a list of
  names. Preserve a prior id when the hypothesis is materially the same.

PHASE 4 — SELECT ONE IMPROVEMENT
- Produce `preliminary_ranking` for every viable Candidate. Rank qualitatively in this
  order of attention: problem evidence/severity, causal fit, expected cross-task outcome
  delta, evaluability/discriminability, one-Loop feasibility, then risk/effort. This is
  an LLM judgment, not a weighted formula.
- Take preliminary ranks 1 and 2 and produce `top_two_comparison`. Give the strongest
  case for each, compare direct evidence, causal fit, expected outcome delta,
  observability/discriminability, risk/effort, and opportunity cost, then state a winner.
  Include a counterfactual: would the baseline likely pass the proposed Scenario already,
  and could a Scenario pass while the claimed agent improvement remains unrealized?
- When only one viable Candidate exists, compare it against the literal option `DEFER`.
  If `DEFER` wins, continue auditing other problems; abstain only when no remaining
  evidence-backed Candidate is worth a Loop.
- The pairwise winner becomes `selected_candidate_id`. It may differ from preliminary
  rank 1, but explain why. Do not select a Candidate merely because it is cheaper,
  historically familiar, or easier to demonstrate.
- Claims of improved "reliability" or "consistency" require an observable outcome delta
  or repeated evidence. A one-off Scenario that only shows the model followed new prompt
  wording is weak mechanism evidence, not strong reliability evidence.
- One Loop selects exactly ONE backlog Candidate and expands it into exactly ONE
  `selected_change_contract`. The Contract is one CHANGE UNIT: the smallest independently
  useful, causally coherent,
  end-to-end verifiable capability delta. It may contain several implementation steps
  and touch multiple files, but it has one primary root cause, one observable delta,
  one Reviewer decision, and one Loop commit.
- Split by outcome and causal boundary, not by file, module, or broad field:
  * same outcome + same mechanism across several files -> one Change Unit;
  * different outcomes that merely touch the same files -> separate Loops;
  * independent improvements -> separate backlog items, even when both are small;
  * one large capability -> choose a vertical slice that is useful and demonstrable
    without the deferred slices.
- If the best Candidate is too large for one bounded Contract, select the smallest coherent
  capability slice that is independently useful and verifiable. Record deferred slices
  as remaining gaps for later Loops; do not create a multi-Task batch.
- Explain why this Candidate is the highest-value unsolved capability and why it is not
  a duplicate of a delivered achievement.

PHASE 5 — CONTRACT
- Set `selected_candidate_id` to one exact key from `candidate_backlog`.
- Ensure the selected id is the qualitative pairwise winner. The framework records this
  decision artifact but does not compute or override your ranking.
- Expand only that Candidate into `selected_change_contract`. Snapshot its diagnosis and
  intervention, then define objective, inputs/expected outputs, required behavior,
  implementation constraints, invariants, prohibited shortcuts, suggested write scope,
  Reviewer focus, acceptance criteria, Delivery scenarios/checklist, and rollback.
- Put whole-picture boundaries in `proposal_guardrails`. They are constitutional
  constraints: implementation may deviate from suggested paths but Writer, Reviewer, and
  Deliverer must not contradict them.
- Before final output call `validate_plan` with one root entry using the Contract id and
  an empty dependency list.

Rules:
- Use the read-only tools (read_file, search_code, grep, list_dir, glob) to inspect the
  REAL source before proposing. Do not guess.
- Ground every claim in inspectable evidence: a code location, a trajectory record, or a test.
- Evidence and Candidate comparison improve the plan, but they are reasoning aids rather
  than schema gates. Use as much structure as the change genuinely needs.
- You must NOT write or modify any code.
- Score the structured Candidate scorecards, then summarize benefit/risk/effort 1-5 and
  confidence 0.0-1.0 as advisory metadata. No score is a hard gate or weighted total.
  Propose proceed, abstain, or needs_human explicitly.
- The recursion limit is a ceiling, never a target. Choose `abstain` when there is no
  new evidence-backed unsolved capability, when only completed achievements remain,
  when the only available work would optimize a Delivery/evaluation artifact, or when
  evidence is too weak to identify a direct cause. Do not invent work to consume Loops.
- Before `abstain`, audit the current dynamic backlog AND the whole source-visible
  architecture. In `decision_reason`, explain why each still-open/deferred/failed
  hypothesis is now solved, unsupported, blocked, or lower-value than the cost/risk.
  One successful Scenario or one `behavior_verified` item is never enough to declare a
  broad field—or the whole agent—complete. Recoverable tool/runtime failures in the
  latest trajectory remain diagnostic evidence even when the final Scenario passed.
- BEFORE emitting the proposal, call `validate_plan` with one item:
  `{id: selected_change_contract.contract_id, dependencies: []}`. Only output the final
  decision once validate_plan passes.
- Contract `allowed_write_paths`, `affected_components`, clauses, and acceptance criteria are
  implementation suggestions for Writer/Reviewer, not hard authorization boundaries.
  Keep them useful and concise; do not over-specify a small change.
- Contract acceptance criteria are review/test hints. Only explicit safety-property
  checks, `delivery_run`, and frozen `delivery_scenarios` participate in Delivery.
- For a target agent with a runnable CLI, prefer one or two `delivery_scenarios` that
  exercise the selected capability end to end. Each scenario contains a frozen prompt,
  a safe argv command, a small isolated fixture, and
  observable expected/forbidden behaviors. Design scenarios before Writer runs; do not
  adapt acceptance difficulty after seeing the implementation.
- Derive each Scenario from the selected Candidate's evaluability scorecard. A Scenario
  must name an observable difference that the intervention is expected to cause. If the
  baseline likely passes it unchanged, treat it as contract-compliance evidence only and
  do not claim it demonstrates improvement; strengthen the Scenario or lower the
  Candidate's discriminability and rank.
- Scenario commands are argv arrays, never shell strings. Inspect the real target
  entrypoint and match its actual CLI. Use `{prompt}` only when that CLI accepts a
  natural-language task argument; use `{workspace}` only when it accepts a workspace
  path. A prompt also describes the scenario to the Deliverer and does not have to be
  passed to the process. Keep scenarios
  bounded and directly causal: they decide Delivery for this Candidate, not general
  benchmark quality. Optional later diagnostic probes may explain failure but cannot
  change the frozen pass/fail scenarios.
- Set `requires_trajectory=true` whenever a scenario must prove internal tool use,
  inspect-before-edit, verify-after-edit, call ordering, or any other process fact.
  A target's own final response is not evidence that it performed those actions. If
  the target cannot emit trajectory evidence, design artifact/output-observable
  requirements instead; do not claim an unobservable process property.
- When a scenario depends on executable availability, declare only the necessary
  typed `executable_conditions` (`name` plus `available|unavailable`). The trusted
  execution tool materializes and records these conditions. Never encode environment setup
  in the prompt, fixture, arbitrary environment variables, or a shell command.
- `delivery_run` contains bounded system-level smoke/start suggestions. The Deliverer
  agent chooses and invokes frozen run/scenario tools, observes commands, exit codes,
  bounded output, artifacts, and trajectory, then judges the result. Execution evidence
  can establish a hard failure but cannot by itself establish goal realization.
  Do not grade general generated-code quality here.
- Verification commands must not leave runtime artifacts in the candidate. Prefer
  `PYTHONDONTWRITEBYTECODE=1 python3` for Python commands (do not assume a `python`
  alias exists) and never include `__pycache__`,
  `*.pyc`, `.pytest_cache`, coverage output, or runtime state files in allowed_write_paths;
  the post-write policy gate hard-denies those artifacts.
- Do NOT create a Change Contract whose job is to prove that Delivery commands leave the
  worktree unchanged. DeliveryCoordinator owns this invariant deterministically by
  comparing pre/post candidate tree snapshots. Contracts may add real product tests or smoke
  scripts, but repository immutability is not a product-code requirement.
- For a genuine safety property, mark its criterion with
  `verified_safety_properties` and use `mode=invariant`. Safety probes are system-owned:
  leave the criterion `command` empty. Never attach a safety property to an ordinary
  capability scenario or generate a prompt asking the target agent to claim that it
  was blocked. The execution tool's target adapter executes the actual traversal
  probe and reports its exit code. Never encode a condition that rewards the
  safety violation.
- `required_safety_properties` is EMPTY BY DEFAULT. Declare `path_confinement` only
  when the selected Candidate itself implements, modifies, or promises to preserve a
  path-taking/filesystem boundary. Never attach it to an unrelated prompt, runtime
  fallback, model, or workflow change. A Contract that declares it must contain one
  matching invariant criterion; the criterion must list `path_confinement` and leave
  `command` empty. Do not require a safety property that the baseline does not have
  unless implementing that property is the Candidate's explicit objective.

When done, output ONLY the proposal as ONE JSON object in a ```json code block. Field types:
- candidate_backlog: {stable_candidate_id: {
    id: same stable_candidate_id,
    status: open|deferred|attempt_failed|behavior_verified,
    title,
    diagnosis: {symptom, root_cause, capability_gap, evidence_refs: [str], uncertainty},
    intervention: {level: prompt|workflow|tool|module|architecture, mechanism,
      expected_capability_delta},
    priority: {
      problem: {evidence_strength: 1-5, failure_severity: 1-5, recurrence: 1-5,
        cross_task_impact: 1-5, evidence_freshness: 1-5, user_relevance: 1-5,
        assessment},
      causal: {root_cause_confidence: 1-5, intervention_fit: 1-5,
        competing_hypotheses: [str], falsification_condition},
      impact: {expected_outcome_impact: 1-5, generality: 1-5,
        one_loop_feasibility: 1-5, regression_risk: 1-5, effort: 1-5,
        expected_delta},
      evaluability: {mechanism_observability: 1-5, outcome_observability: 1-5,
        discriminability: 1-5, attribution_confidence: 1-5, noise_robustness: 1-5,
        evaluation_cost: 1-5, baseline_prediction, candidate_prediction,
        observable_difference, confounders: [str]},
      benefit: 1-5, risk: 1-5, effort: 1-5, confidence: 0-1, rank_reason},
    scope: {affected_components: [str], non_goals: [str]},
    dependencies: [candidate_id], conflicts_with: [candidate_id],
    history: {first_seen_loop: int, last_reviewed_loop: int,
      previous_attempts: [str], verification_scope: [str],
      verification_level: none|implemented|behavior_verified|delta_demonstrated,
      disposition_reason}
  }}
- preliminary_ranking: [{candidate_id: candidate_backlog key, rank: int starting at 1,
    rationale: str}], covering every viable Candidate.
- top_two_comparison: null only for abstain with no viable Candidate, otherwise {
    candidate_a: preliminary rank 1,
    candidate_b: preliminary rank 2 or literal `DEFER` when only one Candidate exists,
    strongest_case_for_a, strongest_case_for_b,
    comparative_judgments: {dimension_name: qualitative comparison},
    baseline_counterfactual, candidate_counterfactual,
    winner: candidate_a|candidate_b, decision_reason}.
- selected_candidate_id: exactly one candidate_backlog key for proceed/needs_human,
  empty for abstain.
- selected_change_contract: null for abstain, otherwise {
    contract_id, backlog_item_id: selected_candidate_id, backlog_revision: int,
    objective, rationale,
    diagnosis: the selected Candidate diagnosis snapshot,
    intervention: the selected Candidate intervention snapshot,
    inputs: [{id, description}], expected_outputs: [{id, description}],
    required_behaviors: [{id, description}],
    implementation_constraints: [{id, description}],
    invariants: [{id, description}],
    prohibited_shortcuts: [{id, description}],
    affected_components: [str], allowed_write_paths: [str], reviewer_focus: [str],
    required_safety_properties: [] by default or [path_confinement],
    acceptance_criteria: [AcceptanceCriterion objects described below],
    delivery_run: [str], delivery_scenarios: [DeliveryScenario objects described below],
    delivery_checklist: [str], rollback_plan}
- proposal_guardrails: [{id, description}] for whole-picture constraints that no
  implementation path may violate.
- evidence[]: {source_type: trajectory|test|benchmark|code|log, reference: str, observation: str}
- acceptance_criteria[]: {id: str, description: str,
  mode: red_green|invariant|metric_improvement|non_regression|manual,
  check_type: unit|integration|smoke,
  verification: command|review|manual, command: str, expected_exit_code: int,
  required_output_contains: [str], forbidden_output_contains: [str],
  verified_safety_properties: [path_confinement],
  test_level: full|focused|basic, required: bool}
- benefit/risk/effort: int 1-5; confidence: float 0-1
- decision: proceed|abstain|needs_human
- selected_change_contract.delivery_run[]: system-level integration/smoke commands.
- selected_change_contract.delivery_scenarios[]: {id: str, prompt: str,
  command: [argv strings, optionally containing {prompt} and/or {workspace} only
    when required by the target entrypoint],
  fixture_files: {repo_relative_path: content},
  expected_behaviors: [str], forbidden_behaviors: [str],
  executable_conditions: [{name: str, state: available|unavailable}],
  requires_trajectory: bool}.
- selected_change_contract.delivery_checklist[]: high-level system requirements.
- The Proposal-level fields `summary`, `problem_statement`, `goals`, `non_goals`,
  `affected_components`, and `dependencies` describe the whole-picture intent.
- The legacy fields `analysis`, `tasks`, `allowed_write_paths`, `acceptance_criteria`,
  `delivery_run`, `delivery_scenarios`, and `delivery_checklist` should be omitted.
- Contract delivery_checklist items are system requirements the Deliverer can judge from the
  full diff and scenario evidence (the selected Candidate is implemented, reachable, and
  observed in the intended path). Do not require it to repeat function-level review.
- also: summary, problem_statement, goals[], non_goals[], affected_components[],
  dependencies[], decision_reason, alternatives_considered[]"""


TRIAGE_PROMPT = """You are the Evidence Triage stage inside one improvement Orchestrator.
You diagnose the target agent; you do not propose code changes, rank solutions, design
tests, or write an ImprovementProposal.

Treat Git identity fields as deterministic facts. A target-agent run evaluates
`target_commit` on a disposable TASK WORKSPACE. Failure of the generated artifact is
capability evidence about that agent version; the task files do not need to exist in
the TARGET AGENT REPOSITORY for the agent-level failure to be actionable.
The task artifact's missing feature is a SYMPTOM, not the improvement itself. Diagnose
the reusable TARGET AGENT capability that caused incomplete work (for example planning,
tool use, state tracking, cross-component completion, or verification budgeting).
Never propose adding the task's domain feature or task-workspace files to the target
agent repository. `ProblemCase.affected_scope` names only target-agent source components
or abstract agent capabilities visible in the supplied repository map.

Audit every `current_run_alert`. For each alert, record the observed failure, its
agent-level interpretation, plausible direct causes, evidence references, and exactly
one disposition. A terminal failure may be deferred when evidence cannot identify an
actionable agent-level problem, but "different task workspace" is not by itself a
valid reason. Separate symptoms from hypotheses and state uncertainty.

Use current runs, the whole source-visible architecture, the dynamic backlog,
achievements, and failed attempts. Current-commit evidence is authoritative; runs in
`non_current_run_ids` are historical only. Use read-only tools to inspect source or a
cited full event when needed. Output ONLY one DiagnosisBoard JSON object:
{
  "alert_dispositions": [{
    "run_id": str, "observed_failure": str, "agent_level_interpretation": str,
    "likely_causes": [str],
    "disposition": "candidate_problem|deferred|unsupported|already_resolved",
    "disposition_reason": str, "evidence_refs": [str]
  }],
  "problem_cases": {problem_id: {
    "id": same problem_id, "symptom": str, "capability_gap": str,
    "likely_root_causes": [str], "affected_scope": [str],
    "evidence_refs": [str], "uncertainty": str
  }},
  "retired_problems": {problem_id: reason},
  "whole_picture_summary": str
}
Every current alert must appear exactly once in alert_dispositions."""


SELECTION_PROMPT = """You are the Candidate Selection stage inside one improvement
Orchestrator. Evidence triage is already complete. Generate interventions from the
DiagnosisBoard, compare them, and select exactly one bounded improvement or abstain.
Do not design a Change Contract or Delivery Scenario.

Treat DiagnosisBoard evidence/commit scope as fixed input. You may inspect cited source
or evidence, but do not silently discard a problem or relabel current and non-current
runs. Generate 1-3 viable Candidates at prompt, workflow, tool/module, or architecture
levels when supported. Each Candidate uses the complete BacklogCandidate schema.
Candidates improve reusable behavior of the TARGET AGENT source in `repository`; they
must never implement the disposable task workspace's domain feature inside the agent
repository. Candidate affected_components must name real target-repository paths or
abstract agent components from the supplied map.

Fill the problem, causal, impact, and evaluability scorecards before ranking. Scores
are advisory (1 low, 5 high); NEVER compute a weighted total. Rank by problem
evidence/severity, causal fit, expected cross-task outcome delta,
observability/discriminability, one-Loop feasibility, then risk/effort. Avoid changes
whose expected effect is too small or noisy to distinguish from model variance.

Produce a preliminary ranking for every viable Candidate, then compare ranks 1 and 2
pairwise. When only one Candidate is viable, compare it with literal DEFER. Pairwise
may overturn rank 1. One Loop selects one Candidate. Output ONLY one
SelectionDecision JSON object with:
summary, problem_statement, candidate_backlog, preliminary_ranking,
top_two_comparison, selected_candidate_id, deferred_candidates, benefit, risk, effort,
confidence, decision (proceed|abstain|needs_human), and decision_reason.
Each candidate_backlog value must use this object shape (never replace nested objects
with prose strings):
{
  "id": str, "status": "open|deferred|attempt_failed|behavior_verified", "title": str,
  "diagnosis": {"symptom": str, "root_cause": str, "capability_gap": str,
    "evidence_refs": [str], "uncertainty": str},
  "intervention": {"level": "prompt|workflow|tool|module|architecture",
    "mechanism": str, "expected_capability_delta": str},
  "priority": {
    "problem": {"evidence_strength": 1-5, "failure_severity": 1-5,
      "recurrence": 1-5, "cross_task_impact": 1-5, "evidence_freshness": 1-5,
      "user_relevance": 1-5, "assessment": str},
    "causal": {"root_cause_confidence": 1-5, "intervention_fit": 1-5,
      "competing_hypotheses": [str], "falsification_condition": str},
    "impact": {"expected_outcome_impact": 1-5, "generality": 1-5,
      "one_loop_feasibility": 1-5, "regression_risk": 1-5, "effort": 1-5,
      "expected_delta": str},
    "evaluability": {"mechanism_observability": 1-5, "outcome_observability": 1-5,
      "discriminability": 1-5, "attribution_confidence": 1-5,
      "noise_robustness": 1-5, "evaluation_cost": 1-5,
      "baseline_prediction": str, "candidate_prediction": str,
      "observable_difference": str, "confounders": [str]},
    "benefit": 1-5, "risk": 1-5, "effort": 1-5, "confidence": 0-1,
    "rank_reason": str
  },
  "scope": {"affected_components": [str], "non_goals": [str]},
  "dependencies": [candidate_id], "conflicts_with": [candidate_id],
  "history": {"first_seen_loop": int, "last_reviewed_loop": int,
    "previous_attempts": [str], "verification_scope": [str],
    "verification_level": "none|implemented|behavior_verified|delta_demonstrated",
    "disposition_reason": str}
}
top_two_comparison contains candidate_a, candidate_b, strongest_case_for_a,
strongest_case_for_b, comparative_judgments (object of qualitative strings),
baseline_counterfactual, candidate_counterfactual, winner (the exact winning Candidate
id), and decision_reason.
For proceed/needs_human, selected_candidate_id must name a backlog key. For abstain it
must be empty and top_two_comparison may be null."""


CONTRACT_PROMPT = """You are the Contract Expansion stage inside one improvement
Orchestrator. Selection is already frozen. Expand only `selected_candidate_id` into one
SelectedChangeContract. Do not re-diagnose, re-rank, choose another Candidate, or
reproduce the SelectionDecision.

Create one causally coherent Change Contract: objective, rationale,
diagnosis/intervention snapshots, inputs,
expected outputs, behaviors, constraints, invariants, prohibited shortcuts, affected
components, concise allowed write suggestions, reviewer focus, acceptance criteria,
delivery runs/scenarios, checklist, and rollback plan.

The Contract is guidance for Writer/Reviewer/Deliverer, not a hard file boundary.
Design one or two frozen end-to-end scenarios only when they can distinguish the
selected capability. Scenario commands are argv arrays; use {prompt}/{workspace} only
when the real CLI accepts them. Process claims require trajectory evidence. Safety
properties are empty by default; path_confinement is declared only when the selected
change touches that boundary. Call validate_plan once with the single contract id.

Each acceptance criterion is an object with: id, description,
mode (red_green|invariant|metric_improvement|non_regression|manual),
check_type (unit|integration|smoke), verification (command|review|manual), command,
expected_exit_code, required_output_contains, forbidden_output_contains,
verified_safety_properties, test_level (full|focused|basic), and required.
Each evidence item is an object with source_type
(trajectory|test|benchmark|code|log), reference, and observation.

Output ONLY one ContractExpansion JSON object:
{
  "proposal_guardrails": [{"id": str, "description": str}],
  "selected_change_contract": {the complete contract described above},
  "evidence": [{"source_type": "trajectory|test|benchmark|code|log",
    "reference": str, "observation": str}],
  "goals": [str], "non_goals": [str], "affected_components": [str],
  "dependencies": [str], "alternatives_considered": [str]
}
The coordinator—not you—joins this artifact with the frozen SelectionDecision."""


class Orchestrator:
    def __init__(
        self,
        client: LlmClient,
        registry: ToolRegistry,
        cwd: str,
        *,
        code_index: Any = None,
        history_index: ImprovementHistoryIndex | None = None,
    ):
        self.client = client
        self.cwd = cwd
        self.code_index = code_index
        self.history_index = history_index
        self.last_diagnosis: DiagnosisBoard | None = None
        self.last_selection: SelectionDecision | None = None
        # analyst gets read-only tools ONLY — it must never modify code.
        self.registry = _read_only_registry(registry)
        # plus a deterministic DAG check it must run before finalizing its plan.
        self.registry.register(validate_plan_tool)

    @traceable(name="orchestrator.analyze", run_type="chain")
    async def analyze(
        self,
        intent: str,
        target_trajectory: list[dict[str, Any]],
        loop_history: list[ReforgeLoopRecord] | None = None,
        max_turns: int = 12,
        context: OrchestratorContext | None = None,
    ) -> ImprovementProposal:
        context = context or OrchestratorContextBuilder(self.cwd).build(
            intent=intent,
            target_trajectory=target_trajectory,
            previous_reforge_loops=loop_history or [],
            run_manifest={},
        )
        # The initial prompt contains a bounded evidence catalog. This read-only
        # drill-down tool exposes the full selected TARGET event by stable id.
        self.registry.register(_evidence_tool(context))
        self.registry.register(_reforge_loop_tool(context))
        if self.history_index is not None:
            self.registry.register(_history_tool(context, self.history_index))

        triage_text = await self._investigate_stage(
            system_prompt=TRIAGE_PROMPT,
            user_message=_build_triage_request(context),
            max_turns=max_turns,
        )
        diagnosis = await self._parse_stage_with_repair(
            triage_text,
            DiagnosisBoard,
            system_prompt=TRIAGE_PROMPT,
            stage_name="DiagnosisBoard",
            validate=lambda board: _validate_triage_coverage(board, context),
        )
        self.last_diagnosis = diagnosis

        selection_text = await self._investigate_stage(
            system_prompt=SELECTION_PROMPT,
            user_message=_build_selection_request(context, diagnosis),
            max_turns=max_turns,
        )
        selection = await self._parse_stage_with_repair(
            selection_text,
            SelectionDecision,
            system_prompt=SELECTION_PROMPT,
            stage_name="SelectionDecision",
            validate=lambda item: _validate_selection(item, context),
        )
        self.last_selection = selection

        if selection.decision == "abstain":
            proposal = _abstaining_proposal(selection)
        else:
            contract_text = await self._investigate_stage(
                system_prompt=CONTRACT_PROMPT,
                user_message=_build_contract_request(context, diagnosis, selection),
                max_turns=max_turns,
            )
            expansion = await self._parse_stage_with_repair(
                contract_text,
                ContractExpansion,
                system_prompt=CONTRACT_PROMPT,
                stage_name="ContractExpansion",
                validate=lambda item: _validate_contract_selection(item, selection),
            )
            proposal = _proposal_from_contract(selection, expansion)

        # Decision artifacts are coordinator-owned and cannot drift during
        # contract expansion. The final Proposal remains the unchanged downstream
        # interface used by Gate, Writer, Reviewer, and Deliverer.
        return _attach_frozen_decision(proposal, diagnosis, selection)

    async def revise(self, proposal: ImprovementProposal, problem: str) -> ImprovementProposal:
        """Re-generate the proposal to fix a specific problem (e.g. an invalid DAG)."""
        # feed the problem + the previous proposal back; ask for a corrected one.
        msg = (
            f"Your previous proposal had a problem that must be fixed:\n{problem}\n\n"
            f"Previous proposal:\n{proposal.model_dump_json(indent=2)}\n\n"
            "Output ONLY a corrected proposal as a single ```json block — fix the "
            "problem and keep everything else valid. For a missing safety-property "
            "mapping, update BOTH sides: tag an invariant acceptance criterion with "
            "`verified_safety_properties`, leave its command empty because the probe is "
            "system-owned, and declare the same safety property in the Selected Change "
            "Contract. The Contract is the only execution unit and has no Task DAG."
        )
        text = await collect_text(
            self.client, [Message(role="user", content=msg)], system_prompt=ORCHESTRATOR_PROMPT
        )
        corrected = await self._parse_proposal_with_repair(text)
        artifacts = proposal.orchestrator_artifacts
        if artifacts is None:
            return corrected
        return _attach_frozen_decision(
            corrected,
            artifacts.diagnosis,
            artifacts.selection,
        )

    async def _parse_proposal_with_repair(
        self,
        text: str,
        *,
        max_repairs: int = 2,
    ) -> ImprovementProposal:
        """Parse a proposal, feeding exact schema errors back with a bounded budget."""

        candidate = text
        for repair_i in range(max_repairs + 1):
            try:
                return parse_json_model(candidate, ImprovementProposal)
            except ValueError as exc:
                if repair_i >= max_repairs:
                    raise
                candidate = await self._repair(candidate, str(exc))
        raise AssertionError("unreachable")

    async def _parse_stage_with_repair(
        self,
        text: str,
        model_cls: type[BaseModel],
        *,
        system_prompt: str,
        stage_name: str,
        validate,
        max_repairs: int = 2,
    ):
        """Repair only the stage whose typed handoff is invalid."""

        candidate = text
        for repair_i in range(max_repairs + 1):
            try:
                parsed = parse_json_model(candidate, model_cls)
                problems = validate(parsed)
                if problems:
                    raise ValueError("; ".join(problems))
                return parsed
            except ValueError as exc:
                if repair_i >= max_repairs:
                    raise
                candidate = await self._repair_stage(
                    candidate,
                    str(exc),
                    system_prompt=system_prompt,
                    stage_name=stage_name,
                )
        raise AssertionError("unreachable")

    async def _investigate_stage(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_turns: int,
    ) -> str:
        text = ""
        async for event in query(
            client=self.client,
            registry=self.registry,
            system_prompt=system_prompt,
            user_message=user_message,
            cwd=self.cwd,
            code_index=self.code_index,
            max_turns=max_turns,
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text

    async def _investigate(
        self,
        context: OrchestratorContext,
        max_turns: int,
    ) -> str:
        """Compatibility entrypoint for callers testing the legacy one-shot prompt."""

        return await self._investigate_stage(
            system_prompt=ORCHESTRATOR_PROMPT,
            user_message=_build_request(context),
            max_turns=max_turns,
        )

    async def _repair(self, bad_output: str, error: str) -> str:
        # feed the model its own output + the validation error; ask for corrected JSON.
        msg = (
            f"Your JSON did not satisfy the schema:\n{error}\n\n"
            f"Your previous output was:\n{bad_output}\n\n"
            "Output ONLY the corrected proposal as a single ```json block that matches "
            "the schema exactly (fix the field types shown in the error)."
        )
        return await collect_text(
            self.client, [Message(role="user", content=msg)], system_prompt=ORCHESTRATOR_PROMPT
        )

    async def _repair_stage(
        self,
        bad_output: str,
        error: str,
        *,
        system_prompt: str,
        stage_name: str,
    ) -> str:
        msg = (
            f"Your {stage_name} handoff was invalid:\n{error}\n\n"
            f"Previous output:\n{bad_output}\n\n"
            f"Output ONLY a corrected {stage_name} JSON object. Preserve valid "
            "judgments and fix only the reported schema or handoff problem."
        )
        return await collect_text(
            self.client,
            [Message(role="user", content=msg)],
            system_prompt=system_prompt,
        )


def _read_only_registry(registry: ToolRegistry) -> ToolRegistry:
    read_only = ToolRegistry()
    read_only.register_all(
        [t for name in registry.list_names() if (t := registry.get(name)) and t.is_read_only]
    )
    return read_only


def _build_request(context: OrchestratorContext) -> str:
    """Serialize one mandatory, typed management view instead of raw mixed logs."""

    return (
        "OrchestratorContext (deterministically prepared; trajectory kinds are separate):\n"
        f"```json\n{json.dumps(context.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n"
        "```\n\nFollow PHASES 1-5 in order. Use evidence references from the context and "
        "inspect the real source with read-only tools. Then output the final "
        "ImprovementProposal JSON."
    )


def _compact_evidence_index(context: OrchestratorContext) -> list[dict[str, Any]]:
    """Keep lookup coordinates in-context; full event content stays behind a tool."""

    keys = (
        "event_id",
        "run_id",
        "type",
        "tool",
        "is_error",
        "evidence_source",
        "target_commit",
    )
    return [
        {key: record.get(key) for key in keys if record.get(key) is not None}
        for record in context.evidence_catalog
    ]


def _build_triage_request(context: OrchestratorContext) -> str:
    payload = {
        "improvement_intent": context.improvement_intent,
        "target_run_semantics": context.target_run_semantics,
        "current_target_commit": context.current_target_commit,
        "current_run_ids": context.current_run_ids,
        "non_current_run_ids": context.non_current_run_ids,
        "current_run_alerts": [
            item.model_dump(mode="json") for item in context.current_run_alerts
        ],
        "target_agent_runs": [
            item.model_dump(mode="json") for item in context.target_agent_runs
        ],
        "previous_reforge_loops": [
            item.model_dump(mode="json") for item in context.previous_reforge_loops
        ],
        "improvement_backlog": {
            key: item.model_dump(mode="json")
            for key, item in context.improvement_backlog.items()
        },
        "repository": context.repository.model_dump(mode="json"),
        "run_manifest": context.run_manifest,
        "evidence_index": _compact_evidence_index(context),
    }
    return (
        "Produce the evidence/problem triage artifact from this deterministic view. "
        "Retrieve full evidence only when a cited detail matters.\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )


def _build_selection_request(
    context: OrchestratorContext,
    diagnosis: DiagnosisBoard,
) -> str:
    payload = {
        "improvement_intent": context.improvement_intent,
        "diagnosis_board": diagnosis.model_dump(mode="json"),
        "previous_reforge_loops": [
            item.model_dump(mode="json") for item in context.previous_reforge_loops
        ],
        "improvement_backlog": {
            key: item.model_dump(mode="json")
            for key, item in context.improvement_backlog.items()
        },
        "repository": context.repository.model_dump(mode="json"),
        "run_manifest": context.run_manifest,
    }
    return (
        "Generate interventions and make the one-Loop selection from this already "
        "triaged decision view.\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )


def _build_contract_request(
    context: OrchestratorContext,
    diagnosis: DiagnosisBoard,
    selection: SelectionDecision,
) -> str:
    selected = selection.candidate_backlog.get(selection.selected_candidate_id)
    refs = set(selected.diagnosis.evidence_refs if selected is not None else [])
    payload = {
        "selection_decision": selection.model_dump(mode="json"),
        "selected_problem_cases": {
            key: item.model_dump(mode="json")
            for key, item in diagnosis.problem_cases.items()
            if not refs or refs.intersection(item.evidence_refs)
        },
        "repository": context.repository.model_dump(mode="json"),
        "run_manifest": context.run_manifest,
        "relevant_evidence_index": [
            item
            for item in _compact_evidence_index(context)
            if not refs or item.get("event_id") in refs
        ],
    }
    return (
        "Expand the frozen selection into one execution-level proposal. Inspect the "
        "real source and call validate_plan before final output.\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )


def _validate_triage_coverage(
    board: DiagnosisBoard,
    context: OrchestratorContext,
) -> list[str]:
    expected = [item.run_id for item in context.current_run_alerts]
    actual = [item.run_id for item in board.alert_dispositions]
    problems = []
    if len(actual) != len(set(actual)):
        problems.append("each current_run_alert may appear only once")
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(context.current_run_ids))
    if missing:
        problems.append("missing current_run_alert dispositions: " + ", ".join(missing))
    if unknown:
        problems.append("dispositions reference non-current or unknown runs: " + ", ".join(unknown))
    mismatched = [
        key for key, problem in board.problem_cases.items() if key != problem.id
    ]
    if mismatched:
        problems.append("problem_cases keys must match ids: " + ", ".join(mismatched))
    return problems


def _validate_selection(
    selection: SelectionDecision,
    context: OrchestratorContext | None = None,
) -> list[str]:
    problems = []
    mismatched = [
        key
        for key, candidate in selection.candidate_backlog.items()
        if key != candidate.id
    ]
    if mismatched:
        problems.append("candidate_backlog keys must match ids: " + ", ".join(mismatched))
    ranked = [item.candidate_id for item in selection.preliminary_ranking]
    if len(ranked) != len(set(ranked)):
        problems.append("preliminary_ranking contains duplicate Candidates")
    unknown_ranked = sorted(set(ranked) - set(selection.candidate_backlog))
    if unknown_ranked:
        problems.append("ranking references unknown Candidates: " + ", ".join(unknown_ranked))
    if selection.decision == "abstain":
        if selection.selected_candidate_id:
            problems.append("abstain must have an empty selected_candidate_id")
    elif selection.selected_candidate_id not in selection.candidate_backlog:
        problems.append("proceed/needs_human must select one declared Candidate")
    comparison = selection.top_two_comparison
    if (
        comparison is not None
        and selection.decision != "abstain"
        and comparison.winner != selection.selected_candidate_id
    ):
        problems.append("Top-2 winner must equal selected_candidate_id")
    if context is not None:
        known_files = set(context.repository.files)
        for candidate_id, candidate in selection.candidate_backlog.items():
            unknown_paths = _unknown_repo_paths(
                candidate.scope.affected_components,
                known_files,
            )
            if unknown_paths:
                problems.append(
                    f"Candidate {candidate_id} references task-workspace/non-repository "
                    "paths instead of target-agent components: "
                    + ", ".join(unknown_paths)
                )
    return problems


def _unknown_repo_paths(items: list[str], known_files: set[str]) -> list[str]:
    """Reject path-shaped task artifacts while allowing abstract component names."""

    suffixes = (".py", ".ts", ".js", ".rs", ".go", ".java", ".toml", ".md")
    return [
        item
        for item in items
        if ("/" in item or item.endswith(suffixes))
        and item not in known_files
        and not any(path.startswith(item.rstrip("/") + "/") for path in known_files)
    ]


def _validate_contract_selection(
    expansion: ContractExpansion,
    selection: SelectionDecision,
) -> list[str]:
    contract = expansion.selected_change_contract
    if contract.backlog_item_id != selection.selected_candidate_id:
        return ["contract backlog_item_id must equal the frozen selection"]
    return []


def _proposal_from_contract(
    selection: SelectionDecision,
    expansion: ContractExpansion,
) -> ImprovementProposal:
    return ImprovementProposal(
        summary=selection.summary,
        problem_statement=selection.problem_statement,
        proposal_guardrails=expansion.proposal_guardrails,
        candidate_backlog=selection.candidate_backlog,
        preliminary_ranking=selection.preliminary_ranking,
        top_two_comparison=selection.top_two_comparison,
        selected_candidate_id=selection.selected_candidate_id,
        selected_change_contract=expansion.selected_change_contract,
        evidence=expansion.evidence,
        goals=expansion.goals,
        non_goals=expansion.non_goals,
        affected_components=expansion.affected_components,
        dependencies=expansion.dependencies,
        benefit=selection.benefit,
        risk=selection.risk,
        effort=selection.effort,
        confidence=selection.confidence,
        decision=selection.decision,
        decision_reason=selection.decision_reason,
        alternatives_considered=expansion.alternatives_considered,
    )


def _attach_frozen_decision(
    proposal: ImprovementProposal,
    diagnosis: DiagnosisBoard,
    selection: SelectionDecision,
) -> ImprovementProposal:
    """Prevent later contract/acceptance repair from changing the chosen problem."""

    return proposal.model_copy(
        update={
            "summary": selection.summary,
            "problem_statement": selection.problem_statement,
            "candidate_backlog": selection.candidate_backlog,
            "preliminary_ranking": selection.preliminary_ranking,
            "top_two_comparison": selection.top_two_comparison,
            "selected_candidate_id": selection.selected_candidate_id,
            "benefit": selection.benefit,
            "risk": selection.risk,
            "effort": selection.effort,
            "confidence": selection.confidence,
            "decision": selection.decision,
            "decision_reason": selection.decision_reason,
            "orchestrator_artifacts": OrchestratorArtifacts(
                diagnosis=diagnosis,
                selection=selection,
            ),
        }
    )


def _abstaining_proposal(selection: SelectionDecision) -> ImprovementProposal:
    return ImprovementProposal(
        summary=selection.summary,
        problem_statement=selection.problem_statement,
        candidate_backlog=selection.candidate_backlog,
        preliminary_ranking=selection.preliminary_ranking,
        top_two_comparison=selection.top_two_comparison,
        selected_candidate_id="",
        benefit=selection.benefit,
        risk=selection.risk,
        effort=selection.effort,
        confidence=selection.confidence,
        decision="abstain",
        decision_reason=selection.decision_reason,
    )


def _evidence_tool(context: OrchestratorContext) -> Tool:
    async def get_evidence(args: dict, _tool_context: ToolContext) -> ToolResult:
        event_id = str(args.get("event_id") or "").strip()
        if not event_id:
            return ToolResult(content="Error: 'event_id' is required.", is_error=True)
        record = context.raw_target_evidence.get(event_id)
        if record is None:
            return ToolResult(
                content=f"Error: target-agent evidence not found: {event_id}",
                is_error=True,
            )
        return ToolResult(content=json.dumps(record, ensure_ascii=False, indent=2))

    return Tool(
        name="get_target_evidence",
        description=(
            "Read one full TARGET AGENT trajectory event by an event_id listed in "
            "OrchestratorContext.evidence_catalog. This never returns AgentReforge "
            "workflow events."
        ),
        parameters=object_schema(
            {
                "event_id": {
                    "type": "string",
                    "description": "Stable target-agent evidence event id",
                }
            },
            required=["event_id"],
        ),
        handler=get_evidence,
    )


def _history_tool(
    context: OrchestratorContext,
    history_index: ImprovementHistoryIndex,
) -> Tool:
    async def search_history(args: dict, _tool_context: ToolContext) -> ToolResult:
        query_text = str(args.get("query") or "").strip()
        if not query_text:
            return ToolResult(content="Error: 'query' is required.", is_error=True)
        matches = history_index.search(
            query_text,
            target_repo=context.run_manifest.get("target") or context.repository.root,
            exclude_run_id=str(context.run_manifest.get("run_id") or ""),
            limit=int(args.get("limit") or 5),
        )
        return ToolResult(
            content=(
                json.dumps(matches, ensure_ascii=False, indent=2)
                if matches
                else "No relevant historical AgentReforge loops found."
            )
        )

    return Tool(
        name="search_improvement_history",
        description=(
            "Search OLDER AgentReforge loop records for similar capability gaps, "
            "interventions, reviewer findings, or delivery outcomes. Never use this "
            "instead of the directly supplied current target run/current-loop facts."
        ),
        parameters=object_schema(
            {
                "query": {
                    "type": "string",
                    "description": "Capability gap, failure pattern, or intervention to find",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches (default 5, maximum 20)",
                },
            },
            required=["query"],
        ),
        handler=search_history,
    )


def _reforge_loop_tool(context: OrchestratorContext) -> Tool:
    async def get_loop(args: dict, _tool_context: ToolContext) -> ToolResult:
        loop_id = str(args.get("loop_id") or "").strip()
        if not loop_id:
            return ToolResult(content="Error: 'loop_id' is required.", is_error=True)
        record = context.raw_reforge_loops.get(loop_id)
        if record is None:
            return ToolResult(
                content=f"Error: AgentReforge loop not found: {loop_id}",
                is_error=True,
            )
        return ToolResult(content=record.model_dump_json(indent=2))

    return Tool(
        name="get_reforge_loop",
        description=(
            "Read the full component-divided record for a previous AgentReforge loop "
            "in THIS recursive run. It is workflow audit data, not target-agent evidence."
        ),
        parameters=object_schema(
            {
                "loop_id": {
                    "type": "string",
                    "description": "Loop id listed in previous_reforge_loops",
                }
            },
            required=["loop_id"],
        ),
        handler=get_loop,
    )
