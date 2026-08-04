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
    CaseAnalysis,
    CaseAnalysisBoard,
    ContractExpansion,
    DeliveryCaseDefinition,
    DeliveryExecutionDesign,
    DeliveryScenario,
    DiagnosisBoard,
    ImprovementProposal,
    OrchestratorArtifacts,
    ScenarioObservation,
    SelectionDecision,
    WorkflowDiagnosisBoard,
)
from .plan_validator import validate_plan_tool
from .records import ReforgeLoopRecord

ORCHESTRATOR_PROMPT = """You are the improvement Orchestrator: an analyst, not an implementer.

You receive typed context artifacts:
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
2. `previous_reforge_loops`: prior recursive-run records used for history and progress.

For a coding agent, distinguish the TARGET AGENT REPOSITORY from a TASK WORKSPACE.
The target repository contains the agent system being improved; each target-agent run
may ask that same agent version to edit a different disposable repository. Different
task files do not make the run irrelevant or turn it into another agent's history.
`target_commit` identifies the agent version whose capability the run measures. A
failure in an idempotency, algorithm, CLI, or other task workspace is current evidence
about that target agent when its `target_commit` is current. Never describe generated
task artifacts as delivered improvements to another repository.
Crucially, a test failure, traceback, command, flag, schema, or interface mentioned
inside a target-agent run describes the GENERATED TASK ARTIFACT by default, not the
target agent's own implementation. First label the artifact origin internally:
`agent_repository`, `task_workspace`, or `execution_environment`. A missing CLI flag
in code the target agent generated is evidence that the agent failed to understand,
implement, or verify a requirement; it is NOT evidence that the agent repository
should add that flag to its own CLI.

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
- Translate task-workspace symptoms into agent-level causes before creating a problem
  case. For example, "generated service omitted a required option" may support a
  requirement-assimilation or end-to-end-verification gap. It never directly supports
  adding that option to the target agent's own entrypoint.
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
  checks, `delivery_run`, and the frozen `delivery_scenario` participate in Delivery.
- One Loop implements one Candidate through one Task and at most ONE primary
  `delivery_scenario`. Multiple inputs, commands, and behavioral checks belong inside
  that one Scenario; they are not separate Scenarios. For a target agent with a
  runnable CLI, design that Scenario to exercise the selected capability end to end.
  It contains a structured `task_contract`, a safe argv command, a small isolated
  fixture, structured component observations, and observable expected/forbidden
  behaviors. Design scenarios before Writer runs; do not adapt acceptance difficulty
  after seeing the implementation.
- Derive each Scenario from the selected Candidate's evaluability scorecard. A Scenario
  must name an observable difference that the intervention is expected to cause. If the
  baseline likely passes it unchanged, treat it as contract-compliance evidence only and
  do not claim it demonstrates improvement; strengthen the Scenario or lower the
  Candidate's discriminability and rank.
- The primary Scenario must exercise the Candidate's positive capability path. Do not
  manufacture an empty, impossible, or missing-dependency workspace and then treat the
  target agent's refusal/blocker report as proof of improved task execution or
  verification. A negative case may prove an error-handling guardrail, but cannot by
  itself demonstrate a Candidate whose expected delta is successful implementation,
  verification, or completion.
- Scenario commands are argv arrays, never shell strings. Inspect the real target
  entrypoint and match its actual CLI. Use `{prompt}` only when that CLI accepts a
  natural-language task argument; use `{workspace}` only when it accepts a workspace
  path. A prompt also describes the scenario to the Deliverer and does not have to be
  passed to the process. Keep scenarios
  bounded and directly causal: they decide Delivery for this Candidate, not general
  benchmark quality. Optional later diagnostic probes may explain failure but cannot
  change the frozen pass/fail scenarios.
- Define `observations` as a small fill-in table: component, expected behavior, and
  admissible evidence sources. Set `requires_trajectory=true` whenever a scenario must
  prove internal tool use,
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
    inputs: [{id, description}],
    expected_outputs: [{
      id, description, rationale, evidence_direction
    }],
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
- selected_change_contract.delivery_scenarios[]: zero or one {id: str,
  task_contract: {
    objective: str, context: [str],
    requirements: [{id: str, description: str}],
    constraints: [{id: str, description: str}],
    outcome_contract: {
      primary_success: [{id: str, description: str, rationale: str,
        evidence_direction: str}],
      guardrails: [{id: str, description: str, rationale: str,
        evidence_direction: str}],
      inconclusive: [{id: str, description: str, rationale: str,
        evidence_direction: str}]
    },
    acceptance_checks: [{id: str, description: str}]
  },
  command: [argv strings, optionally containing {prompt} and/or {workspace} only
    when required by the target entrypoint],
  fixture_files: {repo_relative_path: content},
  environment_requirements: [str], baseline_prediction: str,
  candidate_prediction: str, observable_difference: str,
  expected_behaviors: [str], forbidden_behaviors: [str],
  observations: [{id: str, component: str, expected_behavior: str,
    evidence_sources: [agent_input|agent_output|tool_call|tool_result|artifact|final_output|log]}],
  budgets: {max_agent_turns: int, max_action_steps: int},
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


WORKFLOW_TRIAGE_PROMPT = """You are the Workflow Analysis stage inside AgentReforge's
Orchestrator. Analyze only AgentReforge's own prior Loop execution records: planning,
schema handoffs, Writer/Reviewer coordination, Delivery, rollback, and commit state.
You do not receive target-agent trajectories or target source, and you do not generate
target-agent improvement Candidates.

For every supplied Loop, classify what the record establishes. A Loop may contain a
valid target hypothesis while also failing because AgentReforge's interface or
execution malfunctioned. List Candidate ids that are workflow-only artifacts in
`target_candidate_exclusions`; do not exclude a valid target hypothesis merely because
its implementation attempt failed.

Output ONLY one WorkflowDiagnosisBoard JSON object:
{
  "loop_assessments": [{
    "loop_id": str,
    "stage": str,
    "classification": str,
    "observed_facts": [str],
    "related_candidate_ids": [str],
    "explanation": str
  }],
  "workflow_issues": {issue_id: {
    "id": same issue_id,
    "symptom": str,
    "capability_gap": str,
    "likely_root_causes": [str],
    "affected_scope": [str],
    "evidence_refs": [str],
    "uncertainty": str
  }},
  "target_candidate_exclusions": {candidate_id: reason},
  "whole_picture_summary": str
}
`classification` is one of: delivered, target_attempt_failed,
workflow_interface_failed, workflow_execution_failed, mixed, or inconclusive."""


CASE_ANALYSIS_PROMPT = """You are the Case Analysis stage inside AgentReforge's
Orchestrator. Analyze exactly ONE run of the TARGET AGENT on a disposable TASK
WORKSPACE. Translate what the agent actually did, what code/artifacts it produced, how
the resulting behavior flows, where the run succeeded or failed, and what the evidence
does and does not imply.

Call `get_target_run_trajectory` to inspect the complete ordered target-agent
trajectory. Keep target-agent actions distinct from AgentReforge workflow events.
Final-response prose is self-report; tool calls/results, artifacts, evaluation output,
and ordered trajectory events are stronger evidence. Do not infer code behavior only
from filenames or a final summary.

Separate these ownership layers:
- `target_agent`: a reusable planning/tool-use/state/verification capability;
- `generated_artifact`: code produced in the disposable Task Workspace;
- `task_definition`: an ambiguous, impossible, or non-discriminating requested task;
- `environment`: dependencies, permissions, credentials, or runtime infrastructure;
- `agentreforge_workflow`: evidence capture or execution failed outside the target;
- `none` or `uncertain` when no failure or no attribution is supported.
A missing flag/schema/algorithm in generated task code is an artifact symptom. It may
support a reusable target-agent capability hypothesis, but it is never a request to add
that task-specific feature to the target agent repository.

Output ONLY one CaseAnalysis JSON object:
{
  "case_id": "case:<run_id>",
  "run_id": "the exact supplied run_id",
  "target_commit": str,
  "task_intent": {"objective": str, "requirements": [str]},
  "artifact_analysis": {
    "summary": str,
    "changed_components": [str],
    "implemented_behavior": [str],
    "behavior_flow": [str]
  },
  "final_status": "success|partial|failed|inconclusive",
  "steps": [{
    "step": str, "expected": str, "observed": str,
    "status": "satisfied|failed|partial|not_observed",
    "evidence_refs": [str]
  }],
  "final_outcome": str,
  "causal_analysis": {
    "failure_layer": "one declared ownership enum value",
    "direct_cause": str,
    "contributing_causes": [str],
    "alternative_hypotheses": [{
      "hypothesis": str, "evidence_for": [str], "evidence_against": [str]
    }],
    "uncertainty": str
  },
  "capability_signals": [{
    "capability": str, "signal": "positive|negative|uncertain",
    "evidence_refs": [str], "generalizability": str
  }],
  "improvement_opportunities": [{
    "area": str, "rationale": str, "expected_effect": str
  }]
}
Improvement opportunities are diagnostic possibilities, not selected Candidates or
implementation instructions."""


TRIAGE_PROMPT = """You are the Target-Agent Analysis stage inside one
improvement Orchestrator.
You diagnose the target agent; you do not propose code changes, rank solutions, design
tests, or write an ImprovementProposal.

Case Analysis has already translated each current target-agent run into ordered
actions, artifact behavior, causal hypotheses, and capability signals. Treat its Git
identity fields as deterministic facts. Use the CaseAnalysisBoard as the primary
decision input; inspect cited raw evidence only to resolve a material ambiguity.
Failure of a generated artifact can be capability evidence about that target version,
but preserve Case Analysis ownership and uncertainty.
The task artifact's missing feature is a SYMPTOM, not the improvement itself. Diagnose
the reusable TARGET AGENT capability that caused incomplete work (for example planning,
tool use, state tracking, cross-component completion, or verification budgeting).
Never propose adding the task's domain feature or task-workspace files to the target
agent repository. `ProblemCase.affected_scope` names only target-agent source components
or abstract agent capabilities visible in the supplied repository map.

Audit every supplied current-run case. For each failed, partial, or inconclusive case,
record its observed failure, agent-level interpretation, plausible direct causes,
evidence references, and exactly one disposition. A case may be deferred when evidence
cannot identify an actionable agent-level problem. Successful cases remain whole-picture
capability evidence but do not require an AlertDisposition.

Aggregate across cases instead of converting every local artifact defect into a new
capability problem. Use the whole source-visible architecture and supplied target
capability backlog. Use read-only tools to inspect source or cited evidence when
needed. Output ONLY one DiagnosisBoard JSON object:
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

Keep these objects distinct throughout selection:
- AGENT UNDER IMPROVEMENT: the reusable system whose `target_commit` is evaluated.
- AGENT REPOSITORY: the source Writer may improve.
- EVALUATION TASK: the coding problem used to expose capability evidence.
- TASK WORKSPACE: the disposable project the target agent edits during that task.
- DELIVERY SCENARIO: prompt + task workspace + environment + expected evidence.
Task-specific names, flags, schemas, algorithms, and business rules belong to the
Evaluation Task/Task Workspace. They are symptoms and test material, not candidate
features. Before ranking an intervention, apply a transfer counterfactual: if the
task domain, filenames, and interface names were changed, would the intervention still
improve the agent? If not, classify it as task overfitting and do not select it.
Do not preserve a task-specific noun while merely calling it "contract fidelity":
the Candidate title, mechanism, expected delta, and affected components must describe
the reusable agent mechanism. In particular, never propose adding an evaluation
task's flag or interface to the target agent's own CLI. Trace each cited failure to
the repository that actually owned the failing artifact before choosing scope.

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
diagnosis/intervention snapshots, inputs, expected outputs, behaviors, constraints,
invariants, prohibited shortcuts, affected components, concise allowed write
suggestions, reviewer focus, and safety properties.

The Contract is guidance for Writer/Reviewer/Deliverer, not a hard file boundary.
The Writer edits the AGENT REPOSITORY. The Scenario runs that agent against a disposable
TASK WORKSPACE. Never convert a task-workspace symptom (a flag, schema, algorithm,
service, or business rule) into a feature of the agent repository. The Contract's
mechanism must remain useful when task domain names and interfaces are replaced.
A generic label such as "contract fidelity" does not make a task-specific interface
reusable. Do not introduce task-specific detail during expansion.
When downstream control flow must distinguish completed, blocked, incomplete, or
verification-failed outcomes, require a typed/structured status separate from the
human-readable summary. Never design control flow around broad keyword matching in
free-form final prose.

Do NOT design acceptance criteria, delivery commands, Delivery Scenarios, a delivery
checklist, or rollback plan in this stage. A separate Delivery Design stage receives
this frozen Contract. Leave those defaulted fields absent. Safety properties are empty
by default; declare `path_confinement` only when the selected change touches that
boundary.

Each evidence item is an object with source_type
(trajectory|test|benchmark|code|log), reference, and observation.

Output ONLY one ContractExpansion JSON object:
{
  "proposal_guardrails": [{"id": str, "description": str}],
  "selected_change_contract": {
    "contract_id": str,
    "backlog_item_id": "the exact frozen selected_candidate_id",
    "backlog_revision": int,
    "objective": str,
    "rationale": str,
    "diagnosis": "the selected Candidate diagnosis object unchanged",
    "intervention": "the selected Candidate intervention object unchanged",
    "inputs": [{"id": str, "description": str}],
    "expected_outputs": [{
      "id": str,
      "description": "the concrete externally or internally observable result",
      "rationale": "why this result demonstrates the selected capability",
      "evidence_direction": "what evidence should become present, absent, increase, or decrease"
    }],
    "required_behaviors": [{"id": str, "description": str}],
    "implementation_constraints": [{"id": str, "description": str}],
    "invariants": [{"id": str, "description": str}],
    "prohibited_shortcuts": [{"id": str, "description": str}],
    "affected_components": [str],
    "allowed_write_paths": [str],
    "reviewer_focus": [str],
    "required_safety_properties": []
  },
  "evidence": [{"source_type": "trajectory|test|benchmark|code|log",
    "reference": str, "observation": str}],
  "goals": [str], "non_goals": [str], "affected_components": [str],
  "dependencies": [str], "alternatives_considered": [str]
}
The coordinator—not you—joins this artifact with the frozen SelectionDecision."""


SCENARIO_DEFINITION_PROMPT = """You are the Scenario Definition stage inside one
improvement Orchestrator. Selection and the Change Contract are frozen. Define WHAT
single representative coding case would causally distinguish the selected target-agent
capability. Do not decide HOW to launch or materialize it.

The case is an ordinary product-code task for the TARGET AGENT to solve in a disposable
TASK WORKSPACE. It is never a request for the agent to modify, test, or copy its own
agent repository. Define one case with enough internal requirements to expose the
capability. Do not emit commands, repository paths, source-file contents, fixture
files, environment setup, budgets, or implementation components from the target-agent
repository.

Fill `task_contract` as a structured form: objective, context, numbered requirements,
constraints, outcome_contract, and acceptance checks. Each outcome condition needs an
id, description, rationale, and evidence_direction. Primary success establishes the
positive outcome; guardrails can veto but cannot prove success; inconclusive conditions
say when the run cannot support a conclusion.

Describe abstract workspace roles in `workspace_requirements`, such as "existing
implementation", "focused tests", or "project metadata". Describe the causal contrast
in `comparison`, including confounders. Each `evidence_requirement` is a semantic claim,
classified as `target_agent_action`, `task_artifact`, or `environment_result`, and
references the outcome-condition or acceptance-check ids it supports. Do not choose
concrete event sources here.

Keep this handoff compact enough to remain atomic: the serialized JSON must be under
4,000 characters. Use concise sentences and do not repeat Contract prose. Use at most
3 context facts, 3 requirements, 2 constraints, 2 primary-success conditions,
1 guardrail, 1 inconclusive condition, 3 acceptance checks, 3 workspace requirements,
and 4 evidence requirements. These are protocol size limits, not scoring rules.

Output ONLY one DeliveryCaseDefinition JSON object:
{
  "contract_id": "the exact frozen contract_id",
  "case_id": str,
  "capability_under_test": str,
  "task_contract": "ScenarioTaskContract object",
  "workspace_requirements": [{
    "id": str, "role": str, "required_initial_state": str, "purpose": str
  }],
  "comparison": {
    "baseline_prediction": str, "candidate_prediction": str,
    "observable_difference": str, "confounders": [str]
  },
  "evidence_requirements": [{
    "id": str, "claim": str,
    "subject": "target_agent_action|task_artifact|environment_result",
    "related_condition_ids": [str]
  }]
}"""


SCENARIO_MATERIALIZATION_PROMPT = """You are the Scenario Materialization stage inside
one improvement Orchestrator. The Change Contract and Delivery Case Definition are
frozen. Design only HOW to execute that exact case. Never change its capability,
task_contract, predictions, outcome conditions, or evidence claims.

Materialize the abstract workspace requirements as literal executable source, tests,
and project metadata in `workspace_seed_files`. The files belong only to one disposable
ordinary CODING TASK WORKSPACE. Never copy the target-agent package or its affected
components into that workspace. The Scenario command must launch the real TARGET AGENT
from the candidate AGENT REPOSITORY and direct it to edit the disposable Task Workspace.

`command` is an argv array beginning with an executable. It must include `{prompt}` and
`{workspace}` exactly where the real target-agent CLI accepts the task and workspace;
never replace the structured task with a shorter literal prompt, and never put shell
syntax such as `NAME=value` in argv. Runtime prerequisites belong in
`environment_requirements`. Bind every frozen evidence requirement exactly once in
`observation_bindings`. Evidence sources accept ONLY `agent_input`, `agent_output`,
`tool_call`, `tool_result`, `artifact`, `final_output`, or `log`. Use
`tool_call`/`tool_result` for internal actions and `artifact` for generated code; never
use `trajectory` or `code`.

Use budgets to separate total model turns from meaningful write/run actions; reads and
searches do not consume max_action_steps. Every `delivery_run` item is a directly
executable shell command in the candidate Agent Repository and cannot contain
`{workspace}` or `{prompt}`. Acceptance commands also run in the candidate repository
and cannot contain those placeholders. The primary Scenario is the only execution
against the disposable workspace. Prose belongs in `delivery_checklist`.

Each acceptance criterion contains id, description, mode
(red_green|invariant|metric_improvement|non_regression|manual), check_type
(unit|integration|smoke), verification (command|review|manual), command,
expected_exit_code, required_output_contains, forbidden_output_contains,
verified_safety_properties, test_level (full|focused|basic), and required.
`verified_safety_properties` must be empty unless the frozen Contract declares it.

Output ONLY one DeliveryExecutionDesign JSON object:
{
  "contract_id": "the exact frozen contract_id",
  "case_id": "the exact frozen case_id",
  "acceptance_criteria": ["AcceptanceCriterion objects"],
  "delivery_run": [str],
  "scenario_execution": {
    "command": [str],
    "workspace_seed_files": {"relative/path": "literal file content"},
    "environment_requirements": [str],
    "observation_bindings": [{
      "evidence_requirement_id": str, "component": str,
      "evidence_sources": [
        "agent_input|agent_output|tool_call|tool_result|artifact|final_output|log"
      ]
    }],
    "budgets": {"max_agent_turns": int, "max_action_steps": int},
    "executable_conditions": ["ExecutableCondition objects"]
  },
  "delivery_checklist": [str],
  "rollback_plan": str
}"""


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
        self.last_workflow_diagnosis: WorkflowDiagnosisBoard | None = None
        self.last_case_analysis: CaseAnalysisBoard | None = None
        self.last_diagnosis: DiagnosisBoard | None = None
        self.last_selection: SelectionDecision | None = None
        self.last_contract_expansion: ContractExpansion | None = None
        self.last_case_definition: DeliveryCaseDefinition | None = None
        self.last_delivery_execution: DeliveryExecutionDesign | None = None
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
        # Stage tool surfaces follow the same boundary as their input payloads.
        # Workflow Analysis can drill into Reforge Loops but cannot inspect target
        # evidence/source. Target stages can inspect target evidence/source but
        # cannot read workflow records.
        target_registry = self.registry
        target_registry.register(_evidence_tool(context))
        if context.current_run_ids:
            target_registry.register(_target_run_trajectory_tool(context))
        workflow_registry = ToolRegistry()
        workflow_registry.register(_reforge_loop_tool(context))

        workflow_request = _build_workflow_triage_request(context)
        workflow_text = await self._investigate_stage(
            system_prompt=WORKFLOW_TRIAGE_PROMPT,
            user_message=workflow_request,
            max_turns=max_turns,
            registry=workflow_registry,
        )
        workflow_diagnosis = await self._parse_stage_with_repair(
            workflow_text,
            WorkflowDiagnosisBoard,
            system_prompt=WORKFLOW_TRIAGE_PROMPT,
            stage_name="WorkflowDiagnosisBoard",
            validate=lambda board: _validate_workflow_diagnosis(board, context),
            repair_context=workflow_request,
        )
        self.last_workflow_diagnosis = workflow_diagnosis

        cases: dict[str, CaseAnalysis] = {}
        for target_run in context.target_agent_runs:
            if not target_run.is_current:
                continue
            case_request = _build_case_analysis_request(context, target_run.run_id)
            case_text = await self._investigate_stage(
                system_prompt=CASE_ANALYSIS_PROMPT,
                user_message=case_request,
                max_turns=max_turns,
                registry=target_registry,
            )
            case = await self._parse_stage_with_repair(
                case_text,
                CaseAnalysis,
                system_prompt=CASE_ANALYSIS_PROMPT,
                stage_name=f"CaseAnalysis[{target_run.run_id}]",
                validate=lambda item, run_id=target_run.run_id: (
                    _validate_case_analysis(item, run_id)
                ),
                repair_context=case_request,
            )
            cases[target_run.run_id] = case
        case_analysis = CaseAnalysisBoard(cases=cases)
        self.last_case_analysis = case_analysis

        triage_request = _build_triage_request(
            context,
            workflow_diagnosis,
            case_analysis,
        )
        triage_text = await self._investigate_stage(
            system_prompt=TRIAGE_PROMPT,
            user_message=triage_request,
            max_turns=max_turns,
            registry=target_registry,
        )
        diagnosis = await self._parse_stage_with_repair(
            triage_text,
            DiagnosisBoard,
            system_prompt=TRIAGE_PROMPT,
            stage_name="DiagnosisBoard",
            validate=lambda board: _validate_triage_coverage(board, context),
            repair_context=triage_request,
        )
        self.last_diagnosis = diagnosis

        selection_request = _build_selection_request(
            context,
            diagnosis,
            workflow_diagnosis,
        )
        selection_text = await self._investigate_stage(
            system_prompt=SELECTION_PROMPT,
            user_message=selection_request,
            max_turns=max_turns,
            registry=target_registry,
        )
        selection = await self._parse_stage_with_repair(
            selection_text,
            SelectionDecision,
            system_prompt=SELECTION_PROMPT,
            stage_name="SelectionDecision",
            validate=lambda item: _validate_selection(item, context),
            repair_context=selection_request,
        )
        self.last_selection = selection

        if selection.decision == "abstain":
            proposal = _abstaining_proposal(selection)
        else:
            contract_request = _build_contract_request(context, diagnosis, selection)
            contract_text = await self._investigate_stage(
                system_prompt=CONTRACT_PROMPT,
                user_message=contract_request,
                max_turns=max_turns,
                registry=target_registry,
            )
            expansion = await self._parse_stage_with_repair(
                contract_text,
                ContractExpansion,
                system_prompt=CONTRACT_PROMPT,
                stage_name="ContractExpansion",
                validate=lambda item: _validate_contract_selection(item, selection),
                repair_context=contract_request,
            )
            self.last_contract_expansion = expansion
            case_definition_request = _build_scenario_definition_request(
                context,
                diagnosis,
                selection,
                expansion,
            )
            case_definition_text = await self._complete_typed_stage(
                system_prompt=SCENARIO_DEFINITION_PROMPT,
                user_message=case_definition_request,
            )
            case_definition = await self._parse_stage_with_repair(
                case_definition_text,
                DeliveryCaseDefinition,
                system_prompt=SCENARIO_DEFINITION_PROMPT,
                stage_name="DeliveryCaseDefinition",
                validate=lambda item: _validate_case_definition(item, expansion),
                repair_context=case_definition_request,
            )
            self.last_case_definition = case_definition

            execution_request = _build_scenario_materialization_request(
                context,
                expansion,
                case_definition,
            )
            execution_text = await self._complete_typed_stage(
                system_prompt=SCENARIO_MATERIALIZATION_PROMPT,
                user_message=execution_request,
            )
            delivery_execution = await self._parse_stage_with_repair(
                execution_text,
                DeliveryExecutionDesign,
                system_prompt=SCENARIO_MATERIALIZATION_PROMPT,
                stage_name="DeliveryExecutionDesign",
                validate=lambda item: _validate_delivery_execution(
                    item,
                    expansion,
                    case_definition,
                ),
                repair_context=execution_request,
            )
            self.last_delivery_execution = delivery_execution
            expansion = _attach_delivery_execution(
                expansion,
                case_definition,
                delivery_execution,
            )
            proposal = _proposal_from_contract(selection, expansion)

        # Decision artifacts are coordinator-owned and cannot drift during
        # contract expansion. The final Proposal remains the unchanged downstream
        # interface used by Gate, Writer, Reviewer, and Deliverer.
        return _attach_frozen_decision(
            proposal,
            diagnosis,
            selection,
            workflow_diagnosis=workflow_diagnosis,
            case_analysis=case_analysis,
        )

    async def revise(self, proposal: ImprovementProposal, problem: str) -> ImprovementProposal:
        """Re-generate the proposal to fix a specific problem (e.g. an invalid DAG)."""
        # feed the problem + the previous proposal back; ask for a corrected one.
        msg = json.dumps(
            {
                "request_kind": "revise_improvement_proposal",
                "problem": problem,
                "previous_proposal": proposal.model_dump(mode="json"),
                "revision_constraints": [
                    "Fix the reported problem and keep all other valid fields stable.",
                    (
                        "For a missing safety-property mapping, update both the "
                        "criterion verified_safety_properties and the Selected "
                        "Change Contract; system-owned probes have empty commands."
                    ),
                    "The Contract is the only execution unit and has no Task DAG.",
                ],
            },
            ensure_ascii=False,
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
            workflow_diagnosis=artifacts.workflow_diagnosis,
            case_analysis=artifacts.case_analysis,
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
        repair_context: str = "",
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
                    schema=model_cls.model_json_schema(),
                    immutable_context=repair_context,
                )
        raise AssertionError("unreachable")

    async def _investigate_stage(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_turns: int,
        registry: ToolRegistry | None = None,
    ) -> str:
        text = ""
        async for event in query(
            client=self.client,
            registry=registry or self.registry,
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

    async def _complete_typed_stage(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Generate a pure typed transformation without ReAct text/tool turns."""

        return await collect_text(
            self.client,
            [Message(role="user", content=user_message)],
            system_prompt=system_prompt,
        )

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
        msg = json.dumps(
            {
                "request_kind": "repair_improvement_proposal",
                "validation_error": error,
                "previous_invalid_output": bad_output,
                "required_output": "ImprovementProposal",
            },
            ensure_ascii=False,
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
        schema: dict[str, Any],
        immutable_context: str = "",
    ) -> str:
        output_constraint = (
            "Return one complete compact JSON object under 4,000 characters; "
            "remove repetition before removing required fields."
            if stage_name == "DeliveryCaseDefinition"
            else "Return one complete JSON object and no surrounding prose."
        )
        msg = json.dumps(
            {
                "request_kind": "repair_orchestrator_stage",
                "stage_name": stage_name,
                "validation_error": error,
                "previous_invalid_output": bad_output,
                "authoritative_json_schema": schema,
                "immutable_stage_input": immutable_context[:48_000],
                "repair_scope": (
                    "Preserve valid judgments and fix only the reported schema "
                    "or handoff problem."
                ),
                "output_constraint": output_constraint,
            },
            ensure_ascii=False,
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

    return json.dumps(
        {
            "request_kind": "legacy_full_orchestration",
            "instructions": (
                "Follow phases 1-5, inspect cited source/evidence, and return "
                "ImprovementProposal."
            ),
            "orchestrator_context": context.model_dump(mode="json"),
        },
        ensure_ascii=False,
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


def _build_workflow_triage_request(context: OrchestratorContext) -> str:
    return json.dumps(
        {
            "request_kind": "workflow_analysis",
            "previous_reforge_loops": [
                item.model_dump(mode="json") for item in context.previous_reforge_loops
            ],
            "run_manifest": context.run_manifest,
        },
        ensure_ascii=False,
    )


def _build_case_analysis_request(
    context: OrchestratorContext,
    run_id: str,
) -> str:
    target_run = next(
        item for item in context.target_agent_runs if item.run_id == run_id
    )
    return json.dumps(
        {
            "request_kind": "case_analysis",
            "target_run_semantics": context.target_run_semantics,
            "current_target_commit": context.current_target_commit,
            "target_run": target_run.model_dump(mode="json"),
            "evidence_index": [
                item
                for item in _compact_evidence_index(context)
                if item.get("run_id") == run_id
            ],
            "required_tool_call": {
                "tool": "get_target_run_trajectory",
                "run_id": run_id,
                "purpose": (
                    "Inspect all ordered target-agent actions and results before "
                    "interpreting the case."
                ),
            },
        },
        ensure_ascii=False,
    )


def _target_backlog(
    context: OrchestratorContext,
    workflow: WorkflowDiagnosisBoard,
) -> dict[str, Any]:
    excluded = set(workflow.target_candidate_exclusions)
    return {
        key: item.model_dump(mode="json")
        for key, item in context.improvement_backlog.items()
        if key not in excluded
    }


def _build_triage_request(
    context: OrchestratorContext,
    workflow: WorkflowDiagnosisBoard,
    case_analysis: CaseAnalysisBoard,
) -> str:
    payload = {
        "request_kind": "target_analysis",
        "improvement_intent": context.improvement_intent,
        "target_run_semantics": context.target_run_semantics,
        "current_target_commit": context.current_target_commit,
        "current_run_ids": context.current_run_ids,
        "required_disposition_run_ids": [
            item.run_id for item in context.current_run_alerts
        ],
        "case_analysis_board": case_analysis.model_dump(mode="json"),
        "improvement_backlog": _target_backlog(context, workflow),
        "repository": context.repository.model_dump(mode="json"),
        "run_manifest": context.run_manifest,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_selection_request(
    context: OrchestratorContext,
    diagnosis: DiagnosisBoard,
    workflow: WorkflowDiagnosisBoard,
) -> str:
    payload = {
        "request_kind": "candidate_selection",
        "improvement_intent": context.improvement_intent,
        "diagnosis_board": diagnosis.model_dump(mode="json"),
        "improvement_backlog": _target_backlog(context, workflow),
        "repository": context.repository.model_dump(mode="json"),
        "run_manifest": context.run_manifest,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_contract_request(
    context: OrchestratorContext,
    diagnosis: DiagnosisBoard,
    selection: SelectionDecision,
) -> str:
    selected = selection.candidate_backlog.get(selection.selected_candidate_id)
    refs = set(selected.diagnosis.evidence_refs if selected is not None else [])
    payload = {
        "request_kind": "contract_expansion",
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
    return json.dumps(payload, ensure_ascii=False)


def _build_scenario_definition_request(
    context: OrchestratorContext,
    diagnosis: DiagnosisBoard,
    selection: SelectionDecision,
    expansion: ContractExpansion,
) -> str:
    selected = selection.candidate_backlog.get(selection.selected_candidate_id)
    refs = set(selected.diagnosis.evidence_refs if selected is not None else [])
    return json.dumps(
        {
            "request_kind": "scenario_definition",
            "selected_change_contract": {
                "contract_id": expansion.selected_change_contract.contract_id,
                "objective": expansion.selected_change_contract.objective,
                "rationale": expansion.selected_change_contract.rationale,
                "diagnosis": expansion.selected_change_contract.diagnosis.model_dump(
                    mode="json"
                ),
                "intervention": (
                    expansion.selected_change_contract.intervention.model_dump(
                        mode="json"
                    )
                ),
                "expected_outputs": [
                    item.model_dump(mode="json")
                    for item in expansion.selected_change_contract.expected_outputs
                ],
                "required_behaviors": [
                    item.model_dump(mode="json")
                    for item in expansion.selected_change_contract.required_behaviors
                ],
                "implementation_constraints": [
                    item.model_dump(mode="json")
                    for item in (
                        expansion.selected_change_contract.implementation_constraints
                    )
                ],
                "invariants": [
                    item.model_dump(mode="json")
                    for item in expansion.selected_change_contract.invariants
                ],
                "prohibited_shortcuts": [
                    item.model_dump(mode="json")
                    for item in expansion.selected_change_contract.prohibited_shortcuts
                ],
                "required_safety_properties": (
                    expansion.selected_change_contract.required_safety_properties
                ),
            },
            "proposal_context": {
                "guardrails": [
                    item.model_dump(mode="json")
                    for item in expansion.proposal_guardrails
                ],
                "goals": expansion.goals,
                "non_goals": expansion.non_goals,
            },
            "selected_problem_cases": {
                key: item.model_dump(mode="json")
                for key, item in diagnosis.problem_cases.items()
                if not refs or refs.intersection(item.evidence_refs)
            },
            "relevant_evidence_index": [
                item
                for item in _compact_evidence_index(context)
                if not refs or item.get("event_id") in refs
            ],
        },
        ensure_ascii=False,
    )


def _build_scenario_materialization_request(
    context: OrchestratorContext,
    expansion: ContractExpansion,
    case_definition: DeliveryCaseDefinition,
) -> str:
    return json.dumps(
        {
            "request_kind": "scenario_materialization",
            "frozen_case_definition": case_definition.model_dump(mode="json"),
            "target_agent_repository": context.repository.model_dump(mode="json"),
            "run_manifest": context.run_manifest,
            "declared_safety_properties": (
                expansion.selected_change_contract.required_safety_properties
            ),
        },
        ensure_ascii=False,
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


def _validate_case_analysis(
    case: CaseAnalysis,
    run_id: str,
) -> list[str]:
    problems = []
    if case.run_id != run_id:
        problems.append(f"run_id must preserve the supplied run id {run_id!r}")
    if case.case_id != f"case:{run_id}":
        problems.append(f"case_id must equal 'case:{run_id}'")
    return problems


def _validate_workflow_diagnosis(
    board: WorkflowDiagnosisBoard,
    context: OrchestratorContext,
) -> list[str]:
    known_loops = {item.loop_id for item in context.previous_reforge_loops}
    assessed = [item.loop_id for item in board.loop_assessments]
    problems = []
    if len(assessed) != len(set(assessed)):
        problems.append("each prior Loop may appear only once in loop_assessments")
    unknown = sorted(set(assessed) - known_loops)
    if unknown:
        problems.append("workflow analysis references unknown Loops: " + ", ".join(unknown))
    mismatched = [
        key for key, issue in board.workflow_issues.items() if key != issue.id
    ]
    if mismatched:
        problems.append("workflow_issues keys must match ids: " + ", ".join(mismatched))
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
    return problems


def _validate_contract_selection(
    expansion: ContractExpansion,
    selection: SelectionDecision,
) -> list[str]:
    contract = expansion.selected_change_contract
    if contract.backlog_item_id != selection.selected_candidate_id:
        return ["contract backlog_item_id must equal the frozen selection"]
    return []


def _validate_case_definition(
    definition: DeliveryCaseDefinition,
    expansion: ContractExpansion,
) -> list[str]:
    problems = []
    if definition.contract_id != expansion.selected_change_contract.contract_id:
        problems.append(
            "case definition contract_id must equal the frozen Change Contract id"
        )
    condition_ids = {
        item.id
        for group in (
            definition.task_contract.outcome_contract.primary_success,
            definition.task_contract.outcome_contract.guardrails,
            definition.task_contract.outcome_contract.inconclusive,
        )
        for item in group
    }
    condition_ids.update(
        item.id for item in definition.task_contract.acceptance_checks
    )
    requirement_ids = [item.id for item in definition.evidence_requirements]
    if len(requirement_ids) != len(set(requirement_ids)):
        problems.append("evidence requirement ids must be unique")
    unknown_conditions = sorted(
        {
            condition_id
            for requirement in definition.evidence_requirements
            for condition_id in requirement.related_condition_ids
            if condition_id not in condition_ids
        }
    )
    if unknown_conditions:
        problems.append(
            "evidence requirements reference unknown condition ids: "
            + ", ".join(unknown_conditions)
        )
    return problems


def _validate_delivery_execution(
    design: DeliveryExecutionDesign,
    expansion: ContractExpansion,
    definition: DeliveryCaseDefinition,
) -> list[str]:
    problems = []
    if design.contract_id != expansion.selected_change_contract.contract_id:
        problems.append(
            "delivery execution contract_id must equal the frozen Change Contract id"
        )
    if design.case_id != definition.case_id:
        problems.append("delivery execution case_id must equal the frozen case id")
    declared_safety = set(
        expansion.selected_change_contract.required_safety_properties
    )
    undeclared_safety = sorted(
        {
            safety
            for criterion in design.acceptance_criteria
            for safety in criterion.verified_safety_properties
            if safety not in declared_safety
        }
    )
    if undeclared_safety:
        problems.append(
            "acceptance criteria reference undeclared safety properties: "
            + ", ".join(undeclared_safety)
        )
    scenario = design.scenario_execution
    if scenario is None:
        problems.append("the frozen Delivery Case requires one scenario execution")
        return problems
    flattened_command = "\n".join(scenario.command)
    if "{prompt}" not in flattened_command:
        problems.append(
            "scenario command must pass the frozen task contract via {prompt}"
        )
    if "{workspace}" not in flattened_command:
        problems.append(
            "scenario command must pass the disposable Task Workspace via {workspace}"
        )
    expected_bindings = {item.id for item in definition.evidence_requirements}
    actual_bindings = [
        item.evidence_requirement_id for item in scenario.observation_bindings
    ]
    if len(actual_bindings) != len(set(actual_bindings)):
        problems.append("each evidence requirement may be bound only once")
    missing_bindings = sorted(expected_bindings - set(actual_bindings))
    unknown_bindings = sorted(set(actual_bindings) - expected_bindings)
    if missing_bindings:
        problems.append(
            "missing observation bindings: " + ", ".join(missing_bindings)
        )
    if unknown_bindings:
        problems.append(
            "observation bindings reference unknown evidence requirements: "
            + ", ".join(unknown_bindings)
        )
    placeholder_commands = [
        command
        for command in [
            *design.delivery_run,
            *[
                criterion.command
                for criterion in design.acceptance_criteria
                if criterion.command
            ],
        ]
        if "{workspace}" in command or "{prompt}" in command
    ]
    if placeholder_commands:
        problems.append(
            "repository-level delivery and acceptance commands cannot use "
            "Scenario placeholders"
        )
    return problems


def _attach_delivery_execution(
    expansion: ContractExpansion,
    definition: DeliveryCaseDefinition,
    design: DeliveryExecutionDesign,
) -> ContractExpansion:
    """Join frozen case semantics to concrete execution without LLM reinterpretation."""

    runtime = design.scenario_execution
    scenarios: list[DeliveryScenario] = []
    if runtime is not None:
        requirements = {
            item.id: item for item in definition.evidence_requirements
        }
        observations = [
            ScenarioObservation(
                id=requirement.id,
                component=binding.component,
                expected_behavior=requirement.claim,
                evidence_sources=binding.evidence_sources,
            )
            for binding in runtime.observation_bindings
            if (requirement := requirements.get(binding.evidence_requirement_id))
            is not None
        ]
        trajectory_sources = {
            "agent_input",
            "agent_output",
            "tool_call",
            "tool_result",
        }
        requires_trajectory = any(
            item.subject == "target_agent_action"
            for item in definition.evidence_requirements
        ) or any(
            trajectory_sources.intersection(binding.evidence_sources)
            for binding in runtime.observation_bindings
        )
        scenarios.append(
            DeliveryScenario(
                id=definition.case_id,
                task_contract=definition.task_contract,
                command=runtime.command,
                fixture_files=runtime.workspace_seed_files,
                environment_requirements=runtime.environment_requirements,
                baseline_prediction=definition.comparison.baseline_prediction,
                candidate_prediction=definition.comparison.candidate_prediction,
                observable_difference=definition.comparison.observable_difference,
                observations=observations,
                budgets=runtime.budgets,
                executable_conditions=runtime.executable_conditions,
                requires_trajectory=requires_trajectory,
            )
        )

    contract = expansion.selected_change_contract.model_copy(
        update={
            "acceptance_criteria": design.acceptance_criteria,
            "delivery_run": design.delivery_run,
            "delivery_scenarios": scenarios,
            "delivery_checklist": design.delivery_checklist,
            "rollback_plan": design.rollback_plan,
        }
    )
    return expansion.model_copy(update={"selected_change_contract": contract})


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
    *,
    workflow_diagnosis: WorkflowDiagnosisBoard | None = None,
    case_analysis: CaseAnalysisBoard | None = None,
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
                workflow_diagnosis=workflow_diagnosis,
                case_analysis=case_analysis,
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


def _target_run_trajectory_tool(context: OrchestratorContext) -> Tool:
    """Return every ordered action for one target run, never Reforge workflow events."""

    async def get_run(args: dict, _tool_context: ToolContext) -> ToolResult:
        run_id = str(args.get("run_id") or "").strip()
        if not run_id:
            return ToolResult(content="Error: 'run_id' is required.", is_error=True)
        if run_id not in context.current_run_ids:
            return ToolResult(
                content=f"Error: current target-agent run not found: {run_id}",
                is_error=True,
            )
        events = []
        for event_id, record in context.raw_target_evidence.items():
            record_run_id = str(
                record.get("run_id") or record.get("session_id") or "unknown"
            )
            if record_run_id != run_id:
                continue
            item = dict(record)
            item["event_id"] = event_id
            for key in ("content", "final_response", "error"):
                if isinstance(item.get(key), str):
                    item[key] = item[key][:4_000]
            events.append(item)
        return ToolResult(
            content=json.dumps(
                {
                    "run_id": run_id,
                    "trajectory_kind": "target_agent",
                    "events": events,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return Tool(
        name="get_target_run_trajectory",
        description=(
            "Read the complete ordered TARGET AGENT trajectory for one current run, "
            "including agent turns, tool calls/results, evaluation, and terminal output. "
            "This never returns AgentReforge workflow events."
        ),
        parameters=object_schema(
            {
                "run_id": {
                    "type": "string",
                    "description": "Current target-agent run id",
                    "enum": list(context.current_run_ids),
                }
            },
            required=["run_id"],
        ),
        handler=get_run,
        is_read_only=True,
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
