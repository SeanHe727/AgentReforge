# AgentReforge agent interface contracts

This document lists the control-plane payload at every agent boundary. JSON
examples use `?` for optional fields. Raw source, diffs, command output, and final
user-facing prose are opaque evidence fields; they are not parsed as control text.

## Planner -> Worker

Planner input:

```text
PlannerRequest {
  request_kind: string
  goal: string
  completed_tasks: [{id, description, result}]
  failed_tasks: [{id, description, result}]
  retry_instruction: string
}
```

Planner output:

```text
PlanSpec {
  tasks: [{
    id: string
    description: string
    dependencies: [task_id]
  }]
}
```

Worker input:

```text
{
  request_kind: execute_plan_task | execute_reviewed_task
  overall_goal: string
  task: {id, description, dependencies}
  prerequisite_results: [{task_id, description, status, result}]
  review_feedback?: string
  instruction: string
  output_contract: string
}
```

Worker output:

```text
WorkerHandoff {
  status: completed | blocked
  summary: string
  blocking_reasons: [string]
  evidence: [string]
}
```

## Orchestrator internal stages

### Workflow Analysis

Input:

```text
{
  request_kind: workflow_analysis
  previous_reforge_loops: [ReforgeLoopSummary]
  run_manifest: object
}
```

It intentionally receives no target trajectory, target evidence index, or target
repository map.

Output:

```text
WorkflowDiagnosisBoard {
  loop_assessments: [{
    loop_id: string
    stage: string
    classification:
      delivered | target_attempt_failed | workflow_interface_failed |
      workflow_execution_failed | mixed | inconclusive
    observed_facts: [string]
    related_candidate_ids: [string]
    explanation: string
  }]
  workflow_issues: {
    issue_id: ProblemCase
  }
  target_candidate_exclusions: {
    candidate_id: reason
  }
  whole_picture_summary: string
}
```

### Case Analysis

One independent call is made for each current target-agent run. The stage receives
only that run's deterministic summary and evidence coordinates, then uses the
read-only `get_target_run_trajectory(run_id)` tool to inspect every ordered action.

Output:

```text
CaseAnalysis {
  case_id: "case:<run_id>"
  run_id: string
  target_commit: string
  task_intent: {objective, requirements}
  artifact_analysis: {
    summary
    changed_components
    implemented_behavior
    behavior_flow
  }
  final_status: success | partial | failed | inconclusive
  steps: [{step, expected, observed, status, evidence_refs}]
  final_outcome: string
  causal_analysis: {
    failure_layer:
      target_agent | generated_artifact | task_definition | environment |
      agentreforge_workflow | none | uncertain
    direct_cause
    contributing_causes
    alternative_hypotheses
    uncertainty
  }
  capability_signals: [{capability, signal, evidence_refs, generalizability}]
  improvement_opportunities: [{area, rationale, expected_effect}]
}
```

The deterministic coordinator keys the independent outputs by `run_id` in one
`CaseAnalysisBoard`. Case Analysis may identify possible improvement areas, but it
cannot select an intervention or turn a Task Workspace feature into an Agent
Repository requirement.

### Target-Agent Analysis

Input:

```text
{
  request_kind: target_analysis
  improvement_intent: string
  target_run_semantics: string
  current_target_commit: string
  current_run_ids: [run_id]
  required_disposition_run_ids: [run_id]
  case_analysis_board: CaseAnalysisBoard
  improvement_backlog: {candidate_id: BacklogCandidate}
  repository: RepositoryContext
  run_manifest: object
}
```

The supplied backlog has already removed IDs classified as workflow-only. It
intentionally receives no raw `previous_reforge_loops` or mixed target trajectories.
It may inspect a cited raw target event only to resolve a material ambiguity.

Output:

```text
DiagnosisBoard {
  alert_dispositions: [{
    run_id: string
    observed_failure: string
    agent_level_interpretation: string
    likely_causes: [string]
    disposition: candidate_problem | deferred | unsupported | already_resolved
    disposition_reason: string
    evidence_refs: [event_id]
  }]
  problem_cases: {problem_id: ProblemCase}
  retired_problems: {problem_id: reason}
  whole_picture_summary: string
}
```

`ProblemCase` is:

```text
{
  id: string
  symptom: string
  capability_gap: string
  likely_root_causes: [string]
  affected_scope: [string]
  evidence_refs: [event_id]
  uncertainty: string
}
```

### Candidate Selection

Input:

```text
{
  request_kind: candidate_selection
  improvement_intent: string
  diagnosis_board: DiagnosisBoard
  improvement_backlog: {candidate_id: BacklogCandidate}
  repository: RepositoryContext
  run_manifest: object
}
```

Output:

```text
SelectionDecision {
  summary: string
  problem_statement: string
  candidate_backlog: {candidate_id: BacklogCandidate}
  preliminary_ranking: [{candidate_id, rank, rationale}]
  top_two_comparison?: CandidatePairwiseComparison
  selected_candidate_id: string
  deferred_candidates: {candidate_id: reason}
  benefit: 1..5
  risk: 1..5
  effort: 1..5
  confidence: 0..1
  decision: proceed | abstain | needs_human
  decision_reason: string
}
```

Each `BacklogCandidate` contains typed diagnosis, intervention, problem/causal/
impact/evaluability scorecards, scope, dependencies, conflicts, and history.

### Contract Expansion

Input:

```text
{
  request_kind: contract_expansion
  selection_decision: SelectionDecision
  selected_problem_cases: {problem_id: ProblemCase}
  repository: RepositoryContext
  run_manifest: object
  relevant_evidence_index: [evidence_coordinate]
}
```

Output:

```text
ContractExpansion {
  proposal_guardrails: [ContractClause]
  selected_change_contract: SelectedChangeContract
  evidence: [Evidence]
  goals: [string]
  non_goals: [string]
  affected_components: [string]
  dependencies: [string]
  alternatives_considered: [string]
}
```

Contract Expansion intentionally leaves acceptance criteria, runtime commands,
Delivery Scenarios, checklist, and rollback fields empty. This keeps the semantic
change contract independent from the larger executable fixture.

### Scenario Definition

Input:

```text
{
  request_kind: scenario_definition
  selected_change_contract: filtered semantic contract
  proposal_context: {guardrails, goals, non_goals}
  selected_problem_cases: {problem_id: ProblemCase}
  relevant_evidence_index: [evidence_coordinate]
}
```

Output:

```text
DeliveryCaseDefinition {
  contract_id: string
  case_id: string
  capability_under_test: string
  task_contract: ScenarioTaskContract
  workspace_requirements: [ScenarioWorkspaceRequirement]
  comparison: ScenarioComparison
  evidence_requirements: [ScenarioEvidenceRequirement with related_condition_ids]
}
```

This stage defines an ordinary coding case and its causal evidence requirements.
It does not receive the repository map or emit commands, paths, files, or budgets.

### Scenario Materialization

Input:

```text
{
  request_kind: scenario_materialization
  frozen_case_definition: DeliveryCaseDefinition
  target_agent_repository: RepositoryContext
  run_manifest: object
  declared_safety_properties: [string]
}
```

Output:

```text
DeliveryExecutionDesign {
  contract_id: string
  case_id: string
  acceptance_criteria: [AcceptanceCriterion]
  delivery_run: [string]
  scenario_execution: ScenarioExecutionDesign
  delivery_checklist: [string]
  rollback_plan: string
}
```

The Scenario command launches the real Agent Repository against an ordinary
disposable Task Workspace. The fixture never contains a copied target-agent package,
and the task never asks the target agent to modify itself. Delivery commands and
code-level acceptance commands run in the Agent Repository and cannot contain Scenario
placeholders. The Scenario command itself must contain both `{prompt}` and
`{workspace}`, so the frozen structured task and disposable workspace cannot be
silently replaced by literals. Each frozen evidence requirement is bound exactly once
to concrete event sources.

The coordinator joins Workflow Analysis, Case Analysis, Target Analysis, Selection,
Contract Expansion, Scenario Definition, and Scenario Materialization into
`ImprovementProposal`. It verifies IDs, outcome/evidence references, binding coverage,
and safety-property references. It then constructs the existing `DeliveryScenario`
deterministically and cannot replace Selection or add semantic content.

## Proposal lookup

Writer and Reviewer receive the same compact `WriterTaskContract`, deterministically
compiled from the full auditable Change Contract. Other whole-picture context is
available through:

```text
read_proposal {
  section:
    goals | non_goals | guardrails | evidence | dependencies | alternatives
}
```

This tool is read-only and scoped to the current frozen Proposal. It is background
and consistency context; it cannot redefine or enlarge the selected Change Contract.
The Deliverer receives the relevant Proposal boundary only in its final Goal
Realization phase and reports concrete contradictions in `proposal_violations`.

## Orchestrator -> Writer

Before the first Review Cycle, Writer Planning receives the compact contract through a
read-only tool surface and emits one 2-5 step `WriterPlan`. It does not run again on
Reviewer retries.

Input:

```text
{
  request_kind: writer_task | writer_repair
  writer_task_contract: WriterTaskContract
  writer_plan: WriterPlan
  completed_prerequisites: [string]
  review_feedback?: ReviewResult
  retry_policy: string
  output_contract: {
    status: completed | blocked
    summary: string
    verification: [VerificationRecord]
    finding_resolutions: [FindingResolution]
    blocking_reasons: [string]
  }
}
```

`WriterTaskContract` carries only the objective, capability delta, one selected
implementation direction, the most important requirements/basic
constraints/non-goals, suggested components, safety properties, and up to three
compact acceptance intents (without generated commands). Diagnosis, evidence, alternatives,
Reviewer focus, and repeated output/invariant prose remain in the Proposal lookup.

Writer output:

```text
WriterHandoff {
  status: completed | blocked
  summary: string
  verification: [VerificationRecord]
  finding_resolutions: [{
    finding_id: string
    change_summary: string
    verification: string
  }]
  blocking_reasons: [string]
}
```

The task-scoped Git diff is the authoritative implementation artifact. If the
Writer consumes its last work turn with a tool call, the executor performs one
additional tool-free output turn. That handoff turn is not charged to the work
budget and receives the frozen Task plus authoritative diff.

## Writer -> Reviewer

Input:

```text
{
  request_kind: task_review
  task_contract: {
    writer_task_contract: WriterTaskContract
  }
  writer_artifact: {
    handoff_kind: writer_to_reviewer
    writer_handoff: WriterHandoff
    authoritative_task_diff: string
    diff_is_implementation_evidence: true
    writer_handoff_is_evidence: false
  }
}
```

Output:

```text
ReviewerOutput {
  verdict: approve | needs_fix
  blocking_findings: [Finding]
  non_blocking_findings: [Finding]
  summary: string
}
```

One Review Cycle is exactly one Writer Attempt followed by one Reviewer Pass.
Writer and Reviewer do not have independent round counts; they have separate per-pass
turn budgets. Reviewer findings receive stable IDs. On a retry, Writer reports one
resolution per attempted finding, while Reviewer independently decides whether the
latest cumulative diff actually resolves it.

`Finding` contains severity, location, description, evidence, required fix, and
related acceptance criterion. The executor converts this to:

```text
ReviewResult {
  verdict: accept | revise | escalate
  findings: [Finding]
  summary: string
}
```

## Orchestrator/Runner -> Deliverer

### Scenario Readiness

Input:

```text
{
  selected_capability: CandidateIntervention
  scenario: DeliveryScenario
  fixture_inventory: [path]
  execution_semantics: {
    command_cwd
    candidate_repository_available
    fixture_role
    workspace_placeholder
  }
}
```

Output:

```text
ScenarioReadinessOutput {
  ready: boolean
  missing_requirements: [string]
  execution_focus: [string]
  summary: string
}
```

### Delivery Scenario

The target task is represented by:

```text
ScenarioTaskContract {
  objective: string
  context: [string]
  requirements: [ContractClause]
  constraints: [ContractClause]
  outcome_contract: {
    primary_success: [ScenarioOutcomeCondition]
    guardrails: [ScenarioOutcomeCondition]
    inconclusive: [ScenarioOutcomeCondition]
  }
  acceptance_checks: [ContractClause]
}
```

Every `ScenarioOutcomeCondition` contains `id`, `description`, `rationale`, and
`evidence_direction`. A `DeliveryScenario` additionally contains argv, fixture
files, environment requirements, baseline/candidate predictions, observable
difference, observations, budgets, executable conditions, and trajectory policy.

The Runner produces a structured `ScenarioRunResult`: command, prompt, exit code,
output, changed files, artifacts, trajectory availability/events, environment
facts, and spawn/timeout state.

### Scenario Evidence Analysis

Input contains the frozen Scenario, readiness result, and one structured execution
bundle.

Output:

```text
ScenarioEvidenceCard {
  scenario_id: string
  observed_facts: [string]
  trajectory_findings: [string]
  artifact_findings: [string]
  baseline_consistent: boolean
  candidate_consistent: boolean
  discriminating_evidence: [string]
  outcome_assessments: [{
    condition_id: string
    category: primary_success | guardrail | inconclusive
    status: supported | violated | not_observed | not_applicable
    evidence: [string]
    explanation: string
  }]
  confounders: [string]
  sufficient: boolean
  summary: string
}
```

### Goal Realization

Input contains the frozen Proposal/Contract, Loop diff, Runner evidence, and
`ScenarioEvidenceCard`.

Output:

```text
DeliveryReviewOutput {
  ready: boolean
  failure_kind:
    none | implementation_defect | verification_gap |
    plan_gap | environment_failure
  missing_objectives: [string]
  integration_concerns: [string]
  proposal_violations: [string]
  blocking_evidence: [string]
  summary: string
}
```

## Handoff repair

Input:

```text
{
  request_kind: repair_handoff
  producer: string
  output_contract: string
  validation_error: string
  immutable_context: object | string
  previous_invalid_output: string
}
```

Output is another instance of the original producer schema. It cannot edit product
code or change downstream requirements.

## Target-agent adapter boundary

AgentReforge does not require arbitrary external target agents to implement its
internal Pydantic models. The adapter sends the rendered `ScenarioTaskContract` and
normalizes observable actions into JSONL:

```text
agent_turn {sequence, actor, status, turn, input_messages, content, tool_calls}
tool_result {sequence, actor, status, name, arguments, content, is_error, action_step}
done {sequence, actor, status, outcome, final_response, reason_code, budget}
```

`final_response` is an opaque string and never controls acceptance by keyword.
