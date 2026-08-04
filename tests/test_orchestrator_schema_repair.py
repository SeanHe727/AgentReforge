from __future__ import annotations

import asyncio

import pytest
from conftest import make_proposal

from agentreforge.improve.context import OrchestratorContextBuilder
from agentreforge.improve.models import (
    AlertDisposition,
    BacklogCandidate,
    CandidateDiagnosis,
    CandidateIntervention,
    CandidatePairwiseComparison,
    CandidateRankingEntry,
    CandidateScope,
    CaseAnalysis,
    ContractExpansion,
    DeliveryCaseDefinition,
    DeliveryExecutionDesign,
    DiagnosisBoard,
    SelectedChangeContract,
    SelectionDecision,
    WorkflowDiagnosisBoard,
)
from agentreforge.improve.orchestrator import (
    CASE_ANALYSIS_PROMPT,
    CONTRACT_PROMPT,
    ORCHESTRATOR_PROMPT,
    SCENARIO_DEFINITION_PROMPT,
    SCENARIO_MATERIALIZATION_PROMPT,
    SELECTION_PROMPT,
    TRIAGE_PROMPT,
    WORKFLOW_TRIAGE_PROMPT,
    Orchestrator,
    _attach_frozen_decision,
    _validate_delivery_execution,
    _validate_selection,
    _validate_triage_coverage,
)
from agentreforge.tools.registry import ToolRegistry


class _UnusedClient:
    pass


def test_orchestrator_distinguishes_backlog_dependencies_and_execution_contract():
    assert "dictionary keyed by a stable" in ORCHESTRATOR_PROMPT
    assert "selected_change_contract" in ORCHESTRATOR_PROMPT
    assert "Contract is one CHANGE UNIT" in ORCHESTRATOR_PROMPT


def test_orchestrator_preserves_agent_repository_and_task_artifact_ownership():
    assert "describes the GENERATED TASK ARTIFACT by default" in ORCHESTRATOR_PROMPT
    assert "never propose adding an evaluation" in SELECTION_PROMPT
    assert '"backlog_item_id": "the exact frozen selected_candidate_id"' in CONTRACT_PROMPT
    assert "Do not emit commands" in SCENARIO_DEFINITION_PROMPT
    assert "ordinary product-code task" in SCENARIO_DEFINITION_PROMPT
    assert "NAME=value" in SCENARIO_MATERIALIZATION_PROMPT
    assert "literal executable source" in SCENARIO_MATERIALIZATION_PROMPT
    assert "Every `delivery_run` item" in SCENARIO_MATERIALIZATION_PROMPT
    assert "never\nuse `trajectory` or `code`" in SCENARIO_MATERIALIZATION_PROMPT
    assert "Never copy the target-agent package" in SCENARIO_MATERIALIZATION_PROMPT
    assert "cannot contain\n`{workspace}`" in SCENARIO_MATERIALIZATION_PROMPT
    assert "positive capability path" in ORCHESTRATOR_PROMPT
    assert "broad keyword matching" in CONTRACT_PROMPT


def test_orchestrator_treats_safety_checks_as_explicit_not_heuristic():
    assert "verified_safety_properties" in ORCHESTRATOR_PROMPT
    assert "Safety probes are system-owned" in ORCHESTRATOR_PROMPT
    assert "execution tool's target adapter executes the actual traversal" in ORCHESTRATOR_PROMPT
    assert "Never encode a condition that rewards the" in ORCHESTRATOR_PROMPT
    assert "safety violation" in ORCHESTRATOR_PROMPT
    assert "literal `..` escape" not in ORCHESTRATOR_PROMPT


def test_orchestrator_has_single_task_and_achievement_rules():
    assert "Build an Achievement Ledger" in ORCHESTRATOR_PROMPT
    assert "exactly ONE backlog Candidate" in ORCHESTRATOR_PROMPT
    assert "exactly ONE" in ORCHESTRATOR_PROMPT
    assert "`selected_change_contract`" in ORCHESTRATOR_PROMPT
    assert "The recursion limit is a ceiling, never a target" in ORCHESTRATOR_PROMPT


def test_orchestrator_uses_advisory_scorecards_and_top_two_review():
    assert "PROBLEM TRIAGE before considering solution cost" in ORCHESTRATOR_PROMPT
    assert "failed_verification" in ORCHESTRATOR_PROMPT
    assert "NEVER add them into a mechanical total" in ORCHESTRATOR_PROMPT
    assert "`preliminary_ranking` for every viable Candidate" in ORCHESTRATOR_PROMPT
    assert "`top_two_comparison`" in ORCHESTRATOR_PROMPT
    assert "literal option `DEFER`" in ORCHESTRATOR_PROMPT
    assert "would the baseline likely pass" in ORCHESTRATOR_PROMPT
    assert "does not compute or override your ranking" in ORCHESTRATOR_PROMPT
    assert "TARGET AGENT REPOSITORY from a TASK WORKSPACE" in ORCHESTRATOR_PROMPT
    assert "`target_commit` identifies the agent version" in ORCHESTRATOR_PROMPT
    assert "terminal `failed_verification`" in ORCHESTRATOR_PROMPT
    assert "both likely pass" in ORCHESTRATOR_PROMPT
    assert "stochastic run could easily reverse" in ORCHESTRATOR_PROMPT


def test_proposal_schema_repair_can_recover_on_second_retry():
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), ".")
    valid = make_proposal().model_dump_json()
    repairs = iter(["{}", valid])
    seen: list[tuple[str, str]] = []

    async def repair(bad_output: str, error: str) -> str:
        seen.append((bad_output, error))
        return next(repairs)

    orchestrator._repair = repair  # type: ignore[method-assign]

    proposal = asyncio.run(orchestrator._parse_proposal_with_repair("not json"))

    assert proposal.summary == "Improve a component"
    assert len(seen) == 2
    assert seen[0][0] == "not json"
    assert seen[1][0] == "{}"


def test_proposal_schema_repair_stops_at_budget():
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), ".")
    repair_calls = 0

    async def repair(_bad_output: str, _error: str) -> str:
        nonlocal repair_calls
        repair_calls += 1
        return "{}"

    orchestrator._repair = repair  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="invalid JSON"):
        asyncio.run(
            orchestrator._parse_proposal_with_repair("not json", max_repairs=2)
        )

    assert repair_calls == 2


def _selection(*, decision: str = "proceed") -> SelectionDecision:
    diagnosis = CandidateDiagnosis(
        symptom="the target exhausts its step budget",
        root_cause="the workflow does not preserve a verification budget",
        capability_gap="complete cross-component tasks before stopping",
        evidence_refs=["run:event:2"],
    )
    intervention = CandidateIntervention(
        level="workflow",
        mechanism="reserve and track implementation plus verification phases",
        expected_capability_delta="finish the requested behavior and verify it",
    )
    candidate = BacklogCandidate(
        id="completion-budget",
        title="Cross-component completion budget",
        diagnosis=diagnosis,
        intervention=intervention,
    )
    selected = "" if decision == "abstain" else candidate.id
    return SelectionDecision(
        summary="Improve complete task delivery",
        problem_statement="The target stops with required behavior incomplete.",
        candidate_backlog={candidate.id: candidate},
        preliminary_ranking=[
            CandidateRankingEntry(
                candidate_id=candidate.id,
                rank=1,
                rationale="Current terminal evidence has the largest outcome impact.",
            )
        ],
        top_two_comparison=(
            None
            if decision == "abstain"
            else CandidatePairwiseComparison(
                candidate_a=candidate.id,
                candidate_b="DEFER",
                strongest_case_for_a="It addresses a terminal current failure.",
                strongest_case_for_b="The direct cause remains uncertain.",
                winner=candidate.id,
                decision_reason="The observable failure justifies one bounded loop.",
            )
        ),
        selected_candidate_id=selected,
        benefit=5,
        risk=2,
        effort=3,
        confidence=0.8,
        decision=decision,
        decision_reason="Proceed on the strongest current capability gap.",
    )


def _contract_expansion(selection: SelectionDecision) -> ContractExpansion:
    candidate = selection.candidate_backlog[selection.selected_candidate_id]
    contract = SelectedChangeContract(
        contract_id="contract-completion-budget",
        backlog_item_id=candidate.id,
        objective="Make bounded tasks finish with verification",
        diagnosis=candidate.diagnosis,
        intervention=candidate.intervention,
    )
    return ContractExpansion(
        selected_change_contract=contract,
    )


def _case_definition(expansion: ContractExpansion) -> DeliveryCaseDefinition:
    return DeliveryCaseDefinition.model_validate(
        {
            "contract_id": expansion.selected_change_contract.contract_id,
            "case_id": "case-completion-budget",
            "capability_under_test": "complete and verify a bounded coding task",
            "task_contract": {
                "objective": "Complete the incomplete implementation and verify it.",
                "outcome_contract": {
                    "primary_success": [
                        {
                            "id": "outcome-complete",
                            "description": "The requested behavior works.",
                            "rationale": "This is the task's required result.",
                            "evidence_direction": "Tests and artifacts support completion.",
                        }
                    ]
                },
            },
            "workspace_requirements": [
                {
                    "id": "workspace-code",
                    "role": "existing implementation",
                    "required_initial_state": "One behavior is incomplete.",
                    "purpose": "Provide a representative bounded coding task.",
                }
            ],
            "comparison": {
                "baseline_prediction": "Stops before verification.",
                "candidate_prediction": "Implements and verifies the behavior.",
                "observable_difference": "Candidate produces a passing verified result.",
            },
            "evidence_requirements": [
                {
                    "id": "evidence-complete",
                    "claim": "The generated behavior passes its focused test.",
                    "subject": "task_artifact",
                    "related_condition_ids": ["outcome-complete"],
                }
            ],
        }
    )


def _delivery_execution(
    expansion: ContractExpansion,
    case: DeliveryCaseDefinition,
) -> DeliveryExecutionDesign:
    return DeliveryExecutionDesign.model_validate(
        {
            "contract_id": expansion.selected_change_contract.contract_id,
            "case_id": case.case_id,
            "scenario_execution": {
                "command": ["python", "-m", "demo_agent", "{prompt}", "{workspace}"],
                "workspace_seed_files": {
                    "src/example.py": "def solve():\n    raise NotImplementedError\n"
                },
                "observation_bindings": [
                    {
                        "evidence_requirement_id": "evidence-complete",
                        "component": "generated task artifact",
                        "evidence_sources": ["artifact", "tool_result"],
                    }
                ],
            },
        }
    )


def test_delivery_execution_must_pass_frozen_prompt_and_workspace():
    selection = _selection()
    expansion = _contract_expansion(selection)
    case = _case_definition(expansion)
    execution = _delivery_execution(expansion, case)
    assert _validate_delivery_execution(execution, expansion, case) == []

    assert execution.scenario_execution is not None
    invalid_runtime = execution.scenario_execution.model_copy(
        update={"command": ["python", "-m", "demo_agent", "short literal prompt"]}
    )
    invalid = execution.model_copy(update={"scenario_execution": invalid_runtime})

    problems = _validate_delivery_execution(invalid, expansion, case)

    assert any("{prompt}" in problem for problem in problems)
    assert any("{workspace}" in problem for problem in problems)


def test_orchestrator_runs_typed_stages_without_target_cases(tmp_path):
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), str(tmp_path))
    workflow = WorkflowDiagnosisBoard(
        whole_picture_summary="No prior workflow failures."
    )
    diagnosis = DiagnosisBoard(whole_picture_summary="One current terminal failure.")
    selection = _selection()
    contract = _contract_expansion(selection)
    case_definition = _case_definition(contract)
    delivery = _delivery_execution(contract, case_definition)
    outputs = iter(
        [
            workflow.model_dump_json(),
            diagnosis.model_dump_json(),
            selection.model_dump_json(),
            contract.model_dump_json(),
            case_definition.model_dump_json(),
            delivery.model_dump_json(),
        ]
    )
    calls: list[tuple[str, str]] = []

    async def investigate_stage(*, system_prompt, user_message, max_turns, registry=None):
        calls.append((system_prompt, user_message))
        return next(outputs)

    async def complete_typed_stage(*, system_prompt, user_message):
        calls.append((system_prompt, user_message))
        return next(outputs)

    orchestrator._investigate_stage = investigate_stage  # type: ignore[method-assign]
    orchestrator._complete_typed_stage = complete_typed_stage  # type: ignore[method-assign]
    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve the target",
        target_trajectory=[],
        previous_reforge_loops=[],
        run_manifest={"loop_base": "abcdef123456"},
    )

    proposal = asyncio.run(
        orchestrator.analyze(
            "improve the target",
            [],
            context=context,
        )
    )

    assert [item[0] for item in calls] == [
        WORKFLOW_TRIAGE_PROMPT,
        TRIAGE_PROMPT,
        SELECTION_PROMPT,
        CONTRACT_PROMPT,
        SCENARIO_DEFINITION_PROMPT,
        SCENARIO_MATERIALIZATION_PROMPT,
    ]
    assert "target_agent_runs" not in calls[0][1]
    assert "previous_reforge_loops" not in calls[1][1]
    assert "case_analysis_board" in calls[1][1]
    assert "target_agent_runs" not in calls[1][1]
    assert "target_agent_runs" not in calls[2][1]
    assert "selection_decision" in calls[3][1]
    assert "selected_change_contract" in calls[4][1]
    assert "repository" not in calls[4][1]
    assert "frozen_case_definition" in calls[5][1]
    assert "target_agent_repository" in calls[5][1]
    assert proposal.selected_change_contract is not None
    assert proposal.selected_change_contract.delivery_scenarios[0].requires_trajectory
    assert proposal.selected_candidate_id == "completion-budget"
    assert proposal.orchestrator_artifacts is not None
    assert proposal.orchestrator_artifacts.case_analysis is not None
    assert proposal.orchestrator_artifacts.case_analysis.cases == {}
    assert (
        proposal.orchestrator_artifacts.diagnosis.whole_picture_summary
        == "One current terminal failure."
    )


def test_orchestrator_analyzes_each_current_target_run_before_target_analysis(
    tmp_path,
):
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), str(tmp_path))
    workflow = WorkflowDiagnosisBoard()
    case = CaseAnalysis.model_validate(
        {
            "case_id": "case:run-1",
            "run_id": "run-1",
            "target_commit": "abcdef1234567890",
            "task_intent": {
                "objective": "Implement and verify the requested change.",
                "requirements": ["Complete the target behavior."],
            },
            "artifact_analysis": {
                "summary": "The generated artifact remained incomplete.",
                "changed_components": ["src/example.py"],
                "implemented_behavior": ["Added a partial implementation."],
                "behavior_flow": ["Read -> edit -> stop"],
            },
            "final_status": "failed",
            "steps": [
                {
                    "step": "implementation",
                    "expected": "Complete and verify the change.",
                    "observed": "Stopped before verification.",
                    "status": "failed",
                    "evidence_refs": ["run-1:event:1"],
                }
            ],
            "final_outcome": "The requested behavior was not verified.",
            "causal_analysis": {
                "failure_layer": "target_agent",
                "direct_cause": "The agent stopped before verification.",
            },
            "capability_signals": [
                {
                    "capability": "completion and verification",
                    "signal": "negative",
                    "evidence_refs": ["run-1:event:1"],
                    "generalizability": "May affect other bounded coding tasks.",
                }
            ],
            "improvement_opportunities": [],
        }
    )
    diagnosis = DiagnosisBoard(
        alert_dispositions=[
            AlertDisposition(
                run_id="run-1",
                observed_failure="The task stopped before verification.",
                agent_level_interpretation=(
                    "The target did not preserve a completion/verification phase."
                ),
                likely_causes=["insufficient phase budgeting"],
                disposition="candidate_problem",
                disposition_reason="Current failed case supports a reusable gap.",
                evidence_refs=["run-1:event:1"],
            )
        ],
        whole_picture_summary="One failed current case.",
    )
    selection = _selection()
    contract = _contract_expansion(selection)
    case_definition = _case_definition(contract)
    delivery = _delivery_execution(contract, case_definition)
    outputs = iter(
        [
            workflow.model_dump_json(),
            case.model_dump_json(),
            diagnosis.model_dump_json(),
            selection.model_dump_json(),
            contract.model_dump_json(),
            case_definition.model_dump_json(),
            delivery.model_dump_json(),
        ]
    )
    calls: list[tuple[str, str]] = []

    async def investigate_stage(*, system_prompt, user_message, max_turns, registry=None):
        calls.append((system_prompt, user_message))
        return next(outputs)

    async def complete_typed_stage(*, system_prompt, user_message):
        calls.append((system_prompt, user_message))
        return next(outputs)

    orchestrator._investigate_stage = investigate_stage  # type: ignore[method-assign]
    orchestrator._complete_typed_stage = complete_typed_stage  # type: ignore[method-assign]
    trajectory = [
        {
            "run_id": "run-1",
            "type": "target_run_started",
            "task_prompt": "Implement and verify the requested change.",
            "target_commit": "abcdef1234567890",
        },
        {
            "run_id": "run-1",
            "type": "done",
            "outcome": "incomplete",
            "final_response": "Stopped before verification.",
            "target_commit": "abcdef1234567890",
        },
    ]
    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve the target",
        target_trajectory=trajectory,
        previous_reforge_loops=[],
        run_manifest={"loop_base": "abcdef123456"},
    )

    proposal = asyncio.run(
        orchestrator.analyze("improve the target", trajectory, context=context)
    )

    assert [item[0] for item in calls] == [
        WORKFLOW_TRIAGE_PROMPT,
        CASE_ANALYSIS_PROMPT,
        TRIAGE_PROMPT,
        SELECTION_PROMPT,
        CONTRACT_PROMPT,
        SCENARIO_DEFINITION_PROMPT,
        SCENARIO_MATERIALIZATION_PROMPT,
    ]
    assert '"run_id": "run-1"' in calls[1][1]
    assert "case_analysis_board" in calls[2][1]
    assert proposal.orchestrator_artifacts is not None
    assert proposal.orchestrator_artifacts.case_analysis is not None
    assert proposal.orchestrator_artifacts.case_analysis.cases["run-1"] == case


def test_orchestrator_abstention_skips_contract_expansion(tmp_path):
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), str(tmp_path))
    workflow = WorkflowDiagnosisBoard()
    diagnosis = DiagnosisBoard(whole_picture_summary="No actionable current gap.")
    selection = _selection(decision="abstain")
    outputs = iter(
        [
            workflow.model_dump_json(),
            diagnosis.model_dump_json(),
            selection.model_dump_json(),
        ]
    )
    calls: list[str] = []

    async def investigate_stage(*, system_prompt, user_message, max_turns, registry=None):
        calls.append(system_prompt)
        return next(outputs)

    orchestrator._investigate_stage = investigate_stage  # type: ignore[method-assign]
    proposal = asyncio.run(
        orchestrator.analyze(
            "improve the target",
            [],
            context=OrchestratorContextBuilder(str(tmp_path)).build(
                intent="improve the target",
                target_trajectory=[],
                previous_reforge_loops=[],
                run_manifest={"loop_base": "abcdef123456"},
            ),
        )
    )

    assert calls == [WORKFLOW_TRIAGE_PROMPT, TRIAGE_PROMPT, SELECTION_PROMPT]
    assert proposal.decision == "abstain"
    assert proposal.selected_change_contract is None
    assert proposal.orchestrator_artifacts is not None


def test_triage_handoff_repairs_only_missing_alert_disposition(tmp_path):
    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve the target",
        target_trajectory=[
            {
                "run_id": "current-failure",
                "type": "target_run_started",
                "task_prompt": "complete the integration",
                "target_commit": "abcdef1234567890",
            },
            {
                "run_id": "current-failure",
                "type": "done",
                "outcome": "incomplete",
                "final_response": "(stopped: reached max steps)",
            },
        ],
        previous_reforge_loops=[],
        run_manifest={"loop_base": "abcdef123456"},
    )
    repaired = DiagnosisBoard(
        alert_dispositions=[
            AlertDisposition(
                run_id="current-failure",
                observed_failure="The task stopped before completion.",
                agent_level_interpretation="The bounded workflow did not finish.",
                likely_causes=["step allocation"],
                disposition="candidate_problem",
                disposition_reason="Current terminal evidence requires diagnosis.",
                evidence_refs=["current-failure:event:1"],
            )
        ]
    )
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), str(tmp_path))
    seen: list[tuple[str, str]] = []

    async def repair_stage(
        bad_output,
        error,
        *,
        system_prompt,
        stage_name,
        schema,
        immutable_context="",
    ):
        assert schema["title"] == "DiagnosisBoard"
        assert immutable_context == '{"current_run_alerts":["current-failure"]}'
        seen.append((stage_name, error))
        return repaired.model_dump_json()

    orchestrator._repair_stage = repair_stage  # type: ignore[method-assign]
    result = asyncio.run(
        orchestrator._parse_stage_with_repair(
            DiagnosisBoard().model_dump_json(),
            DiagnosisBoard,
            system_prompt=TRIAGE_PROMPT,
            stage_name="DiagnosisBoard",
            validate=lambda board: _validate_triage_coverage(board, context),
            repair_context='{"current_run_alerts":["current-failure"]}',
        )
    )

    assert result.alert_dispositions[0].run_id == "current-failure"
    assert seen == [
        (
            "DiagnosisBoard",
            "missing current_run_alert dispositions: current-failure",
        )
    ]


def test_triage_may_audit_successful_current_runs_beyond_required_alerts(tmp_path):
    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve",
        target_trajectory=[
            {
                "run_id": "current-success",
                "type": "target_run_started",
                "target_commit": "abcdef1234567890",
            },
            {
                "run_id": "current-success",
                "type": "done",
                "outcome": "completed",
                "final_response": "done",
            },
        ],
        previous_reforge_loops=[],
        run_manifest={"loop_base": "abcdef123456"},
    )
    board = DiagnosisBoard(
        alert_dispositions=[
            AlertDisposition(
                run_id="current-success",
                observed_failure="No failure observed.",
                agent_level_interpretation="The sampled capability succeeded.",
                disposition="already_resolved",
                disposition_reason="Current execution and evaluation passed.",
            )
        ]
    )

    assert _validate_triage_coverage(board, context) == []


def test_selection_leaves_component_semantics_to_llm_review(tmp_path):
    package = tmp_path / "demo_agent"
    package.mkdir()
    (package / "agent.py").write_text("", encoding="utf-8")
    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve",
        target_trajectory=[],
        previous_reforge_loops=[],
        run_manifest={"loop_base": "abcdef123456"},
    )
    selection = _selection()
    candidate = selection.candidate_backlog["completion-budget"].model_copy(
        update={
            "scope": CandidateScope(
                affected_components=["jobqueue/service.py", "demo_agent/agent.py"]
            )
        }
    )
    selection = selection.model_copy(
        update={"candidate_backlog": {candidate.id: candidate}}
    )

    problems = _validate_selection(selection, context)

    assert problems == []
    assert all("demo_agent/agent.py" not in problem for problem in problems)


def test_contract_revision_cannot_change_frozen_selection_or_drop_artifacts():
    diagnosis = DiagnosisBoard(whole_picture_summary="Current failure is actionable.")
    selection = _selection()
    proposal = _attach_frozen_decision(
        # Coordinator assembly is exercised by the three-stage test above; this
        # simulates a later acceptance repair returning drifted decision fields.
        make_proposal(),
        diagnosis,
        selection,
    )
    drifted = proposal.model_copy(
        update={
            "summary": "different problem",
            "selected_candidate_id": "easy-prompt-tweak",
            "candidate_backlog": {},
            "orchestrator_artifacts": None,
        }
    )

    repaired = _attach_frozen_decision(drifted, diagnosis, selection)

    assert repaired.summary == selection.summary
    assert repaired.selected_candidate_id == selection.selected_candidate_id
    assert repaired.candidate_backlog == selection.candidate_backlog
    assert repaired.orchestrator_artifacts is not None
    assert (
        repaired.orchestrator_artifacts.diagnosis.whole_picture_summary
        == "Current failure is actionable."
    )
