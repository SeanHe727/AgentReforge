from __future__ import annotations

from agentreforge.improve.acceptance import validate_acceptance
from agentreforge.improve.acceptance_runner import AcceptanceRun, RunResult
from agentreforge.improve.deliverer import goal_review_message
from agentreforge.improve.models import (
    AcceptanceCriterion,
    BacklogCandidate,
    CandidateDiagnosis,
    CandidateIntervention,
    CandidatePairwiseComparison,
    CandidatePriority,
    CandidateRankingEntry,
    ContractClause,
    ImprovementProposal,
    ScenarioOutcomeCondition,
    SelectedChangeContract,
)
from agentreforge.improve.pipeline import PipelineResult, _analysis_problems, render_report


def _new_interface_proposal() -> ImprovementProposal:
    diagnosis = CandidateDiagnosis(
        symptom="the agent guesses nested file locations",
        root_cause="the active tool surface cannot search recursively",
        capability_gap="reliable confined repository discovery",
        evidence_refs=["baseline:event:4"],
    )
    intervention = CandidateIntervention(
        level="tool",
        mechanism="add and wire a workspace-confined recursive search tool",
        expected_capability_delta="locate relevant nested source before editing",
    )
    contract = SelectedChangeContract(
        contract_id="change-repo-discovery",
        backlog_item_id="repo-discovery",
        objective="Make confined recursive repository discovery reachable",
        rationale="The observed failure is caused by the missing active tool.",
        diagnosis=diagnosis,
        intervention=intervention,
        expected_outputs=[
            ScenarioOutcomeCondition(
                id="OUT1",
                description="The target can report a relevant nested source path.",
                rationale="This demonstrates that recursive discovery is reachable.",
                evidence_direction=(
                    "A target-agent trajectory should contain the discovered nested path "
                    "before the edit decision."
                ),
            )
        ],
        invariants=[
            ContractClause(
                id="INV1",
                description="File operations remain confined to the workspace.",
            )
        ],
        allowed_write_paths=["demo_agent/"],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC1",
                description="The package starts after the change.",
                mode="non_regression",
                check_type="smoke",
                command="python3 -m demo_agent --help",
            )
        ],
        delivery_run=["python3 -m demo_agent --help"],
        delivery_checklist=["The search mechanism is wired into the active agent path."],
    )
    return ImprovementProposal(
        summary="Improve repository discovery without weakening confinement",
        problem_statement="The target cannot reliably locate nested source files.",
        proposal_guardrails=[
            ContractClause(
                id="PG1",
                description="Do not weaken workspace path confinement.",
            )
        ],
        candidate_backlog={
            "repo-discovery": BacklogCandidate(
                id="repo-discovery",
                title="Confined recursive repository discovery",
                diagnosis=diagnosis,
                intervention=intervention,
                priority=CandidatePriority(
                    problem={
                        "evidence_strength": 5,
                        "failure_severity": 4,
                        "recurrence": 3,
                        "cross_task_impact": 4,
                        "evidence_freshness": 5,
                        "user_relevance": 5,
                        "assessment": "A current trajectory shows a terminal discovery failure.",
                    },
                    causal={
                        "root_cause_confidence": 5,
                        "intervention_fit": 5,
                        "competing_hypotheses": ["planning guidance"],
                        "falsification_condition": "Existing tools already find nested source.",
                    },
                    impact={
                        "expected_outcome_impact": 4,
                        "generality": 4,
                        "one_loop_feasibility": 5,
                        "regression_risk": 2,
                        "effort": 2,
                        "expected_delta": "Nested source is found before editing.",
                    },
                    evaluability={
                        "mechanism_observability": 5,
                        "outcome_observability": 5,
                        "discriminability": 5,
                        "attribution_confidence": 4,
                        "noise_robustness": 4,
                        "evaluation_cost": 2,
                        "baseline_prediction": "The nested file is not found.",
                        "candidate_prediction": "The nested file is found and edited.",
                        "observable_difference": "A task artifact appears in the nested file.",
                        "confounders": ["The model may guess the path."],
                    },
                ),
            ),
            "unselected-planning": BacklogCandidate(
                id="unselected-planning",
                title="Planning guidance",
                diagnosis=CandidateDiagnosis(
                    capability_gap="inspect before edit",
                ),
                intervention=CandidateIntervention(
                    level="prompt",
                    mechanism="add planning guidance",
                    expected_capability_delta="more consistent inspection",
                ),
            ),
        },
        preliminary_ranking=[
            CandidateRankingEntry(
                candidate_id="repo-discovery",
                rank=1,
                rationale="Direct current failure with a discriminative scenario.",
            ),
            CandidateRankingEntry(
                candidate_id="unselected-planning",
                rank=2,
                rationale="Broader but less directly evidenced.",
            ),
        ],
        top_two_comparison=CandidatePairwiseComparison(
            candidate_a="repo-discovery",
            candidate_b="unselected-planning",
            strongest_case_for_a="Directly addresses the observed failure.",
            strongest_case_for_b="Could improve general task discipline.",
            comparative_judgments={
                "evidence": "repo-discovery has current terminal evidence",
                "observability": "repo-discovery has a stronger baseline/candidate contrast",
            },
            baseline_counterfactual="The baseline cannot locate the nested file.",
            candidate_counterfactual="The candidate should locate and modify it.",
            winner="repo-discovery",
            decision_reason="The direct and observable cause outweighs broader speculation.",
        ),
        selected_candidate_id="repo-discovery",
        selected_change_contract=contract,
        goals=["Improve repository discovery"],
        non_goals=["General semantic code search"],
        benefit=4,
        risk=2,
        effort=2,
        confidence=0.9,
        decision="proceed",
        decision_reason="The selected change addresses the highest-value direct cause.",
    )


def test_new_orchestrator_interface_drives_one_execution_contract():
    proposal = _new_interface_proposal()

    assert _analysis_problems(proposal) == []
    assert validate_acceptance(proposal).valid
    execution_tasks = proposal.execution_tasks()
    assert [task.id for task in execution_tasks] == ["change-repo-discovery"]
    assert {clause.id for clause in execution_tasks[0].invariants} == {
        "PG1",
        "INV1",
    }
    assert proposal.contract_allowed_write_paths() == ["demo_agent/"]


def test_deliverer_receives_proposal_guardrails_and_only_selected_contract():
    proposal = _new_interface_proposal()
    acceptance = AcceptanceRun(
        passed=True,
        runs=[RunResult("python3 -m demo_agent --help", 0, "usage: demo_agent")],
    )

    message = goal_review_message(proposal, "diff --git a/demo_agent/tools.py", acceptance)

    assert "Do not weaken workspace path confinement" in message
    assert "change-repo-discovery" in message
    assert "add and wire a workspace-confined recursive search tool" in message
    assert "unselected-planning" not in message
    assert "diff --git a/demo_agent/tools.py" in message


def test_expected_output_explains_why_and_what_evidence_should_change():
    proposal = _new_interface_proposal()
    expected = proposal.selected_change_contract.expected_outputs[0]

    assert expected.rationale
    assert "trajectory" in expected.evidence_direction


def test_scorecards_and_pairwise_review_are_recorded_for_human_inspection():
    proposal = _new_interface_proposal()

    assert proposal.candidate_backlog[
        "repo-discovery"
    ].priority.problem.evidence_strength == 5
    assert [item.candidate_id for item in proposal.preliminary_ranking] == [
        "repo-discovery",
        "unselected-planning",
    ]
    assert proposal.top_two_comparison is not None
    assert proposal.top_two_comparison.winner == proposal.selected_candidate_id

    report = render_report(PipelineResult(stage="approved", proposal=proposal))

    assert "## Orchestrator decision review" in report
    assert "Direct current failure with a discriminative scenario" in report
    assert "**Winner:** `repo-discovery`" in report
