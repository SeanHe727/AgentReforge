from __future__ import annotations

from agentreforge.improve.acceptance import validate_acceptance
from agentreforge.improve.acceptance_runner import AcceptanceRun, RunResult
from agentreforge.improve.deliverer import goal_review_message
from agentreforge.improve.models import (
    AcceptanceCriterion,
    BacklogCandidate,
    CandidateDiagnosis,
    CandidateIntervention,
    ContractClause,
    ImprovementProposal,
    SelectedChangeContract,
)
from agentreforge.improve.pipeline import _analysis_problems


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
            ContractClause(
                id="OUT1",
                description="The target can report a relevant nested source path.",
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
