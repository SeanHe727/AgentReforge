from __future__ import annotations

from conftest import make_proposal

from agentreforge.improve.models import (
    DiagnosticFinding,
    ImprovementBatchBudget,
    InterventionCandidate,
    OrchestratorAnalysis,
)
from agentreforge.improve.pipeline import _analysis_problems


def _analysis(*, selected: list[str], efforts: tuple[int, int] = (2, 2)):
    effort_by_name = {
        "navigation tools": efforts[0],
        "concise tool guidance": efforts[1],
    }
    return OrchestratorAnalysis(
        findings=[
            DiagnosticFinding(
                symptom="the target guesses repository structure",
                root_cause="the observation tools are incomplete",
                capability_gap="repository navigation",
                evidence_refs=["coder/tools.py:1"],
            )
        ],
        candidates=[
            InterventionCandidate(
                name="navigation tools",
                level="tool",
                mechanism="add confined list and search tools",
                expected_capability_delta="repository awareness",
                effort=efforts[0],
            ),
            InterventionCandidate(
                name="concise tool guidance",
                level="prompt",
                mechanism="tell the coder when to inspect before editing",
                expected_capability_delta="more consistent tool use",
                effort=efforts[1],
            ),
        ],
        selected_candidates=selected,
        batch_budget=ImprovementBatchBudget(
            selected_total_effort=sum(effort_by_name[name] for name in selected)
        ),
        packing_reason="both small Candidates fit one bounded Improvement Batch",
        compatibility_notes=["the prompt consumes the tool surface without conflicting writes"],
        selection_reason="highest benefit per effort",
        causal_mechanism="add observation capability and teach the active agent to use it",
        expected_capability_delta="the coder inspects relevant source before editing",
    )


def test_small_independent_candidates_can_share_one_improvement_batch():
    proposal = make_proposal(
        analysis=_analysis(selected=["navigation tools", "concise tool guidance"])
    )
    proposal.tasks[0].candidate = "navigation tools"
    proposal.tasks.append(
        proposal.tasks[0].model_copy(
            update={
                "id": "prompt",
                "candidate": "concise tool guidance",
                "description": "Add concise inspection guidance",
                "dependencies": ["implement"],
            }
        )
    )

    assert _analysis_problems(proposal) == []


def test_large_candidate_cannot_be_packed_with_another_candidate():
    proposal = make_proposal(
        analysis=_analysis(
            selected=["navigation tools", "concise tool guidance"],
            efforts=(3, 2),
        )
    )
    proposal.tasks[0].candidate = "navigation tools"
    proposal.tasks.append(
        proposal.tasks[0].model_copy(
            update={"id": "prompt", "candidate": "concise tool guidance"}
        )
    )

    assert any("only small Candidates" in item for item in _analysis_problems(proposal))


def test_every_task_and_selected_candidate_has_explicit_ownership():
    proposal = make_proposal(analysis=_analysis(selected=["navigation tools"]))
    proposal.tasks[0].candidate = "unselected candidate"

    problems = _analysis_problems(proposal)

    assert any("every Task must name" in item for item in problems)
    assert any("must own at least one Task" in item for item in problems)


def test_old_singular_candidate_json_is_migrated():
    analysis = OrchestratorAnalysis.model_validate(
        {
            "selected_candidate": "navigation tools",
            "packing_reason": "legacy proposal",
        }
    )

    assert analysis.selected_candidates == ["navigation tools"]
    assert analysis.selected_candidate == "navigation tools"
