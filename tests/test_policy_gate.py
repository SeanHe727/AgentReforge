from __future__ import annotations

from conftest import make_proposal

from agentreforge.improve.policy_gate import evaluate, evaluate_changes


def test_prewrite_gate_proceeds_for_grounded_executable_contract():
    decision = evaluate(make_proposal())

    assert decision.decision == "proceed"


def test_acceptance_detail_is_not_a_policy_gate():
    proposal = make_proposal()
    proposal.acceptance_criteria[0].verification = "manual"
    proposal.acceptance_criteria[0].command = ""

    decision = evaluate(proposal)

    assert decision.decision == "proceed"


def test_suggested_scope_is_not_a_policy_gate():
    proposal = make_proposal(
        allowed_write_paths=["agentreforge/**/*.py"],
        affected_components=["agentreforge"],
    )

    decision = evaluate(proposal)

    assert decision.decision == "proceed"


def test_postwrite_gate_allows_changes_anywhere_inside_the_worktree():
    decision = evaluate_changes(
        make_proposal(),
        ["src/agent.py", "unexpected/generated.py"],
    )

    assert decision.decision == "proceed"


def test_postwrite_gate_accepts_directory_prefix_scope():
    decision = evaluate_changes(
        make_proposal(),
        ["src/agent.py", "tests/unit/test_agent.py"],
    )

    assert decision.decision == "proceed"


def test_postwrite_gate_hard_denies_generated_artifacts_even_in_scope():
    proposal = make_proposal(allowed_write_paths=["demo_agent/"])

    decision = evaluate_changes(
        proposal,
        ["demo_agent/agent.py", "demo_agent/__pycache__/agent.cpython-312.pyc"],
    )

    assert decision.decision == "deny"
    assert "generated artifacts" in decision.reasons[0]
