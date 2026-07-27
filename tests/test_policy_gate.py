from __future__ import annotations

from conftest import make_proposal

from agentreforge.improve.policy_gate import evaluate, evaluate_changes


def test_prewrite_gate_proceeds_for_grounded_executable_contract():
    decision = evaluate(make_proposal())

    assert decision.decision == "proceed"


def test_prewrite_gate_escalates_non_executable_required_criterion():
    proposal = make_proposal()
    proposal.acceptance_criteria[0].verification = "manual"
    proposal.acceptance_criteria[0].command = ""

    decision = evaluate(proposal)

    assert decision.decision == "needs_human"


def test_prewrite_gate_conservatively_detects_protected_glob():
    proposal = make_proposal(
        allowed_write_paths=["agentreforge/**/*.py"],
        affected_components=["agentreforge"],
    )

    decision = evaluate(proposal)

    assert decision.decision == "needs_human"
    assert "protected paths" in decision.reasons[0]


def test_postwrite_gate_rejects_out_of_scope_actual_path():
    decision = evaluate_changes(
        make_proposal(),
        ["src/agent.py", "unexpected/generated.py"],
    )

    assert decision.decision == "needs_human"
    assert "unexpected/generated.py" in decision.reasons[0]


def test_postwrite_gate_accepts_directory_prefix_scope():
    decision = evaluate_changes(
        make_proposal(),
        ["src/agent.py", "tests/unit/test_agent.py"],
    )

    assert decision.decision == "proceed"


def test_postwrite_gate_hard_denies_generated_artifacts_even_in_scope():
    proposal = make_proposal(allowed_write_paths=["coder/"])

    decision = evaluate_changes(
        proposal,
        ["coder/agent.py", "coder/__pycache__/agent.cpython-312.pyc"],
    )

    assert decision.decision == "deny"
    assert "generated artifacts" in decision.reasons[0]
