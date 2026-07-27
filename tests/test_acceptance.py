from __future__ import annotations

from conftest import make_proposal

from metaimprove.improve.acceptance import validate_acceptance


def test_acceptance_contract_is_traceable_and_executable():
    result = validate_acceptance(make_proposal())

    assert result.valid
    assert result.errors == []


def test_acceptance_contract_rejects_unknown_and_unassigned_criteria():
    proposal = make_proposal()
    proposal.tasks[0].acceptance_criteria_ids = ["missing"]

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "unknown criteria" in result.summary()
    assert "not assigned" in result.summary()


def test_acceptance_contract_requires_scope_and_command():
    proposal = make_proposal(allowed_write_paths=[])
    proposal.acceptance_criteria[0].command = ""

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "allowed_write_paths" in result.summary()
    assert "has no command" in result.summary()


def test_acceptance_contract_rejects_thin_writer_reviewer_handoff():
    proposal = make_proposal()
    task = proposal.tasks[0]
    task.rationale = ""
    task.capability_change = ""
    task.required_behaviors = []
    task.invariants = []
    task.reviewer_focus = []

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "has no rationale" in result.summary()
    assert "has no capability_change" in result.summary()
    assert "has no required_behaviors" in result.summary()
    assert "has no invariants" in result.summary()
    assert "has no reviewer_focus" in result.summary()


def test_acceptance_contract_requires_unique_traceable_clause_ids():
    proposal = make_proposal()
    task = proposal.tasks[0]
    task.invariants[0].id = "RB1"

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "contract clause ids must be unique" in result.summary()
