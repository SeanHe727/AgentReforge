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
