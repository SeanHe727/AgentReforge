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
    task.candidate = ""
    task.rationale = ""
    task.capability_change = ""
    task.required_behaviors = []
    task.invariants = []
    task.reviewer_focus = []

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "has no candidate" in result.summary()
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


def test_acceptance_contract_rejects_unavailable_python_interpreter(monkeypatch):
    proposal = make_proposal()
    proposal.acceptance_criteria[0].command = "PYTHONDONTWRITEBYTECODE=1 python -m tests"
    proposal.delivery_run = ["python -m package --help"]
    monkeypatch.setattr(
        "metaimprove.improve.acceptance.shutil.which",
        lambda executable: None if executable == "python" else f"/bin/{executable}",
    )

    result = validate_acceptance(proposal)

    assert not result.valid
    assert result.summary().count("unavailable interpreter 'python'") == 2


def test_path_tool_task_requires_executable_traversal_acceptance():
    proposal = make_proposal()
    task = proposal.tasks[0]
    criterion = proposal.acceptance_criteria[0]
    task.affected_components = ["coder/tools.py"]
    task.description = "Add repository navigation and directory-listing tools"

    missing_declaration = validate_acceptance(proposal)
    assert "does not require safety property 'path_confinement'" in (
        missing_declaration.summary()
    )

    task.required_safety_properties = ["path_confinement"]
    missing_coverage = validate_acceptance(proposal)
    assert "no assigned acceptance criterion" in missing_coverage.summary()

    criterion.verified_safety_properties = ["path_confinement"]
    weak_criterion = validate_acceptance(proposal)
    assert "must exercise a '..' traversal attempt" in weak_criterion.summary()
    assert "stable blocked/error output marker" in weak_criterion.summary()

    criterion.command = (
        "PYTHONDONTWRITEBYTECODE=1 python3 -c \"from coder.tools import execute; "
        "result = execute('list_dir', {'path': '..'}, '.'); "
        "assert result.startswith('error:'); print('PATH_BLOCKED')\""
    )
    criterion.required_output_contains = ["PATH_BLOCKED"]

    assert validate_acceptance(proposal).valid
