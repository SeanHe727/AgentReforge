from __future__ import annotations

from conftest import make_proposal

from metaimprove.improve.deliverer import RunResult, _acceptance_failures, _dangerous


def test_acceptance_hard_gate_checks_exit_and_output():
    proposal = make_proposal()
    criterion = proposal.acceptance_criteria[0]
    criterion.expected_exit_code = 0
    criterion.required_output_contains = ["1 passed"]
    criterion.forbidden_output_contains = ["Traceback"]

    failures = _acceptance_failures(
        proposal,
        [RunResult(criterion.command, 0, "1 passed in 0.1s")],
    )

    assert failures == []


def test_acceptance_hard_gate_reports_output_mismatch():
    proposal = make_proposal()
    criterion = proposal.acceptance_criteria[0]
    criterion.required_output_contains = ["1 passed"]

    failures = _acceptance_failures(
        proposal,
        [RunResult(criterion.command, 1, "failed")],
    )

    assert any("exit 1" in failure for failure in failures)
    assert any("output missing" in failure for failure in failures)


def test_delivery_command_denylist_blocks_destructive_git():
    assert _dangerous("git reset --hard HEAD") is not None
    assert _dangerous("python -m pytest tests") is None
