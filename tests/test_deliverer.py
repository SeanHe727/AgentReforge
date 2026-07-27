from __future__ import annotations

import asyncio

from conftest import make_proposal

from metaimprove.improve.acceptance_runner import (
    AcceptanceRun,
    RunResult,
    acceptance_failures,
    dangerous_command,
)
from metaimprove.improve.deliverer import GoalReview, goal_review_message
from metaimprove.improve.delivery_coordinator import DeliveryCoordinator
from metaimprove.improve.models import (
    DiagnosticFinding,
    InterventionCandidate,
    OrchestratorAnalysis,
)


def test_acceptance_hard_gate_checks_exit_and_output():
    proposal = make_proposal()
    criterion = proposal.acceptance_criteria[0]
    criterion.expected_exit_code = 0
    criterion.required_output_contains = ["1 passed"]
    criterion.forbidden_output_contains = ["Traceback"]

    failures = acceptance_failures(
        proposal,
        [RunResult(criterion.command, 0, "1 passed in 0.1s")],
    )

    assert failures == []


def test_acceptance_hard_gate_reports_output_mismatch():
    proposal = make_proposal()
    criterion = proposal.acceptance_criteria[0]
    criterion.required_output_contains = ["1 passed"]

    failures = acceptance_failures(
        proposal,
        [RunResult(criterion.command, 1, "failed")],
    )

    assert any("exit 1" in failure for failure in failures)
    assert any("output missing" in failure for failure in failures)


def test_delivery_command_denylist_blocks_destructive_git():
    assert dangerous_command("git reset --hard HEAD") is not None
    assert dangerous_command("python -m pytest tests") is None


def test_goal_review_message_is_proposal_and_diff_not_command_output():
    proposal = make_proposal(
        goals=["make repository navigation reachable"],
        analysis=OrchestratorAnalysis(
            findings=[
                DiagnosticFinding(
                    symptom="no navigation",
                    root_cause="missing tools",
                    capability_gap="repository awareness",
                    evidence_refs=["coder/tools.py"],
                )
            ],
            candidates=[
                InterventionCandidate(
                    name="navigation tools",
                    level="tool",
                    mechanism="add and wire list/search tools",
                    expected_capability_delta="repository awareness",
                )
            ],
            selected_candidate="navigation tools",
            causal_mechanism="wire tools into the active tool surface",
            expected_capability_delta="repository awareness",
        ),
        delivery_checklist=["new tools are wired into the active agent loop"],
    )

    message = goal_review_message(proposal, "diff --git a/coder/tools.py")

    assert "navigation tools" in message
    assert "wire tools into the active tool surface" in message
    assert "new tools are wired into the active agent loop" in message
    assert "diff --git a/coder/tools.py" in message
    assert "Run output:" not in message


def test_delivery_coordinator_requires_both_runner_and_deliverer():
    class FakeRunner:
        async def run(self, proposal, *, cwd):
            return AcceptanceRun(
                passed=False,
                runs=[RunResult("test", 1, "failed")],
                failures=["test failed"],
            )

    class FakeDeliverer:
        async def review(self, proposal, *, loop_diff):
            return GoalReview(
                accepted=True,
                text="GOAL: ACHIEVED\nVERDICT: ACCEPT",
            )

    coordinator = DeliveryCoordinator(
        runner=FakeRunner(),
        deliverer=FakeDeliverer(),
    )

    result = asyncio.run(
        coordinator.deliver(
            make_proposal(),
            cwd="/candidate",
            loop_diff="diff --git a/a.py b/a.py",
        )
    )

    assert not result.passed
    assert not result.hard_gate_ok
    assert result.acceptance_failures == ["test failed"]
    assert result.goal_accepted
    assert "test failed" in result.reasons
    assert result.goal_review.startswith("GOAL: ACHIEVED")
