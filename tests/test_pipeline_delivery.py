from __future__ import annotations

import asyncio

from conftest import make_proposal

from agentreforge.improve.delivery_coordinator import Delivery
from agentreforge.improve.models import ReviewResult
from agentreforge.improve.pipeline import (
    ImprovementPipeline,
    ImprovementVersion,
    PipelineResult,
    _as_converged,
    _as_partial,
    _combine_execution_outcomes,
    _repair_instruction,
)
from agentreforge.improve.writer_reviewer import ExecutionOutcome, TaskOutcome
from agentreforge.tools.registry import ToolRegistry


class FakeClient:
    model_name = "fake"
    provider_name = "fake"


class FakeDeliverer:
    def __init__(self):
        self.received_diff = ""

    async def deliver(self, proposal, *, cwd, loop_diff):
        self.received_diff = loop_diff
        return Delivery(passed=True, reasons=["accepted"])


class FakeWorktree:
    def __init__(self, trees):
        self.trees = iter(trees)
        self.diff_calls = 0

    async def snapshot(self):
        return next(self.trees)

    async def diff_since(self, ref):
        self.diff_calls += 1
        if ref == "candidate":
            return "delivery mutation"
        return f"loop diff {self.diff_calls}"


def _pipeline(deliverer):
    return ImprovementPipeline(
        client=FakeClient(),
        cwd=".",
        registry=ToolRegistry(),
        deliverer=deliverer,
    )


def test_deliverer_receives_loop_base_diff_and_unchanged_tree_passes():
    deliverer = FakeDeliverer()
    worktree = FakeWorktree(["candidate", "candidate"])

    delivery, final_diff = asyncio.run(
        _pipeline(deliverer)._deliver_immutable(
            make_proposal(), worktree, "loop-base", "/candidate"
        )
    )

    assert deliverer.received_diff == "loop diff 1"
    assert final_diff == "loop diff 2"
    assert delivery.passed
    assert delivery.integrity_ok
    assert delivery.verified_tree == "candidate"


def test_delivery_mutation_is_rejected_and_reported():
    deliverer = FakeDeliverer()
    worktree = FakeWorktree(["candidate", "mutated"])

    delivery, _ = asyncio.run(
        _pipeline(deliverer)._deliver_immutable(
            make_proposal(), worktree, "loop-base", "/candidate"
        )
    )

    assert not delivery.passed
    assert not delivery.integrity_ok
    assert delivery.verified_tree == ""
    assert delivery.mutation_diff == "delivery mutation"
    assert "mutated the candidate" in delivery.reasons[0]


def test_later_loop_failure_is_reported_as_partial_delivery():
    success = PipelineResult(
        stage="delivered",
        loop=0,
        delivery=Delivery(passed=True),
        version=ImprovementVersion(
            loop=0,
            base_commit="base",
            branch="improve/test",
            verified_commit="verified",
            proposal_hash="proposal",
            evaluation_hash="",
        ),
    )
    terminal = PipelineResult(
        stage="rejected_delivery",
        loop=1,
        delivery=Delivery(passed=False, reasons=["bad command"]),
        report_path="/tmp/terminal.md",
    )

    result = _as_partial(success, terminal)

    assert result.stage == "partially_delivered"
    assert result.version.verified_commit == "verified"
    assert result.terminal_loop == 1
    assert result.terminal_stage == "rejected_delivery"
    assert result.terminal_error == "bad command"
    assert result.terminal_report_path == "/tmp/terminal.md"


def test_later_orchestrator_abstention_is_graceful_convergence():
    success = PipelineResult(
        stage="delivered",
        loop=2,
        delivery=Delivery(passed=True),
        version=ImprovementVersion(
            loop=2,
            base_commit="base",
            branch="improve/test",
            verified_commit="verified",
            proposal_hash="proposal",
            evaluation_hash="",
        ),
    )
    abstention = PipelineResult(stage="abstained", loop=3)

    result = _as_converged(success, abstention)

    assert result.stage == "converged"
    assert result.version.verified_commit == "verified"
    assert result.terminal_loop == 3
    assert result.terminal_stage == "abstained"


def test_delivery_repair_appends_to_original_task_history():
    original = ExecutionOutcome(
        completed=True,
        task_outcomes=[
            TaskOutcome(
                task_id="t1",
                status="completed",
                rounds=1,
                review=ReviewResult(verdict="accept"),
                writer_summary="implemented tools",
                phase="implementation",
            )
        ],
        diff="initial diff",
    )
    repair = ExecutionOutcome(
        completed=True,
        task_outcomes=[
            TaskOutcome(
                task_id="repair",
                status="completed",
                rounds=1,
                review=ReviewResult(verdict="accept"),
                writer_summary="fixed import",
                phase="repair",
                repair_iteration=1,
            )
        ],
        diff="repaired diff",
    )

    combined = _combine_execution_outcomes(original, repair)

    assert combined.completed
    assert [task.task_id for task in combined.task_outcomes] == ["t1", "repair"]
    assert [task.phase for task in combined.task_outcomes] == ["implementation", "repair"]
    assert combined.task_outcomes[1].repair_iteration == 1
    assert combined.diff == "repaired diff"


def test_repair_instruction_includes_output_assertion_failures_with_exit_zero():
    delivery = Delivery(
        passed=False,
        hard_gate_ok=False,
        acceptance_failures=["ac1: output missing 'verify'"],
        runs=[],
        goal_accepted=True,
        goal_review="GOAL: ACHIEVED\nVERDICT: ACCEPT",
    )

    instruction = _repair_instruction(delivery)

    assert "AcceptanceRunner failures" in instruction
    assert "ac1: output missing 'verify'" in instruction
    assert "Deliverer goal review" not in instruction
