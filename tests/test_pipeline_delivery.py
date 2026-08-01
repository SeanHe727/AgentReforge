from __future__ import annotations

import asyncio

from conftest import make_proposal

from agentreforge.improve import pipeline as pipeline_module
from agentreforge.improve.acceptance_runner import RunResult, ScenarioRunResult
from agentreforge.improve.delivery_coordinator import Delivery
from agentreforge.improve.models import ReviewResult
from agentreforge.improve.pipeline import (
    ImprovementPipeline,
    ImprovementVersion,
    PipelineResult,
    _as_converged,
    _as_partial,
    _changed_files_from_diff,
    _combine_execution_outcomes,
    _delivered_target_trajectory,
    _delivery_is_writer_repairable,
    _has_next_loop,
    _loop_failure_summary,
    _negative_attempt_problems,
    _proposal_attempt_fingerprint,
    _repair_instruction,
    _tag_baseline_trajectory,
)
from agentreforge.improve.records import ComponentRecord, ReforgeLoopRecord
from agentreforge.improve.run_config import RecursionPolicy
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


def test_repair_instruction_includes_exit_zero_output_mismatches():
    delivery = Delivery(
        passed=False,
        acceptance_failures=["AC5: output missing 'verify'"],
        runs=[
            RunResult(
                "python3 -c \"print('verification')\"",
                0,
                "verification",
            )
        ],
    )

    instruction = _repair_instruction(delivery)

    assert "output missing 'verify'" in instruction
    assert "(exit 0)" in instruction
    assert "verification" in instruction
    assert "do not game" in instruction


def test_only_implementation_delivery_failure_returns_to_writer():
    implementation = Delivery(passed=False, failure_kind="implementation_defect")
    verification = Delivery(passed=False, failure_kind="verification_gap")
    plan = Delivery(passed=False, failure_kind="plan_gap")

    assert _delivery_is_writer_repairable(
        implementation,
        repairs=0,
        max_repairs=1,
    )
    assert not _delivery_is_writer_repairable(
        verification,
        repairs=0,
        max_repairs=1,
    )
    assert not _delivery_is_writer_repairable(
        plan,
        repairs=0,
        max_repairs=1,
    )
    assert not _delivery_is_writer_repairable(
        implementation,
        repairs=1,
        max_repairs=1,
    )


def test_negative_attempt_ledger_rejects_same_failed_verification_strategy():
    proposal = make_proposal()
    proposal.analysis.selected_candidates = ["focused fix"]
    prior = ReforgeLoopRecord(
        run_id="run",
        loop_id="run/loop_0",
        loop=0,
        base_commit="base",
        stage="rejected_delivery",
        diagnosis={"selected_candidates": ["focused fix"]},
        failure_kind="verification_gap",
        attempt_fingerprint=_proposal_attempt_fingerprint(proposal),
        error="missing trajectory",
    )

    problems = _negative_attempt_problems(proposal, [prior])

    assert len(problems) == 1
    assert "same Candidate and verification strategy" in problems[0]


def test_negative_attempt_ledger_allows_changed_candidate_and_strategy():
    previous = make_proposal()
    previous.analysis.selected_candidates = ["navigation"]
    current = make_proposal()
    current.analysis.selected_candidates = ["runtime fallback"]
    prior = ReforgeLoopRecord(
        run_id="run",
        loop_id="run/loop_0",
        loop=0,
        base_commit="base",
        stage="rejected_delivery",
        diagnosis={"selected_candidates": ["navigation"]},
        failure_kind="verification_gap",
        attempt_fingerprint=_proposal_attempt_fingerprint(previous),
        error="missing trajectory",
    )

    assert _negative_attempt_problems(current, [prior]) == []


def test_negative_attempt_ledger_rejects_repeated_failed_safety_contract():
    proposal = make_proposal()
    proposal.analysis.selected_candidates = ["runtime fallback"]
    proposal.tasks[0].required_safety_properties = ["path_confinement"]
    prior = ReforgeLoopRecord(
        run_id="run",
        loop_id="run/loop_0",
        loop=0,
        base_commit="base",
        stage="rejected_delivery",
        diagnosis={"selected_candidates": ["runtime fallback"]},
        failure_kind="plan_gap",
        error=(
            "system safety probe 'adapter:safety:path_confinement': "
            "exit 1, expected 0"
        ),
        components=[
            ComponentRecord(
                component="orchestrator",
                status="proceed",
                details={"proposal": proposal.model_dump(mode="json")},
            )
        ],
    )
    proposal.delivery_run = ["python3 -c \"print('different strategy')\""]

    problems = _negative_attempt_problems(proposal, [prior])

    assert any("already failed the same safety contract" in problem for problem in problems)


def test_run_report_changed_files_come_from_authoritative_diff():
    diff = """diff --git a/demo_agent/agent.py b/demo_agent/agent.py
index 111..222 100644
--- a/demo_agent/agent.py
+++ b/demo_agent/agent.py
diff --git a/README.md b/README.md
index 333..444 100644
--- a/README.md
+++ b/README.md
"""

    assert _changed_files_from_diff(diff) == [
        "demo_agent/agent.py",
        "README.md",
    ]


def test_delivered_scenario_becomes_current_target_trajectory():
    result = PipelineResult(
        stage="delivered",
        loop=1,
        delivery=Delivery(
            passed=True,
            scenario_runs=[
                ScenarioRunResult(
                    scenario_id="fallback",
                    prompt="verify with an available interpreter",
                    command=["python3", "-m", "demo_agent"],
                    exit_code=0,
                    output="verified",
                    trajectory=[
                        {
                            "type": "tool_result",
                            "name": "run_bash",
                            "arguments": {"command": "python3 -m unittest"},
                            "is_error": False,
                            "content": "(exit 0) OK",
                        },
                        {
                            "type": "done",
                            "outcome": "completed",
                            "final_response": "verified",
                        },
                    ],
                    trajectory_available=True,
                )
            ],
        ),
        version=ImprovementVersion(
            loop=1,
            base_commit="base",
            branch="improve/test",
            verified_commit="new-commit",
            proposal_hash="proposal",
            evaluation_hash="",
        ),
    )

    records = _delivered_target_trajectory("run", result)

    assert records[0]["type"] == "target_run_started"
    assert records[0]["target_commit"] == "new-commit"
    assert records[0]["evidence_source"] == "delivered_scenario"
    assert records[1]["name"] == "run_bash"
    assert records[-1]["type"] == "done"
    assert all(record["run_id"] == "run:loop:1:scenario:fallback" for record in records)


def test_baseline_trajectory_is_bound_to_recursive_run_base_commit():
    records = _tag_baseline_trajectory(
        [{"run_id": "baseline", "type": "target_run_started"}],
        commit="base-commit",
    )

    assert records[0]["target_commit"] == "base-commit"
    assert records[0]["evidence_source"] == "baseline"


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


def test_failed_gate_uses_remaining_budget_instead_of_stopping_recursive_run():
    assert _has_next_loop(loop_i=0, max_loops=3)
    assert _has_next_loop(loop_i=1, max_loops=3)
    assert not _has_next_loop(loop_i=2, max_loops=3)


def test_recursive_run_continues_after_a_rejected_delivery(monkeypatch, tmp_path):
    class FakeWorktreeSession:
        def __init__(self, _cwd, *, base, keep):
            self.branch = "improve/test"
            self.base_commit = "base"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def head(self):
            return "base"

    class SequencedPipeline(ImprovementPipeline):
        def __init__(self):
            super().__init__(
                client=FakeClient(),
                cwd=str(tmp_path),
                registry=ToolRegistry(),
                deliverer=FakeDeliverer(),
                recursion=RecursionPolicy(max_loops=2),
            )
            self.loop_calls = 0

        async def _preflight(self, wt, intent, traj):
            return {
                "run_id": "recovering-run",
                "intent": intent,
                "target": str(tmp_path),
                "base_commit": "base",
            }

        async def _run_loop(self, *_args):
            loop_i = _args[-1]
            self.loop_calls += 1
            if loop_i == 0:
                return PipelineResult(
                    stage="rejected_delivery",
                    loop=0,
                    delivery=Delivery(passed=False, reasons=["handoff failed"]),
                )
            return PipelineResult(
                stage="delivered",
                loop=1,
                delivery=Delivery(passed=True),
            )

        async def _finalize(self, _wt, result, _manifest, _all_results):
            return result

    monkeypatch.setattr(pipeline_module, "WorktreeSession", FakeWorktreeSession)

    pipeline = SequencedPipeline()
    result = asyncio.run(pipeline.run(intent="improve planning", target_trajectory=[]))

    assert pipeline.loop_calls == 2
    assert result.stage == "delivered"


def test_delivery_failure_summary_is_available_to_next_orchestrator_loop():
    result = PipelineResult(
        stage="rejected_delivery",
        loop=0,
        delivery=Delivery(
            passed=False,
            reasons=[
                "deterministic delivery/commit gate failed",
                "AC1: output missing 'planned'",
            ],
        ),
    )

    assert _loop_failure_summary(result) == (
        "deterministic delivery/commit gate failed; "
        "AC1: output missing 'planned'"
    )


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
        delivery_gate_ok=False,
        acceptance_failures=["ac1: output missing 'verify'"],
        runs=[],
        goal_accepted=True,
        goal_review="GOAL: ACHIEVED\nVERDICT: ACCEPT",
    )

    instruction = _repair_instruction(delivery)

    assert "AcceptanceRunner failures" in instruction
    assert "ac1: output missing 'verify'" in instruction
    assert "Deliverer goal review" not in instruction
