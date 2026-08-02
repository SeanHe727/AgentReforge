from __future__ import annotations

import asyncio
import json

from agentreforge.improve.context import OrchestratorContextBuilder, summarize_target_trajectory
from agentreforge.improve.history_index import ImprovementHistoryIndex
from agentreforge.improve.records import (
    CapabilityCompletionScope,
    ComponentRecord,
    ImprovementRecordStore,
    RecursiveRunRecord,
    ReforgeLoopRecord,
)
from agentreforge.improve.trajectory import (
    append_target_trajectory,
    list_trajectories,
    load_trajectory,
    log_target_trajectory,
    tool_result_is_error,
)
from agentreforge.snapshot.service import _project_key
from agentreforge.types import Message


def test_target_trajectory_records_prompt_arguments_and_final_response(tmp_path):
    async def run():
        async def events():
            yield {
                "type": "tool_result",
                "name": "read_file",
                "arguments": {"path": "agent.py", "api_key": "must-not-leak"},
                "content": "source",
                "is_error": False,
            }
            yield {
                "type": "done",
                "turns": 1,
                "total_tokens": 4,
                "messages": [Message(role="assistant", content="finished")],
            }

        observed = []
        async for event in log_target_trajectory(
            events(),
            cwd=str(tmp_path),
            session_id="target-1",
            task_prompt="add a planning step",
            store_root=tmp_path / "traces",
        ):
            observed.append(event)
        return observed

    observed = asyncio.run(run())
    records = load_trajectory(
        str(tmp_path), "target-1", store_root=tmp_path / "traces"
    )
    assert len(observed) == 2
    assert records[0]["trajectory_kind"] == "target_agent"
    assert records[0]["task_prompt"] == "add a planning step"
    assert records[1]["arguments"]["path"] == "agent.py"
    assert records[1]["arguments"]["api_key"] == "[REDACTED]"
    assert records[2]["final_response"] == "finished"
    history_files = list((tmp_path / "traces").rglob("*.jsonl"))
    assert [path.name for path in history_files] == ["trajectory.jsonl"]

    append_target_trajectory(
        str(tmp_path),
        [
            {
                "session_id": "target-2",
                "run_id": "target-2",
                "type": "done",
                "outcome": "completed",
            }
        ],
        store_root=tmp_path / "traces",
    )
    assert list_trajectories(
        str(tmp_path), store_root=tmp_path / "traces"
    ) == ["target-2", "target-1"]
    assert load_trajectory(
        str(tmp_path), "target-2", store_root=tmp_path / "traces"
    )[0]["outcome"] == "completed"


def test_legacy_per_session_trajectory_is_imported_once(tmp_path):
    store = tmp_path / "traces"
    project = store / _project_key(tmp_path.resolve())
    project.mkdir(parents=True)
    legacy = project / "old-session.jsonl"
    legacy.write_text(
        '{"session_id":"old-session","run_id":"old-session","type":"done"}\n',
        encoding="utf-8",
    )

    first = load_trajectory(
        str(tmp_path), "old-session", store_root=store
    )
    second = load_trajectory(
        str(tmp_path), "old-session", store_root=store
    )

    assert len(first) == 1
    assert second == first
    assert (project / "trajectory.jsonl").is_file()


def test_context_keeps_target_and_reforge_histories_separate(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    target_records = [
        {
            "run_id": "target-1",
            "type": "target_run_started",
            "task_prompt": "implement multiply",
            "evidence_source": "baseline",
            "target_commit": "abc",
        },
        {
            "run_id": "target-1",
            "type": "done",
            "outcome": "completed",
            "final_response": "done",
        },
    ]
    loop = ReforgeLoopRecord(
        run_id="reforge-1",
        loop_id="reforge-1/loop_0",
        loop=0,
        base_commit="abc",
        stage="delivered",
        components=[
            ComponentRecord(
                component="orchestrator",
                status="proceed",
                summary="add structured planning",
            )
        ],
        completed=True,
        achievements=["structured planning: the agent creates an explicit plan"],
    )

    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve the coder",
        target_trajectory=target_records,
        previous_reforge_loops=[loop],
        run_manifest={"run_id": "reforge-1", "loop_base": "abc"},
    )

    assert context.target_agent_runs[0].run_id == "target-1"
    assert context.target_agent_runs[0].task_prompt == "implement multiply"
    assert context.target_agent_runs[0].evidence_source == "baseline"
    assert context.target_agent_runs[0].target_commit == "abc"
    assert context.target_agent_runs[0].is_current
    assert context.previous_reforge_loops[0].loop_id == "reforge-1/loop_0"
    assert context.previous_reforge_loops[0].component_status == {
        "orchestrator": "proceed"
    }
    assert context.previous_reforge_loops[0].achievements == [
        "structured planning: the agent creates an explicit plan"
    ]
    assert context.raw_reforge_loops["reforge-1/loop_0"].run_id == "reforge-1"
    serialized = context.model_dump(mode="json")
    assert "raw_target_evidence" not in serialized
    assert "raw_reforge_loops" not in serialized
    assert "pyproject.toml" in context.repository.manifests
    assert "main.py" in context.repository.entrypoints


def test_context_reconstructs_dynamic_backlog_with_precise_completion_scope(tmp_path):
    delivered = ReforgeLoopRecord(
        run_id="reforge-1",
        loop_id="reforge-1/loop_0",
        loop=0,
        base_commit="abc",
        stage="delivered",
        completed=True,
        diagnosis={
            "candidates": [
                {
                    "name": "confined repository search",
                    "capability_gap": "cannot discover files below the workspace",
                    "mechanism": "add a confined recursive search tool",
                    "expected_capability_delta": "find relevant nested source files",
                    "evidence_refs": ["baseline:event:2"],
                    "rejected_reason": "",
                },
                {
                    "name": "planning guidance",
                    "capability_gap": "edits before inspecting",
                    "mechanism": "add inspect-before-edit guidance",
                    "expected_capability_delta": "inspect relevant files first",
                    "rejected_reason": "lower causal confidence",
                },
            ],
            "selected_candidates": ["confined repository search"],
        },
        completion_scopes=[
            CapabilityCompletionScope(
                candidate="confined repository search",
                capability_gap="cannot discover files below the workspace",
                mechanism="add a confined recursive search tool",
                expected_capability_delta="find relevant nested source files",
                evidence_scope=["nested-search"],
                verification_level="behavior_verified",
            )
        ],
    )

    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve the coder",
        target_trajectory=[],
        previous_reforge_loops=[delivered],
        run_manifest={"run_id": "reforge-1", "loop_base": "def"},
    )

    by_name = context.improvement_backlog
    assert by_name["confined repository search"].status == "behavior_verified"
    assert (
        by_name["confined repository search"].diagnosis.capability_gap
        == "cannot discover files below the workspace"
    )
    assert by_name["confined repository search"].history.verification_scope == [
        "nested-search"
    ]
    assert by_name["planning guidance"].status == "deferred"
    assert (
        by_name["planning guidance"].history.disposition_reason
        == "lower causal confidence"
    )


def test_target_summary_uses_stable_evidence_references():
    summaries, evidence = summarize_target_trajectory(
        [
            {
                "run_id": "target-1",
                "event_id": "target-1:event:4",
                "type": "tool_result",
                "name": "grep",
                "arguments": {"pattern": "plan"},
                "content": "agent.py:4",
                "is_error": False,
            }
        ]
    )
    assert summaries[0].evidence_refs == ["target-1:event:4"]
    assert evidence[0]["trajectory_kind"] == "target_agent"
    assert evidence[0]["arguments"] == {"pattern": "plan"}


def test_nonzero_string_exit_is_normalized_for_new_and_legacy_trajectory():
    assert not tool_result_is_error("(exit 0)\nOK")
    assert tool_result_is_error("(exit 127)\npython: command not found")
    assert tool_result_is_error("error: missing file")

    summaries, evidence = summarize_target_trajectory(
        [
            {
                "run_id": "legacy",
                "event_id": "legacy:event:1",
                "type": "tool_result",
                "name": "run_bash",
                "content": "(exit 127)\npython: command not found",
                "is_error": False,
            }
        ]
    )

    assert summaries[0].tool_errors == 1
    assert evidence[0]["is_error"] is True


def test_external_evaluation_failure_is_separate_target_evidence():
    summaries, evidence = summarize_target_trajectory(
        [
            {
                "run_id": "baseline-complex",
                "event_id": "baseline-complex:event:0",
                "type": "target_run_started",
                "task_prompt": "repair the integration",
            },
            {
                "run_id": "baseline-complex",
                "event_id": "baseline-complex:event:1",
                "type": "done",
                "outcome": "completed",
                "final_response": "implemented",
            },
            {
                "run_id": "baseline-complex",
                "event_id": "baseline-complex:event:2",
                "type": "evaluation_result",
                "passed": False,
                "content": "real CLI invocation failed",
            },
        ]
    )

    assert summaries[0].outcome == "failed_verification"
    assert summaries[0].error_messages == ["real CLI invocation failed"]
    assert evidence[-1]["type"] == "evaluation_result"
    assert evidence[-1]["is_error"] is True


def test_target_summary_marks_only_current_commit_evidence_current():
    summaries, _ = summarize_target_trajectory(
        [
            {
                "run_id": "baseline",
                "type": "target_run_started",
                "target_commit": "old",
                "evidence_source": "baseline",
            },
            {
                "run_id": "delivered",
                "type": "target_run_started",
                "target_commit": "new",
                "evidence_source": "delivered_scenario",
            },
        ],
        current_commit="new",
    )

    by_id = {summary.run_id: summary for summary in summaries}
    assert not by_id["baseline"].is_current
    assert by_id["delivered"].is_current


def test_record_store_writes_run_loop_and_diff(tmp_path):
    run = RecursiveRunRecord(
        run_id="reforge-1",
        intent="improve planning",
        target_repo=str(tmp_path),
        branch="improve/reforge-1",
        base_commit="abc",
    )
    store = ImprovementRecordStore(tmp_path, run.run_id)
    store.start(run)
    store.append_loop(
        ReforgeLoopRecord(
            run_id=run.run_id,
            loop_id="reforge-1/loop_0",
            loop=0,
            base_commit="abc",
            stage="delivered",
            completed=True,
        ),
        "diff --git a/a.py b/a.py\n",
    )
    run.loop_refs.append("loops/loop_0/record.json")
    run.status = "delivered"
    store.finish(run)

    stored_run = json.loads(store.run_path.read_text(encoding="utf-8"))
    stored_loop = json.loads(
        (store.loops_dir / "loop_0" / "record.json").read_text(encoding="utf-8")
    )
    assert stored_run["record_kind"] == "reforge_recursive_run"
    assert stored_run["status"] == "delivered"
    assert stored_loop["record_kind"] == "reforge_loop"
    assert stored_loop["diff_ref"] == "loops/loop_0/diff.patch"


def test_history_index_retrieves_old_reforge_experience_not_current_facts(tmp_path):
    index = ImprovementHistoryIndex(tmp_path / ".agentreforge" / "history.db")
    old = ReforgeLoopRecord(
        run_id="old-run",
        loop_id="old-run/loop_0",
        loop=0,
        base_commit="abc",
        stage="delivered",
        proposal_summary="Add explicit planning state before tool execution",
        diagnosis={"capability_gap": "planning state"},
        completed=True,
    )
    current = ReforgeLoopRecord(
        run_id="current-run",
        loop_id="current-run/loop_0",
        loop=0,
        base_commit="def",
        stage="delivered",
        proposal_summary="Refine planning state transitions",
        completed=True,
    )
    index.index_loop(old, tmp_path)
    index.index_loop(current, tmp_path)

    matches = index.search(
        "planning state",
        target_repo=tmp_path,
        exclude_run_id="current-run",
    )

    assert matches
    assert {match["run_id"] for match in matches} == {"old-run"}
