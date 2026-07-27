from __future__ import annotations

import asyncio
import json

from metaimprove.improve.context import OrchestratorContextBuilder, summarize_target_trajectory
from metaimprove.improve.history_index import ImprovementHistoryIndex
from metaimprove.improve.records import (
    ComponentRecord,
    ImprovementRecordStore,
    RecursiveRunRecord,
    ReforgeLoopRecord,
)
from metaimprove.improve.trajectory import load_trajectory, log_target_trajectory
from metaimprove.types import Message


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


def test_context_keeps_target_and_reforge_histories_separate(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    target_records = [
        {
            "run_id": "target-1",
            "type": "target_run_started",
            "task_prompt": "implement multiply",
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
    )

    context = OrchestratorContextBuilder(str(tmp_path)).build(
        intent="improve the coder",
        target_trajectory=target_records,
        previous_reforge_loops=[loop],
        run_manifest={"run_id": "reforge-1"},
    )

    assert context.target_agent_runs[0].run_id == "target-1"
    assert context.target_agent_runs[0].task_prompt == "implement multiply"
    assert context.previous_reforge_loops[0].loop_id == "reforge-1/loop_0"
    assert context.previous_reforge_loops[0].component_status == {
        "orchestrator": "proceed"
    }
    assert context.raw_reforge_loops["reforge-1/loop_0"].run_id == "reforge-1"
    serialized = context.model_dump(mode="json")
    assert "raw_target_evidence" not in serialized
    assert "raw_reforge_loops" not in serialized
    assert "pyproject.toml" in context.repository.manifests
    assert "main.py" in context.repository.entrypoints


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
    index = ImprovementHistoryIndex(tmp_path / ".meta-improve" / "history.db")
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
