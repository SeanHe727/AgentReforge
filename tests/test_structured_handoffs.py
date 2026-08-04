from __future__ import annotations

import asyncio
import json

from agentreforge.agent.orchestrator import _worker_message
from agentreforge.agent.plan_execute import _task_message
from agentreforge.orchestration.task_executor import (
    ExecutorConfig,
    TaskExecutor,
    worker_handoff_error,
)
from agentreforge.plan.models import ExecutionPlan, Task


def _plan() -> ExecutionPlan:
    prerequisite = Task(id="inspect", description="Inspect the implementation")
    prerequisite.mark_completed('{"status":"completed","summary":"found entrypoint"}')
    task = Task(
        id="change",
        description="Implement the selected change",
        dependencies=["inspect"],
    )
    return ExecutionPlan(
        goal="Improve the agent",
        tasks={prerequisite.id: prerequisite, task.id: task},
    )


def test_plan_worker_input_is_a_structured_envelope():
    plan = _plan()

    payload = json.loads(_task_message(plan, plan.get("change")))

    assert payload["request_kind"] == "execute_plan_task"
    assert payload["task"]["id"] == "change"
    assert payload["prerequisite_results"][0]["task_id"] == "inspect"
    assert "output_contract" in payload


def test_reviewed_worker_input_keeps_feedback_in_its_own_field():
    plan = _plan()

    payload = json.loads(
        _worker_message(plan, plan.get("change"), "The integration test failed.")
    )

    assert payload["request_kind"] == "execute_reviewed_task"
    assert payload["review_feedback"] == "The integration test failed."
    assert payload["task"]["description"] == "Implement the selected change"


def test_worker_completion_is_not_inferred_from_free_text():
    assert worker_handoff_error("DONE: tests passed")
    assert not worker_handoff_error(
        json.dumps(
            {
                "status": "completed",
                "summary": "Implemented the selected change.",
                "blocking_reasons": [],
                "evidence": ["tests/test_feature.py passed"],
            }
        )
    )


def test_missing_worker_final_response_gets_budget_independent_handoff(monkeypatch):
    plan = ExecutionPlan(
        goal="Improve the agent",
        tasks={"change": Task(id="change", description="Implement the change")},
    )
    executor = TaskExecutor(client=object(), registry=object(), reviewer=object())
    finalized = []

    async def run_worker(*args, **kwargs):
        return "(no output)", None

    async def finalize(client, *, producer, contract, context):
        finalized.append(context)
        return json.dumps(
            {
                "status": "completed",
                "summary": "Implementation work finished.",
                "blocking_reasons": [],
                "evidence": [],
            }
        )

    executor._run_worker = run_worker  # type: ignore[method-assign]
    monkeypatch.setattr(
        "agentreforge.orchestration.task_executor.finalize_handoff_output",
        finalize,
    )

    result = asyncio.run(
        executor.run(
            plan,
            cwd=".",
            config=ExecutorConfig(
                review=False,
                parallel=False,
                max_task_turns=1,
                worker_output_validate=worker_handoff_error,
                worker_output_contract="WorkerHandoff",
            ),
            system_prompt="test",
            build_task_message=lambda task, _plan, _feedback: task.description,
        )
    )

    assert result.completed
    assert len(finalized) == 1
    assert finalized[0]["worker_execution_output"] == "(no output)"
