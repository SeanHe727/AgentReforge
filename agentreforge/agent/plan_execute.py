"""Plan-and-Execute mode — Planner + the shared TaskExecutor (no review).

A layer ABOVE the ReAct loop: it decomposes the goal into a task DAG (Planner),
then runs the DAG through the shared `TaskExecutor` with review OFF and
independent tasks in parallel. The scheduling/worker machinery is no longer
duplicated here; this module only wires the planner, the config, and the
per-task message.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..llm.base import LlmClient
from ..orchestration.task_executor import (
    WORKER_HANDOFF_CONTRACT,
    ExecutorConfig,
    TaskExecutor,
    run_streamed,
    worker_handoff_error,
)
from ..plan.models import ExecutionPlan, Task
from ..plan.planner import Planner
from ..prompt.assembler import PromptAssembler
from ..tools.registry import ToolRegistry


async def plan_execute(
    *,
    client: LlmClient,
    registry: ToolRegistry,
    goal: str,
    cwd: str,
    memory: Any = None,
    code_index: Any = None,
    max_task_turns: int = 8,
) -> AsyncIterator[dict[str, Any]]:
    # 1. PLAN: ask the Planner to turn `goal` into a task DAG.
    yield {"type": "text_delta", "text": f"Planning: {goal}\n"}
    try:
        plan = await Planner(client).create_plan(goal)
    except Exception as exc:  # noqa: BLE001 - surface planning failure as an event
        yield {"type": "error", "error": exc}
        return
    yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}

    # 2. EXECUTE: run the DAG through the shared executor (no review, parallel).
    system_prompt = PromptAssembler(
        cwd=cwd,
        model=client.model_name,
        provider=client.provider_name,
        tool_names=registry.list_names(),
    ).build() + "\n\nFINAL OUTPUT CONTRACT:\n" + WORKER_HANDOFF_CONTRACT
    executor = TaskExecutor(client=client, registry=registry)
    config = ExecutorConfig(
        review=False,
        parallel=True,
        max_task_turns=max_task_turns,
        worker_output_validate=worker_handoff_error,
        worker_output_contract=WORKER_HANDOFF_CONTRACT,
    )
    async for event in run_streamed(
        executor,
        plan=plan,
        cwd=cwd,
        config=config,
        system_prompt=system_prompt,
        build_task_message=lambda task, pl, _fb: _task_message(pl, task),
        memory=memory,
        code_index=code_index,
    ):
        yield event

    # 3. SUMMARIZE: the executor mutated `plan` in place, so read it back.
    yield {"type": "text_delta", "text": "\n" + _final_summary(plan)}
    yield {"type": "done", "total_turns": 0, "total_tokens": 0, "messages": []}


def _task_message(plan: ExecutionPlan, task: Task) -> str:
    # Give the task typed context; dependency results remain explicitly attributed.
    prerequisites = []
    for dep_id in task.dependencies:
        dep = plan.get(dep_id)
        if dep:
            prerequisites.append(
                {
                    "task_id": dep.id,
                    "description": dep.description,
                    "status": dep.status.value,
                    "result": dep.result,
                }
            )
    return json.dumps(
        {
            "request_kind": "execute_plan_task",
            "overall_goal": plan.goal,
            "task": {
                "id": task.id,
                "description": task.description,
                "dependencies": task.dependencies,
            },
            "prerequisite_results": prerequisites,
            "instruction": "Complete this task concretely, using tools when needed.",
            "output_contract": WORKER_HANDOFF_CONTRACT,
        },
        ensure_ascii=False,
    )


def _final_summary(plan: ExecutionPlan) -> str:
    lines = ["Plan complete. Task results:"]
    for task in plan.tasks.values():
        lines.append(f"- [{task.id}] {task.status.value}: {task.description}")
    return "\n".join(lines)
