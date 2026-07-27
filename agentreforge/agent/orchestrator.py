"""Multi-Agent mode (Planner / Worker / Reviewer) — Planner + shared TaskExecutor.

Same as Plan-and-Execute but with the Reviewer ON: each task's result is reviewed
and, if rejected, retried with feedback (bounded). The review loop is no longer
implemented here — it is the shared `TaskExecutor` with review=True; this module
only wires the planner, the config, and the per-task worker message.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..llm.base import LlmClient
from ..orchestration.task_executor import (
    ExecutorConfig,
    TaskExecutor,
    run_streamed,
)
from ..plan.models import ExecutionPlan, Task
from ..plan.planner import Planner
from ..prompt.assembler import PromptAssembler
from ..tools.registry import ToolRegistry


async def multi_agent(
    *,
    client: LlmClient,
    registry: ToolRegistry,
    goal: str,
    cwd: str,
    memory: Any = None,
    code_index: Any = None,
    max_task_turns: int = 8,
    max_retries: int = 2,
) -> AsyncIterator[dict[str, Any]]:
    # 1. PLANNER decomposes the goal into a task DAG.
    yield {"type": "text_delta", "text": f"[planner] planning: {goal}\n"}
    try:
        plan = await Planner(client).create_plan(goal)
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "error": exc}
        return
    yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}

    # 2. EXECUTE with review ON: workers run in parallel, each gated by the Reviewer.
    system_prompt = PromptAssembler(
        cwd=cwd,
        model=client.model_name,
        provider=client.provider_name,
        tool_names=registry.list_names(),
    ).build()
    executor = TaskExecutor(client=client, registry=registry)
    config = ExecutorConfig(
        review=True, parallel=True, max_rounds=max_retries + 1, max_task_turns=max_task_turns
    )
    async for event in run_streamed(
        executor,
        plan=plan,
        cwd=cwd,
        config=config,
        system_prompt=system_prompt,
        build_task_message=lambda task, pl, fb: _worker_message(pl, task, fb.summary if fb else ""),
        memory=memory,
        code_index=code_index,
    ):
        yield event

    # 3. SUMMARIZE from the plan the executor mutated in place.
    yield {"type": "text_delta", "text": "\n" + _final_summary(plan)}
    yield {"type": "done", "total_turns": 0, "total_tokens": 0, "messages": []}


def _worker_message(plan: ExecutionPlan, task: Task, feedback: str) -> str:
    lines = [f"Overall goal: {plan.goal}", f"Your task: {task.description}"]
    if task.dependencies:
        lines.append("\nResults from prerequisite tasks:")
        for dep_id in task.dependencies:
            dep = plan.get(dep_id)
            if dep:
                lines.append(f"- [{dep.id}] {dep.description}: {dep.result}")
    if feedback:
        lines.append(f"\nA reviewer rejected your previous attempt. Fix this: {feedback}")
    lines.append("\nComplete this task concretely, using tools when needed.")
    return "\n".join(lines)


def _final_summary(plan: ExecutionPlan) -> str:
    lines = ["Team run complete. Task results:"]
    for task in plan.tasks.values():
        lines.append(f"- [{task.id}] {task.status.value}: {task.description}")
    return "\n".join(lines)
