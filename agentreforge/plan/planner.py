from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm.base import LlmClient
from ..llm.collect import collect_text
from ..llm.parse import parse_json_model
from ..plan.models import ExecutionPlan, Task, TaskStatus
from ..types import Message

PLANNER_PROMPT = """You are agent-reforge's planner.
Decompose the user's goal into a small executable DAG of tasks.
Return ONLY JSON of this shape (no prose, no markdown):
{
  "tasks": [
    {"id": "task1", "description": "concrete step", "dependencies": []},
    {"id": "task2", "description": "next step", "dependencies": ["task1"]}
  ]
}
Make tasks independent (empty dependencies) when they can run in parallel.
Keep it to 2-5 tasks."""


# Pydantic schema: the exact shape we require from the LLM's output.
class TaskSpec(BaseModel):
    id: str
    description: str
    dependencies: list[str] = []


class PlanSpec(BaseModel):
    tasks: list[TaskSpec]


class PlannerRequest(BaseModel):
    request_kind: str
    goal: str
    completed_tasks: list[dict[str, str]] = Field(default_factory=list)
    failed_tasks: list[dict[str, str]] = Field(default_factory=list)
    retry_instruction: str = ""


class Planner:
    def __init__(self, client: LlmClient):
        self.client = client

    async def create_plan(self, goal: str) -> ExecutionPlan:
        # send request to LLM to generate a plan (no tools, just JSON text back)
        text = await collect_text(
            self.client,
            [
                Message(
                    role="user",
                    content=PlannerRequest(
                        request_kind="create_plan",
                        goal=goal,
                    ).model_dump_json(indent=2),
                )
            ],
            system_prompt=PLANNER_PROMPT,
        )
        return self._to_plan(goal, self._parse(text))

    async def replan(self, failed_plan: ExecutionPlan, reason: str = "") -> ExecutionPlan:
        # build a fresh plan for the SAME goal, informed by what already happened.
        completed = [t for t in failed_plan.tasks.values() if t.status is TaskStatus.COMPLETED]
        failed = [t for t in failed_plan.tasks.values() if t.status is TaskStatus.FAILED]

        text = await collect_text(
            self.client,
            [
                Message(
                    role="user",
                    content=PlannerRequest(
                        request_kind="replan",
                        goal=failed_plan.goal,
                        completed_tasks=[
                            {
                                "id": task.id,
                                "description": task.description,
                                "result": _preview(task.result),
                            }
                            for task in completed
                        ],
                        failed_tasks=[
                            {
                                "id": task.id,
                                "description": task.description,
                                "result": _preview(task.result),
                            }
                            for task in failed
                        ],
                        retry_instruction=(
                            reason.strip()
                            or "Diagnose the failure and replan only the remaining work."
                        ),
                    ).model_dump_json(indent=2),
                )
            ],
            system_prompt=PLANNER_PROMPT,
        )
        return self._to_plan(failed_plan.goal, self._parse(text))

    def _to_plan(self, goal: str, spec: PlanSpec) -> ExecutionPlan:
        # convert a validated PlanSpec into our ExecutionPlan domain model.
        tasks = {
            t.id: Task(id=t.id, description=t.description, dependencies=t.dependencies)
            for t in spec.tasks
        }
        return ExecutionPlan(goal=goal, tasks=tasks)

    def _parse(self, text: str) -> PlanSpec:
        # extract + validate the plan JSON (shared helper handles fences/prose).
        return parse_json_model(text, PlanSpec)


def _preview(text: str, limit: int = 160) -> str:
    value = (text or "").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."
