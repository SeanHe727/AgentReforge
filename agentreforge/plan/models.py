from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)  # ids this task waits on
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.result = error


@dataclass
class ExecutionPlan:
    goal: str
    tasks: dict[str, Task] = field(default_factory=dict)

    def executable_tasks(self) -> list[Task]:
        # runnable now = still pending AND every dependency is completed.
        runnable = []
        for task in self.tasks.values():
            if task.status is not TaskStatus.PENDING:
                continue
            if all(self._is_completed(dep) for dep in task.dependencies):
                runnable.append(task)
        return runnable

    def is_all_completed(self) -> bool:
        return all(t.status is TaskStatus.COMPLETED for t in self.tasks.values())

    def is_failed(self) -> bool:
        return any(t.status is TaskStatus.FAILED for t in self.tasks.values())

    def get(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def _is_completed(self, task_id: str) -> bool:
        # read-only check: is this task done? (does not change any status)
        task = self.tasks.get(task_id)
        return task is not None and task.status is TaskStatus.COMPLETED

    def summarize(self) -> str:
        # goal, then a short line per task with its dependencies.
        lines = [f"Plan for: {self.goal}"]
        for task in self.tasks.values():
            deps = f" (after {', '.join(task.dependencies)})" if task.dependencies else ""
            lines.append(f"- [{task.id}] {task.description}{deps}")
        return "\n".join(lines)
