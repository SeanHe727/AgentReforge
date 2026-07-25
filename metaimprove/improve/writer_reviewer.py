"""The improvement implementation loop — a thin config over the TaskExecutor.

The scheduler, Writer<->Reviewer rounds, structural guard, and blocker logic all
live in `orchestration.task_executor` now. This module only supplies the
improve-specific bits: the writer prompt, the per-task message (acceptance
criteria + prior feedback), the review brief, and the worktree config
(serial, reviewed, guarded, stop-on-block). Its `ExecutionOutcome` return type is
kept so the pipeline is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..orchestration.task_executor import ExecutorConfig, TaskExecutor
from ..plan.models import ExecutionPlan, Task
from ..prompt.assembler import PromptAssembler
from .models import (
    ExecutionBlocker,
    ImprovementProposal,
    ImprovementTask,
    ReviewResult,
)
from .reviewer import AgenticReviewer
from .worktree import WorktreeSession

WRITER_PROMPT_SUFFIX = """
You are the Writer on a self-improvement team. Implement EXACTLY the assigned
task by editing code in the current working directory. Stay within the task's
scope; satisfy its acceptance criteria.

CRITICAL editing rules:
- To change an EXISTING file, use `edit_file` (anchored replace) — NEVER `write_file`.
  Read the file first, then replace only the exact lines you need to change.
- Use `write_file` ONLY to create a brand-new file.
- NEVER abbreviate or elide code. Do not write placeholders like "... (existing
  content)", "# rest unchanged", or "# ...". Leave untouched code exactly as it is.

If a Reviewer rejected a previous attempt, address the feedback directly. When done,
briefly state what you changed and why."""


@dataclass
class TaskOutcome:
    """What became of one proposal task after its Writer<->Reviewer rounds."""

    task_id: str
    status: str  # "completed" | "blocked"
    rounds: int
    review: ReviewResult
    writer_summary: str
    commit: str | None = None
    attempts: list[str] = field(default_factory=list)


@dataclass
class ExecutionOutcome:
    """The result of running the whole task DAG in the worktree."""

    completed: bool
    task_outcomes: list[TaskOutcome] = field(default_factory=list)
    blocker: ExecutionBlocker | None = None
    final_commit: str | None = None
    diff: str = ""


class WriterReviewer:
    def __init__(
        self,
        *,
        client: Any,
        registry: Any,
        max_rounds: int = 3,
        max_task_turns: int = 8,
    ):
        self.client = client
        self.registry = registry
        self.max_rounds = max_rounds
        self.max_task_turns = max_task_turns

    def _executor(self, cwd: str) -> TaskExecutor:
        """An executor whose Reviewer is agentic (runs checks in the worktree)."""
        reviewer = AgenticReviewer(
            client=self.client,
            registry=self.registry,
            cwd=cwd,
            max_turns=self.max_task_turns,
        )
        return TaskExecutor(client=self.client, registry=self.registry, reviewer=reviewer)

    async def run(
        self,
        *,
        proposal: ImprovementProposal,
        worktree: WorktreeSession,
        system_prompt: str | None = None,
        memory: Any = None,
        code_index: Any = None,
    ) -> ExecutionOutcome:
        """Implement proposal.tasks in the worktree via the shared executor."""
        cwd = str(worktree.path)
        system_prompt = system_prompt or PromptAssembler(
            cwd=cwd,
            model=self.client.model_name,
            provider=self.client.provider_name,
            tool_names=self.registry.list_names(),
        ).build()

        # map the ImprovementTask DAG onto a plan; keep the richer specs by id.
        specs: dict[str, ImprovementTask] = {t.id: t for t in proposal.tasks}
        plan = ExecutionPlan(
            goal=proposal.summary,
            tasks={
                t.id: Task(id=t.id, description=t.description, dependencies=list(t.dependencies))
                for t in proposal.tasks
            },
        )

        # improve config: serial, reviewed, structurally guarded, stop on a block.
        config = ExecutorConfig(
            max_rounds=self.max_rounds,
            review=True,
            structural_guard=True,
            worktree=worktree,
            parallel=False,
            max_task_turns=self.max_task_turns,
            stop_on_block=True,
        )

        def build_message(task: Task, pl: ExecutionPlan, fb: ReviewResult | None) -> str:
            return _writer_message(specs[task.id], pl, fb)

        # the Reviewer is agentic and reviews each task from fixed perspectives.
        executor = self._executor(cwd)
        result = await executor.run(
            plan,
            cwd=cwd,
            config=config,
            system_prompt=system_prompt,
            build_task_message=build_message,
            task_brief=lambda task: _task_brief(specs[task.id]),
            memory=memory,
            code_index=code_index,
        )

        return await self._to_outcome(result, worktree)

    async def repair(
        self,
        *,
        worktree: WorktreeSession,
        instruction: str,
        system_prompt: str | None = None,
        memory: Any = None,
        code_index: Any = None,
    ) -> ExecutionOutcome:
        """Run one verify-driven repair task in the worktree (fix failing tests)."""
        cwd = str(worktree.path)
        system_prompt = system_prompt or PromptAssembler(
            cwd=cwd,
            model=self.client.model_name,
            provider=self.client.provider_name,
            tool_names=self.registry.list_names(),
        ).build()
        # a one-task plan: fix the implementation so the failing tests pass.
        plan = ExecutionPlan(
            goal="repair", tasks={"repair": Task(id="repair", description=instruction)}
        )
        config = ExecutorConfig(
            max_rounds=self.max_rounds,
            review=True,
            structural_guard=True,
            worktree=worktree,
            parallel=False,
            max_task_turns=self.max_task_turns,
            stop_on_block=False,
        )
        executor = self._executor(cwd)
        result = await executor.run(
            plan,
            cwd=cwd,
            config=config,
            system_prompt=system_prompt,
            build_task_message=lambda task, pl, fb: _repair_message(instruction, fb),
            task_brief=lambda task: instruction,
            memory=memory,
            code_index=code_index,
        )
        return await self._to_outcome(result, worktree)

    async def _to_outcome(self, result: Any, worktree: WorktreeSession) -> ExecutionOutcome:
        """Adapt the executor's ExecutionResult to the pipeline's ExecutionOutcome."""
        outcomes = [
            TaskOutcome(
                task_id=tr.task_id,
                status=tr.status,
                rounds=tr.rounds,
                review=tr.review or ReviewResult(verdict="revise", summary="no review"),
                writer_summary=tr.result,
                commit=tr.commit,
                attempts=tr.attempts,
            )
            for tr in result.tasks
        ]
        return ExecutionOutcome(
            completed=result.completed,
            task_outcomes=outcomes,
            blocker=result.blocker,
            final_commit=outcomes[-1].commit if outcomes else worktree.base_commit,
            diff=await worktree.diff(),
        )


def _repair_message(instruction: str, feedback: ReviewResult | None) -> str:
    """Build the Writer's prompt for a repair task."""
    lines = [WRITER_PROMPT_SUFFIX.strip(), f"\nRepair task: {instruction}"]
    if feedback and feedback.findings:
        fb = "\n".join(f"- {f.description}" for f in feedback.findings)
        lines.append(f"\nThe Reviewer REJECTED your previous attempt. Fix these:\n{fb}")
    return "\n".join(lines)


def _task_brief(spec: ImprovementTask) -> str:
    """What THIS task should do — the Reviewer's goal-realization anchor."""
    return spec.description


def _writer_message(
    spec: ImprovementTask,
    plan: ExecutionPlan,
    feedback: ReviewResult | None,
) -> str:
    """Build the Writer's prompt: task + done prerequisites + prior feedback."""
    lines = [
        WRITER_PROMPT_SUFFIX.strip(),
        f"\nTask [{spec.id}]: {spec.description}",
    ]
    # include the descriptions of already-completed prerequisites for context.
    done_deps = [
        plan.tasks[d].description
        for d in spec.dependencies
        if d in plan.tasks and plan.tasks[d].status.value == "completed"
    ]
    if done_deps:
        prereqs = "\n".join(f"- {d}" for d in done_deps)
        lines.append(f"\nAlready implemented prerequisites:\n{prereqs}")
    # on a retry, surface the Reviewer's findings.
    if feedback and feedback.findings:
        fb = "\n".join(f"- {f.description}" for f in feedback.findings)
        lines.append(f"\nThe Reviewer REJECTED your previous attempt. Fix these:\n{fb}")
    return "\n".join(lines)
