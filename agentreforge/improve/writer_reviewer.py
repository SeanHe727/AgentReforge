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

from ..llm.parse import parse_json_model
from ..observability import traceable
from ..orchestration.task_executor import ExecutorConfig, TaskExecutor
from ..plan.models import ExecutionPlan, Task
from ..prompt.assembler import PromptAssembler
from .models import (
    AcceptanceCriterion,
    ExecutionBlocker,
    ImprovementProposal,
    ImprovementTask,
    ReviewResult,
    WriterReport,
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

Own the task through a coherent candidate:
- Treat the supplied Shared Task Contract as immutable. Implement every required
  behavior, preserve every invariant, obey every implementation constraint, and
  do not use a prohibited shortcut.
- Modify only the exact files listed under Affected components. Never invent another
  file path; report a deviation instead if the frozen scope cannot satisfy the task.
- Inspect the relevant code before editing.
- Make focused changes, then run the task's executable acceptance commands.
- Self-inspect the final task diff.
- The Orchestrator invokes an independent read-only Reviewer after you finish. Address
  its concrete findings in the next bounded round; do not negotiate scope with it.

If a Reviewer rejected a previous attempt, address the feedback directly.

FINAL RESPONSE CONTRACT:
When done, output exactly one ```json block containing a WriterReport:
{
  "task_id": "the assigned task id",
  "summary": "what changed and why",
  "changed_files": ["repo/relative/path"],
  "clause_evidence": [
    {"clause_id": "RB1", "implementation": "what implements it",
     "evidence": "file:line or observed behavior"}
  ],
  "commands_run": [
    {"command": "exact command", "exit_code": 0, "output_summary": "key result"}
  ],
  "known_limitations": [],
  "deviations": []
}
Include one clause_evidence entry for every required review clause id. Do not claim
commands you did not run. The report is evidence for review, not proof by itself."""


@dataclass
class TaskOutcome:
    """What became of one proposal task after its Writer<->Reviewer rounds."""

    task_id: str
    status: str  # "completed" | "blocked"
    rounds: int
    review: ReviewResult
    writer_summary: str
    writer_report: WriterReport | None = None
    commit: str | None = None
    attempts: list[str] = field(default_factory=list)
    phase: str = "implementation"  # implementation | repair
    repair_iteration: int = 0


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

    @traceable(name="writer_reviewer.run", run_type="chain")
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
        criteria = {criterion.id: criterion for criterion in proposal.acceptance_criteria}
        plan = ExecutionPlan(
            goal=proposal.summary,
            tasks={
                t.id: Task(id=t.id, description=t.description, dependencies=list(t.dependencies))
                for t in proposal.tasks
            },
        )

        # The Orchestrator owns hand-offs: Writer produces one coherent task candidate,
        # then the independent Reviewer evaluates the task-scoped diff.
        config = ExecutorConfig(
            max_rounds=self.max_rounds,
            review=True,
            structural_guard=True,
            worktree=worktree,
            parallel=False,
            max_task_turns=self.max_task_turns,
            stop_on_block=True,
            writer_driven_review=False,
        )

        def build_message(task: Task, pl: ExecutionPlan, fb: ReviewResult | None) -> str:
            return _writer_message(specs[task.id], pl, fb, criteria)

        # the Reviewer is agentic and reviews each task from fixed perspectives.
        executor = self._executor(cwd)
        result = await executor.run(
            plan,
            cwd=cwd,
            config=config,
            system_prompt=system_prompt,
            build_task_message=build_message,
            task_brief=lambda task: _task_brief(specs[task.id], criteria),
            memory=memory,
            code_index=code_index,
        )

        return await self._to_outcome(result, worktree, phase="implementation")

    async def repair(
        self,
        *,
        worktree: WorktreeSession,
        instruction: str,
        allowed_write_paths: list[str],
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
            writer_driven_review=False,
        )
        executor = self._executor(cwd)
        result = await executor.run(
            plan,
            cwd=cwd,
            config=config,
            system_prompt=system_prompt,
            build_task_message=lambda task, pl, fb: _repair_message(
                instruction, allowed_write_paths, fb
            ),
            task_brief=lambda task: _repair_brief(instruction, allowed_write_paths),
            memory=memory,
            code_index=code_index,
        )
        return await self._to_outcome(result, worktree, phase="repair")

    async def _to_outcome(
        self,
        result: Any,
        worktree: WorktreeSession,
        *,
        phase: str,
    ) -> ExecutionOutcome:
        """Adapt the executor's ExecutionResult to the pipeline's ExecutionOutcome."""
        outcomes = [
            TaskOutcome(
                task_id=tr.task_id,
                status=tr.status,
                rounds=tr.rounds,
                review=tr.review or ReviewResult(verdict="revise", summary="no review"),
                writer_summary=tr.result,
                writer_report=_parse_writer_report(tr.result),
                commit=tr.commit,
                attempts=tr.attempts,
                phase=phase,
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


def _repair_message(
    instruction: str,
    allowed_write_paths: list[str],
    feedback: ReviewResult | None,
) -> str:
    """Build the Writer's prompt for a repair task."""
    lines = [
        WRITER_PROMPT_SUFFIX.strip(),
        f"\n{_repair_brief(instruction, allowed_write_paths)}",
    ]
    if feedback and feedback.findings:
        fb = "\n".join(f"- {f.description}" for f in feedback.findings)
        lines.append(f"\nThe Reviewer REJECTED your previous attempt. Fix these:\n{fb}")
    return "\n".join(lines)


def _repair_brief(instruction: str, allowed_write_paths: list[str]) -> str:
    return (
        "Shared Task Contract [repair]\n"
        "Required review clause ids: REPAIR1\n"
        f"Objective: {instruction}\n"
        "Affected components: "
        + (", ".join(allowed_write_paths) or "(none; do not modify product files)")
        + "\n"
        "Required behaviors:\n"
        "- [REPAIR1] Resolve the reported delivery failure without regressing the candidate. "
        "Revert any existing repair change outside Affected components before finishing."
    )


def _parse_writer_report(text: str) -> WriterReport | None:
    try:
        return parse_json_model(text, WriterReport)
    except ValueError:
        return None


def _task_brief(
    spec: ImprovementTask,
    criteria: dict[str, AcceptanceCriterion],
) -> str:
    """What THIS task should do — the Reviewer's goal-realization anchor."""
    checks = _criteria_text(spec, criteria)
    return f"{_task_contract_text(spec)}\n\nExecutable acceptance checks:\n{checks}"


def _writer_message(
    spec: ImprovementTask,
    plan: ExecutionPlan,
    feedback: ReviewResult | None,
    criteria: dict[str, AcceptanceCriterion],
) -> str:
    """Build the Writer's prompt: task + done prerequisites + prior feedback."""
    lines = [
        WRITER_PROMPT_SUFFIX.strip(),
        f"\n{_task_contract_text(spec)}",
        f"\nExecutable acceptance checks:\n{_criteria_text(spec, criteria)}",
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
    # on a retry, surface the Reviewer's findings and require an incremental fix:
    # the previous attempt is already on disk, so build on it, don't restart.
    if feedback and feedback.findings:
        fb = "\n".join(f"- {f.description}" for f in feedback.findings)
        lines.append(
            "\nThe Reviewer REJECTED your previous attempt. Your previous work is ALREADY "
            "on disk — read the relevant files first, then make TARGETED edits (edit_file) "
            "to fix ONLY these findings. Do NOT rewrite the files from scratch:\n" + fb
        )
    return "\n".join(lines)


def _task_contract_text(spec: ImprovementTask) -> str:
    """Render the same immutable task contract for both Writer and Reviewer."""

    def clauses(title: str, values: list[Any]) -> list[str]:
        lines = [f"{title}:"]
        lines.extend(f"- [{item.id}] {item.description}" for item in values)
        if len(lines) == 1:
            lines.append("- (none declared)")
        return lines

    review_clause_ids = [
        clause.id
        for clause in (
            *spec.required_behaviors,
            *spec.implementation_constraints,
            *spec.invariants,
            *spec.prohibited_shortcuts,
        )
    ]
    lines = [
        f"Shared Task Contract [{spec.id}]",
        f"Owning Candidate: {spec.candidate or '(legacy/unassigned)'}",
        "Required review clause ids: " + ", ".join(review_clause_ids),
        f"Objective: {spec.description}",
        f"Rationale: {spec.rationale}",
        f"Capability change: {spec.capability_change}",
        "Required safety properties: "
        + (", ".join(spec.required_safety_properties) or "(none declared)"),
        "Affected components: "
        + (", ".join(spec.affected_components) or "(use proposal write scope)"),
        *clauses("Required behaviors", spec.required_behaviors),
        *clauses("Implementation constraints", spec.implementation_constraints),
        *clauses("Invariants", spec.invariants),
        *clauses("Prohibited shortcuts", spec.prohibited_shortcuts),
        "Reviewer focus:",
        *(f"- {focus}" for focus in spec.reviewer_focus),
    ]
    return "\n".join(lines)


def _criteria_text(
    spec: ImprovementTask,
    criteria: dict[str, AcceptanceCriterion],
) -> str:
    lines: list[str] = []
    for criterion_id in spec.acceptance_criteria_ids:
        criterion = criteria.get(criterion_id)
        if criterion is None:
            continue
        command = f"; run: {criterion.command}" if criterion.command else ""
        lines.append(
            f"- [{criterion.id}/{criterion.check_type}] {criterion.description} "
            f"(verification={criterion.verification}; safety="
            f"{','.join(criterion.verified_safety_properties) or 'none'}{command})"
        )
    return "\n".join(lines) or "- (none)"
