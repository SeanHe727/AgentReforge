"""One task-graph executor shared by every execution mode.

plan-execute, multi-agent, and the improvement Writer<->Reviewer loop were three
copies of the same skeleton: schedule the runnable tasks, drive each through the
ReAct worker, optionally review with bounded retries, and propagate failure. This
is that skeleton, once. Callers vary it through `ExecutorConfig` (review on/off,
serial/parallel, worktree scope, structural guard, stop-on-block) and two
message-building callbacks — nothing else is duplicated.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Protocol

from ..agent.query import query
from ..improve.models import ExecutionBlocker, Finding, ReviewResult
from ..plan.models import ExecutionPlan, Task
from ..tools.base import Tool, ToolResult, object_schema
from ..tools.registry import ToolRegistry


class WorktreeLike(Protocol):
    # the slice of WorktreeSession the executor needs (keeps it decoupled).
    path: Path

    async def commit(self, message: str, *, expected_tree: str | None = None) -> str | None: ...
    async def diff(self) -> str: ...
    async def changed_paths(self) -> list[str]: ...
    async def changed_since(self, ref: str) -> list[str]: ...
    async def snapshot(self) -> str: ...
    async def diff_since(self, ref: str) -> str: ...
    async def restore_snapshot(self, tree: str) -> None: ...
    async def read_base(self, rel_path: str) -> str | None: ...

# text meaning the model abbreviated instead of writing real code — a clobber sign.
_PLACEHOLDER_MARKERS = (
    "... (existing content)",
    "... (truncated)",
    "# rest unchanged",
    "# ... rest",
    "# ... (unchanged)",
    "# existing code",
    "<!-- ... -->",
)

# builds the worker prompt for a task (caller-specific): (task, plan, feedback).
BuildTaskMessage = Callable[[Task, ExecutionPlan, "ReviewResult | None"], str]
# what the Reviewer reviews a task AGAINST; defaults to the task description.
TaskBrief = Callable[[Task], str]
# optional side-channel for streaming progress events to a consumer.
EventSink = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ExecutorConfig:
    max_rounds: int = 1  # worker<->reviewer rounds per task (1 = single shot)
    review: bool = False  # run the Reviewer between rounds
    structural_guard: bool = False  # deterministic file checks before the Reviewer
    worktree: WorktreeLike | None = None  # commit per round + review the diff if set
    parallel: bool = True  # run independent tasks concurrently (forced off with a worktree)
    max_task_turns: int = 8
    stop_on_block: bool = False  # stop the whole run + emit a blocker on non-convergence
    # give the worker a request_review tool so the Writer pushes each increment itself.
    writer_driven_review: bool = False


@dataclass
class TaskResult:
    task_id: str
    status: str  # "completed" | "blocked"
    rounds: int
    result: str
    review: ReviewResult | None = None
    commit: str | None = None
    attempts: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    completed: bool
    tasks: list[TaskResult] = field(default_factory=list)
    blocker: ExecutionBlocker | None = None
    plan: ExecutionPlan | None = None


class TaskExecutor:
    def __init__(self, *, client: Any, registry: Any, reviewer: Any = None):
        # Reviewer is injected so tests can pass a fake Reviewer; only used if review=on.
        from ..agent.reviewer import Reviewer

        self.client = client
        self.registry = registry
        self.reviewer = reviewer or Reviewer(client)

    async def run(
        self,
        plan: ExecutionPlan,
        *,
        cwd: str,
        config: ExecutorConfig,
        system_prompt: str,
        build_task_message: BuildTaskMessage,
        task_brief: TaskBrief | None = None,
        memory: Any = None,
        code_index: Any = None,
        event_sink: EventSink | None = None,
    ) -> ExecutionResult:
        """Execute the plan's task DAG in dependency order, mutating task state."""
        task_brief = task_brief or (lambda t: t.description)
        # a shared worktree can't take concurrent writes, so it forces serial.
        parallel = config.parallel and config.worktree is None

        results: list[TaskResult] = []
        # scheduler: repeatedly take the tasks whose dependencies are satisfied.
        while batch := plan.executable_tasks():
            run_now = batch if parallel else batch[:1]
            if len(run_now) > 1:
                await _emit(event_sink, {
                    "type": "text_delta",
                    "text": f"Running in parallel: {', '.join(t.id for t in run_now)}\n",
                })
                converged = await asyncio.gather(*(
                    self._converge_task(t, plan, config, cwd, system_prompt, build_task_message,
                                        task_brief, memory, code_index, event_sink)
                    for t in run_now
                ))
            else:
                converged = [await self._converge_task(
                    run_now[0], plan, config, cwd, system_prompt,
                    build_task_message, task_brief, memory, code_index, event_sink)]

            # record each result and reflect it back onto the plan's task state.
            for tr in converged:
                results.append(tr)
                task = plan.get(tr.task_id)
                if task is None:
                    continue
                if tr.status == "completed":
                    task.mark_completed(tr.result)
                    await _emit(event_sink, _task_event(tr, is_error=False))
                else:
                    task.mark_failed(tr.review.summary if tr.review else "did not converge")
                    await _emit(event_sink, _task_event(tr, is_error=True))
                    # stop-on-block ends the whole run with a blocker (improve flow).
                    if config.stop_on_block:
                        blocker = self._build_blocker(plan, tr)
                        return ExecutionResult(False, results, blocker, plan)

        return ExecutionResult(plan.is_all_completed(), results, None, plan)

    async def _converge_task(
        self,
        task: Task,
        plan: ExecutionPlan,
        config: ExecutorConfig,
        cwd: str,
        system_prompt: str,
        build_task_message: BuildTaskMessage,
        task_brief: TaskBrief,
        memory: Any,
        code_index: Any,
        event_sink: EventSink | None,
    ) -> TaskResult:
        """Drive one task through up to max_rounds of worker + optional Reviewer."""
        attempts: list[str] = []
        last_review: ReviewResult | None = None
        # task-scoped snapshot: the Reviewer + guard see only THIS task's increment,
        # not prior tasks/loops. No per-round commits — the loop commits once, later.
        task_start = await config.worktree.snapshot() if config.worktree is not None else None
        brief = task_brief(task)

        # writer-driven review: give the worker a request_review tool so IT pushes each
        # increment to the Reviewer as it goes, instead of only being reviewed at round end.
        worker_registry: Any = self.registry
        if config.writer_driven_review and config.worktree is not None:
            worker_registry = self._registry_with(
                self._request_review_tool(config, task_start, brief)
            )

        for round_i in range(1, config.max_rounds + 1):
            # run the worker (with the prior Reviewer's feedback on retries).
            feedback = last_review if round_i > 1 else None
            worker_text, err = await self._run_worker(
                build_task_message(task, plan, feedback), cwd, system_prompt,
                config.max_task_turns, memory, code_index, registry=worker_registry,
            )
            attempts.append(worker_text or err or "(no output)")

            # a worker error is treated like a rejected round: retry or block.
            if err is not None:
                last_review = _revise(task.id, f"worker error: {err}")
                await _emit(event_sink, _reject_event(task.id, round_i, err))
                continue

            # no review configured -> accept the single shot.
            if not config.review:
                return TaskResult(task.id, "completed", round_i, worker_text, None, None, attempts)

            # end-of-round gate: the SAME structural-guard + Reviewer check the worker can
            # invoke inline via request_review, now enforced on the round's final state.
            verdict = await self._run_review(
                config, task_start, brief, worker_text, location=task.id
            )
            last_review = verdict
            if verdict.verdict == "accept":
                await _emit(event_sink,
                            {"type": "text_delta", "text": f"[reviewer] approved {task.id}\n"})
                return TaskResult(
                    task.id, "completed", round_i, worker_text, verdict, None, attempts
                )
            await _emit(event_sink, _reject_event(task.id, round_i, verdict.summary))

        # rounds exhausted without an accept.
        return TaskResult(task.id, "blocked", config.max_rounds,
                          attempts[-1] if attempts else "", last_review, None, attempts)

    async def _run_review(
        self,
        config: ExecutorConfig,
        task_start: str | None,
        brief: str,
        body_text: str,
        *,
        location: str = "review",
    ) -> ReviewResult:
        """Structural guard + Reviewer on THIS task's increment (shared by the round
        gate and the Writer's inline request_review tool)."""
        # deterministic structural guard first (a clobber/syntax/shrink is a hard reject).
        if config.structural_guard and config.worktree is not None:
            findings = await self._structural_findings(
                config.worktree,
                since=task_start,
                declared_scope=_declared_affected_components(brief),
            )
            if findings:
                return ReviewResult(verdict="revise", findings=findings,
                                    summary="Structural guard rejected the edit.")
        # the Reviewer sees the body (worker output or the Writer's note) + the diff.
        review_input = body_text
        if config.worktree is not None and task_start is not None:
            diff = await config.worktree.diff_since(task_start)
            review_input += f"\n\n--- diff (this task) ---\n{diff}"
        review_tree = (
            await config.worktree.snapshot()
            if config.worktree is not None
            else None
        )
        review = await self.reviewer.review(task=brief, result=review_input)
        if config.worktree is not None and review_tree is not None:
            after_review = await config.worktree.snapshot()
            if after_review != review_tree:
                mutation = await config.worktree.diff_since(review_tree)
                await config.worktree.restore_snapshot(review_tree)
                return _revise(
                    location,
                    "Reviewer mutated the candidate while verifying it; the mutation was "
                    f"discarded. Use read-only checks.\n{mutation[:2000]}",
                )
        return self._to_review(review, location=location)

    def _request_review_tool(
        self, config: ExecutorConfig, task_start: str | None, brief: str
    ) -> Tool:
        """A tool the Writer calls to push its latest increment to the Reviewer."""
        async def handler(args: dict[str, Any], _context: Any) -> ToolResult:
            note = str(args.get("summary") or "(no summary)")
            verdict = await self._run_review(config, task_start, brief, note,
                                             location="request_review")
            return ToolResult(content=_render_verdict(verdict), is_error=False)

        return Tool(
            name="request_review",
            description=(
                "Ask the Reviewer to review the increment you JUST implemented. Call this "
                "after finishing a small, coherent part (e.g. one file or function). It "
                "returns APPROVED or concrete changes to make. You must obtain an APPROVED "
                "covering your final state before you finish."
            ),
            parameters=object_schema(
                {"summary": {"type": "string",
                             "description": "Briefly, what you just implemented for review."}},
                ["summary"],
            ),
            handler=handler,
            is_read_only=True,
        )

    def _registry_with(self, extra: Tool) -> ToolRegistry:
        """A per-task copy of the registry with one extra tool added."""
        reg = ToolRegistry()
        reg.register_all([t for n in self.registry.list_names() if (t := self.registry.get(n))])
        reg.register(extra)
        return reg

    async def _run_worker(
        self,
        message: str,
        cwd: str,
        system_prompt: str,
        max_task_turns: int,
        memory: Any,
        code_index: Any,
        registry: Any = None,
    ) -> tuple[str, str | None]:
        """One worker turn: drive the ReAct loop, returning (text, error-or-None)."""
        text = ""
        async for event in query(
            client=self.client,
            registry=registry or self.registry,
            system_prompt=system_prompt,
            user_message=message,
            cwd=cwd,
            memory=memory,
            code_index=code_index,
            max_turns=max_task_turns,
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                # capture the error instead of raising, so one bad task can't crash the run.
                return text.strip(), str(event["error"])
        return text.strip() or "(no output)", None

    async def _structural_findings(
        self,
        worktree: WorktreeLike,
        since: str | None = None,
        declared_scope: list[str] | None = None,
    ) -> list[Finding]:
        """Deterministic checks on changed files that signal a clobbered file.
        Scoped to files changed since `since` (this task) when given."""
        findings: list[Finding] = []
        changed = await (worktree.changed_since(since) if since else worktree.changed_paths())
        outside = _outside_declared_scope(changed, declared_scope or [])
        if outside:
            findings.append(
                Finding(
                    severity="blocker",
                    location=", ".join(outside),
                    description=(
                        "Task diff exceeds its frozen Affected components: "
                        + ", ".join(outside)
                    ),
                    required_fix=(
                        "Revert every out-of-scope file while preserving only the "
                        "authorized task files."
                    ),
                )
            )
        for rel in changed:
            if not rel.endswith(".py"):
                continue
            try:
                src = (worktree.path / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            # check placeholder text (the model abbreviated the file).
            marker = next((m for m in _PLACEHOLDER_MARKERS if m in src), None)
            if marker is not None:
                findings.append(Finding(
                    severity="blocker", location=rel,
                    description=f"{rel} contains an elision placeholder ({marker!r}). "
                    "Never abbreviate code — use edit_file for the target lines only.",
                    required_fix="Restore the full file and edit it with edit_file, "
                    "not write_file.",
                ))
                continue

            # check the file still parses.
            try:
                compile(src, rel, "exec")
            except SyntaxError as exc:
                findings.append(Finding(
                    severity="blocker", location=rel,
                    description=f"{rel} no longer parses: {exc.msg} (line {exc.lineno}).",
                    required_fix="Fix the syntax; keep the rest of the file intact.",
                ))
                continue

            # check the line count didn't drop drastically (an overwrite dropped code).
            base = await worktree.read_base(rel)
            if base is not None:
                base_lines, now_lines = base.count("\n") + 1, src.count("\n") + 1
                if base_lines >= 20 and now_lines < base_lines * 0.5:
                    findings.append(Finding(
                        severity="blocker", location=rel,
                        description=f"{rel} shrank from {base_lines} to {now_lines} lines — "
                        "likely an accidental overwrite that dropped existing code.",
                        required_fix="Use edit_file to change only what the task needs.",
                    ))
        return findings


def _declared_affected_components(brief: str) -> list[str]:
    marker = "Affected components:"
    for line in brief.splitlines():
        if line.startswith(marker):
            rendered = line.removeprefix(marker).strip()
            if not rendered or rendered.startswith("("):
                return []
            return [item.strip() for item in rendered.split(",") if item.strip()]
    return []


def _outside_declared_scope(changed: list[str], scopes: list[str]) -> list[str]:
    if not scopes:
        return []
    outside: list[str] = []
    for path in changed:
        clean = path.removeprefix("./")
        if not any(
            clean == scope.removeprefix("./")
            or (
                scope.removeprefix("./").endswith("/")
                and clean.startswith(scope.removeprefix("./"))
            )
            or fnmatch(clean, scope.removeprefix("./"))
            for scope in scopes
        ):
            outside.append(path)
    return outside

    def _to_review(self, review: Any, *, location: str) -> ReviewResult:
        """Adapt the Reviewer's approved/feedback verdict into a ReviewResult."""
        # an explicit approval accepts; anything else is a revise with a finding.
        if getattr(review, "approved", False):
            return ReviewResult(verdict="accept", summary="Reviewer approved the change.")
        raw = getattr(review, "feedback", "") or "Rejected without specific feedback."
        return _revise(location, raw.strip())

    def _build_blocker(self, plan: ExecutionPlan, tr: TaskResult) -> ExecutionBlocker:
        # report the blast radius; the recommendation is always request_human.
        affected = _dependents(plan, tr.task_id)
        return ExecutionBlocker(
            task_id=tr.task_id,
            review_rounds=tr.rounds,
            unresolved_findings=tr.review.findings if tr.review else [],
            attempted_fixes=tr.attempts,
            affected_tasks=affected,
            recommendation="request_human",
        )


def _render_verdict(verdict: ReviewResult) -> str:
    """Turn a review verdict into the text the Writer reads from request_review."""
    if verdict.verdict == "accept":
        return "APPROVED — the increment looks good. Continue with the next part, or finish."
    if verdict.findings:
        lines = ["CHANGES REQUESTED — fix these before finishing:"]
        lines += [f"- [{f.severity}] {f.location}: {f.description}" for f in verdict.findings]
        return "\n".join(lines)
    return f"CHANGES REQUESTED: {verdict.summary}"


def _revise(location: str, feedback: str) -> ReviewResult:
    finding = Finding(
        severity="major", location=location, description=feedback, required_fix=feedback
    )
    return ReviewResult(verdict="revise", findings=[finding], summary=feedback)


def _dependents(plan: ExecutionPlan, task_id: str) -> list[str]:
    """All tasks that transitively depend on `task_id`."""
    reverse: dict[str, list[str]] = {t.id: [] for t in plan.tasks.values()}
    for t in plan.tasks.values():
        for dep in t.dependencies:
            if dep in reverse:
                reverse[dep].append(t.id)
    seen: list[str] = []
    stack = list(reverse.get(task_id, []))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.append(cur)
        stack.extend(reverse.get(cur, []))
    return seen


def _task_event(tr: TaskResult, *, is_error: bool) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "name": f"task:{tr.task_id}",
        "content": tr.result,
        "is_error": is_error,
    }


def _reject_event(task_id: str, round_i: int, why: str) -> dict[str, Any]:
    text = f"[reviewer] rejected {task_id} (round {round_i}): {why}\n"
    return {"type": "text_delta", "text": text}


async def _emit(sink: EventSink | None, event: dict[str, Any]) -> None:
    if sink is not None:
        await sink(event)


async def run_streamed(executor: TaskExecutor, **run_kwargs: Any):
    """Bridge executor.run's event_sink into an async generator for CLI streaming.

    The plan is mutated in place, so a caller can summarize from it after the
    stream ends; the ExecutionResult itself is not surfaced here.
    """
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    async def sink(ev: dict[str, Any]) -> None:
        await queue.put(("event", ev))

    async def drive() -> None:
        try:
            await executor.run(event_sink=sink, **run_kwargs)
        except Exception as exc:  # noqa: BLE001 - surface a run failure as an event
            await queue.put(("error", exc))
        await queue.put((done, None))

    driver = asyncio.create_task(drive())
    while True:
        kind, payload = await queue.get()
        if kind is done:
            break
        if kind == "event":
            yield payload
        elif kind == "error":
            yield {"type": "error", "error": payload}
    await driver
