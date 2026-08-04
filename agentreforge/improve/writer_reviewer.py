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
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..agent.query import query
from ..observability import traceable
from ..orchestration.handoff import repair_handoff_output
from ..orchestration.task_executor import ExecutorConfig, TaskExecutor
from ..plan.models import ExecutionPlan, Task
from ..prompt.assembler import PromptAssembler
from ..tools.registry import ToolRegistry
from .models import (
    AcceptanceCriterion,
    ExecutionBlocker,
    ImprovementProposal,
    ImprovementTask,
    ReviewResult,
    WriterAcceptanceCheck,
    WriterTaskContract,
)
from .proposal_context import proposal_lookup_tool
from .reviewer import AgenticReviewer
from .worktree import WorktreeSession

WRITER_PROMPT_SUFFIX = """
You are the Writer on a self-improvement team. Implement EXACTLY the assigned
task by editing code in the current repository worktree.

Object boundary:
- This worktree is the AGENT REPOSITORY: the reusable Agent Under Improvement.
- Evaluation Tasks and Delivery Scenarios run that agent against separate disposable
  TASK WORKSPACES.
- Task-specific flags, schemas, algorithms, filenames, and business behavior from
  trajectory evidence belong to those Task Workspaces. Do not hardcode them into the
  Agent Repository. Implement the reusable agent capability named by the Contract.
- Trace every failure to the repository that owned the failing artifact. If the
  Contract asks this Agent Repository to add a concrete interface that belonged to
  generated task code, do not implement that overfit; report the Contract as blocked.
  Renaming the request as "contract fidelity" does not change its ownership.

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
- Affected components are suggestions, not an authorization wall. You may change
  another repository file when the objective genuinely requires it; explain why.
- Every changed path and behavioral hunk must map to a specific Contract clause or be
  strictly necessary to integrate that clause. Do not add unrelated hardening,
  refactors, documentation, or safety behavior merely because it is beneficial.
- Treat `implementation_direction` as the selected causal mechanism. For a behavioral
  capability change, identify and modify the executable control flow that implements
  that mechanism. Documentation-only or prompt-description-only edits cannot satisfy
  a behavioral Contract unless the objective explicitly asks for documentation.
- When code branches on an agent/model outcome, use an explicit typed or structured
  status separate from its human-readable summary. Never use broad substring/keyword
  matching over free-form final prose as a completion, blocking, or verification gate.
- Inspect the relevant code before editing.
- Make focused changes, then run the task's executable acceptance commands.
- Self-inspect the final task diff.
- `read_proposal` is optional, read-only whole-picture context for consistency
  verification. The Shared Task Contract remains the direct implementation
  instruction; never replace or enlarge it from Proposal alternatives.
- The Orchestrator invokes an independent read-only Reviewer after you finish. Address
  its concrete findings in the next bounded round; do not negotiate scope with it.

If a Reviewer rejected a previous attempt, address each finding directly. In the
final hand-off, add one `finding_resolutions` item per finding you attempted. A
resolution reports the change and verification; it does not declare itself accepted.

Your implementation in the worktree is the deliverable. The Reviewer receives the
frozen Task and authoritative Git diff, then independently inspects and tests the
current repository. Your final response is a typed hand-off, not implementation
evidence. Return ONLY:
{
  "status":"completed|blocked",
  "summary":str,
  "verification":[
    {"command":str,"outcome":"passed|failed|not_run","summary":str}
  ],
  "finding_resolutions":[
    {"finding_id":str,"change_summary":str,"verification":str}
  ],
  "blocking_reasons":[str]
}"""


WRITER_PLANNING_PROMPT = """You are the read-only planning phase of the Writer.
Inspect the repository and turn the supplied compact Writer Task Contract into a
short implementation plan before any code is changed.

Use 2-5 ordered steps. Each step names the action, likely target components, and
one verification direction. Resolve uncertainty by reading/searching the repository;
map `implementation_direction` to the actual executable control flow before proposing
an edit. A behavioral capability plan cannot consist only of documentation changes.
Do not propose a redesign, expand task scope, or repeat Contract prose. Return ONLY
one WriterPlan JSON object:
{
  "approach": str,
  "steps": [{
    "id": str,
    "action": str,
    "target_components": [str],
    "verification": str
  }],
  "risks": [str]
}"""


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
    phase: str = "implementation"  # implementation | repair
    repair_iteration: int = 0
    writer_plan: dict[str, Any] | None = None


class WriterPlanStep(BaseModel):
    id: str
    action: str
    target_components: list[str] = Field(default_factory=list, max_length=5)
    verification: str = ""


class WriterPlan(BaseModel):
    approach: str
    steps: list[WriterPlanStep] = Field(min_length=2, max_length=5)
    risks: list[str] = Field(default_factory=list, max_length=4)


class VerificationRecord(BaseModel):
    command: str
    outcome: Literal["passed", "failed", "not_run"]
    summary: str = ""


class FindingResolution(BaseModel):
    finding_id: str
    change_summary: str
    verification: str = ""


class WriterHandoff(BaseModel):
    status: Literal["completed", "blocked"]
    summary: str
    verification: list[VerificationRecord] = Field(default_factory=list)
    finding_resolutions: list[FindingResolution] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


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
        max_review_cycles: int = 3,
        writer_plan_turns: int = 4,
        writer_attempt_turns: int = 12,
        reviewer_pass_turns: int = 6,
        # Compatibility aliases for callers predating separate stage budgets.
        max_rounds: int | None = None,
        max_task_turns: int | None = None,
    ):
        self.client = client
        self.registry = registry
        self.max_review_cycles = max_rounds or max_review_cycles
        self.writer_plan_turns = writer_plan_turns
        self.writer_attempt_turns = max_task_turns or writer_attempt_turns
        self.reviewer_pass_turns = reviewer_pass_turns

    def _executor(self, cwd: str, registry: Any | None = None) -> TaskExecutor:
        """An executor whose Reviewer is agentic (runs checks in the worktree)."""
        registry = registry or self.registry
        reviewer = AgenticReviewer(
            client=self.client,
            registry=registry,
            cwd=cwd,
            max_turns=self.reviewer_pass_turns,
        )
        return TaskExecutor(client=self.client, registry=registry, reviewer=reviewer)

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
        """Implement the selected Change Contract via the shared executor."""
        cwd = str(worktree.path)
        execution_registry = _registry_with_proposal(self.registry, proposal)
        base_system_prompt = system_prompt or PromptAssembler(
            cwd=cwd,
            model=self.client.model_name,
            provider=self.client.provider_name,
            tool_names=execution_registry.list_names(),
        ).build()
        system_prompt = base_system_prompt + "\n\n" + WRITER_PROMPT_SUFFIX.strip()

        # The executor still consumes a one-task plan internally.  The source of
        # truth is the frozen Change Contract, exposed through compatibility views.
        execution_tasks = proposal.execution_tasks()
        specs: dict[str, ImprovementTask] = {t.id: t for t in execution_tasks}
        criteria = {
            criterion.id: criterion
            for criterion in proposal.contract_acceptance_criteria()
        }
        writer_contracts = {
            spec.id: _writer_contract(proposal, spec, criteria)
            for spec in execution_tasks
        }
        writer_plans = {
            spec.id: await self._plan_writer(
                contract=writer_contracts[spec.id],
                cwd=cwd,
                registry=execution_registry,
            )
            for spec in execution_tasks
        }
        plan = ExecutionPlan(
            goal=proposal.summary,
            tasks={
                t.id: Task(id=t.id, description=t.description, dependencies=list(t.dependencies))
                for t in execution_tasks
            },
        )

        # The Orchestrator owns hand-offs: Writer produces one coherent task candidate,
        # then the independent Reviewer evaluates the task-scoped diff.
        config = ExecutorConfig(
            max_rounds=self.max_review_cycles,
            review=True,
            structural_guard=True,
            worktree=worktree,
            parallel=False,
            max_task_turns=self.writer_attempt_turns,
            stop_on_block=True,
            writer_driven_review=False,
            worker_output_validate=_writer_handoff_error,
            worker_output_contract=(
                "One WriterHandoff JSON object with status completed|blocked, "
                "summary:string, verification, finding_resolutions, and "
                "blocking_reasons:list[str]."
            ),
        )

        def build_message(task: Task, pl: ExecutionPlan, fb: ReviewResult | None) -> str:
            return _writer_message(
                specs[task.id],
                pl,
                fb,
                criteria,
                contract=writer_contracts[task.id],
                writer_plan=writer_plans[task.id],
            )

        # the Reviewer is agentic and reviews each task from fixed perspectives.
        executor = self._executor(cwd, execution_registry)
        result = await executor.run(
            plan,
            cwd=cwd,
            config=config,
            system_prompt=system_prompt,
            build_task_message=build_message,
            task_brief=lambda task: _task_brief(
                specs[task.id],
                criteria,
                contract=writer_contracts[task.id],
            ),
            memory=memory,
            code_index=code_index,
        )

        return await self._to_outcome(
            result,
            worktree,
            phase="implementation",
            writer_plans=writer_plans,
        )

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
        system_prompt = system_prompt + "\n\n" + WRITER_PROMPT_SUFFIX.strip()
        # a one-task plan: fix the implementation so the failing tests pass.
        plan = ExecutionPlan(
            goal="repair", tasks={"repair": Task(id="repair", description=instruction)}
        )
        config = ExecutorConfig(
            max_rounds=self.max_review_cycles,
            review=True,
            structural_guard=True,
            worktree=worktree,
            parallel=False,
            max_task_turns=self.writer_attempt_turns,
            stop_on_block=False,
            writer_driven_review=False,
            worker_output_validate=_writer_handoff_error,
            worker_output_contract=(
                "One WriterHandoff JSON object with status completed|blocked, "
                "summary:string, verification, finding_resolutions, and "
                "blocking_reasons:list[str]."
            ),
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

    async def _plan_writer(
        self,
        *,
        contract: WriterTaskContract,
        cwd: str,
        registry: ToolRegistry,
    ) -> WriterPlan:
        """Run one bounded read-only planning phase before implementation."""

        planning_registry = ToolRegistry()
        planning_registry.register_all(
            [
                tool
                for name in registry.list_names()
                if (tool := registry.get(name)) is not None and tool.is_read_only
            ]
        )
        planning_system_prompt = PromptAssembler(
            cwd=cwd,
            model=self.client.model_name,
            provider=self.client.provider_name,
            tool_names=planning_registry.list_names(),
        ).build()
        message = _json_handoff(
            {
                "request_kind": "writer_plan",
                "writer_task_contract": contract.model_dump(mode="json"),
                "output_contract": "WriterPlan",
            }
        )
        text = ""
        async for event in query(
            client=self.client,
            registry=planning_registry,
            system_prompt=(
                planning_system_prompt + "\n\n" + WRITER_PLANNING_PROMPT.strip()
            ),
            user_message=message,
            cwd=cwd,
            max_turns=self.writer_plan_turns,
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise ValueError(f"Writer planning failed: {event['error']}")
        error = _writer_plan_error(text)
        if error:
            repaired = await repair_handoff_output(
                self.client,
                producer="WriterPlanner",
                invalid_output=text,
                validation_error=error,
                contract="One WriterPlan JSON object with 2-5 ordered steps.",
                context=message,
                validate=_writer_plan_error,
            )
            if repaired.error:
                raise ValueError(f"Writer planning handoff failed: {repaired.error}")
            text = repaired.text
        return WriterPlan.model_validate_json(text)

    async def _to_outcome(
        self,
        result: Any,
        worktree: WorktreeSession,
        *,
        phase: str,
        writer_plans: dict[str, WriterPlan] | None = None,
    ) -> ExecutionOutcome:
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
                phase=phase,
                writer_plan=(
                    writer_plans[tr.task_id].model_dump(mode="json")
                    if writer_plans and tr.task_id in writer_plans
                    else None
                ),
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
    return _json_handoff(
        {
            "request_kind": "writer_repair",
            "task_contract": {
                "id": "repair",
                "objective": instruction,
                "suggested_components": allowed_write_paths,
                "required_behaviors": [
                    {
                        "id": "REPAIR1",
                        "description": (
                            "Resolve the reported delivery failure without "
                            "regressing the candidate."
                        ),
                    }
                ],
            },
            "review_feedback": (
                feedback.model_dump(mode="json") if feedback is not None else None
            ),
            "output_contract": {
                "status": "completed|blocked",
                "summary": "string",
                "verification": ["VerificationRecord"],
                "finding_resolutions": ["FindingResolution"],
                "blocking_reasons": ["string"],
            },
        }
    )


def _repair_brief(instruction: str, allowed_write_paths: list[str]) -> str:
    return _json_handoff(
        {
            "contract_kind": "delivery_repair",
            "id": "repair",
            "objective": instruction,
            "affected_components": allowed_write_paths,
            "required_behaviors": [
                {
                    "id": "REPAIR1",
                    "description": (
                        "Resolve the reported delivery failure without regressing "
                        "the candidate."
                    ),
                }
            ],
        }
    )


def _task_brief(
    spec: ImprovementTask,
    criteria: dict[str, AcceptanceCriterion],
    *,
    contract: WriterTaskContract | None = None,
) -> str:
    """What THIS task should do — the Reviewer's goal-realization anchor."""
    contract = contract or _compact_contract_from_spec(spec, criteria)
    return _json_handoff(
        {
            "writer_task_contract": contract.model_dump(mode="json"),
        }
    )


def _writer_message(
    spec: ImprovementTask,
    execution_plan: ExecutionPlan,
    feedback: ReviewResult | None,
    criteria: dict[str, AcceptanceCriterion],
    *,
    contract: WriterTaskContract | None = None,
    writer_plan: WriterPlan | None = None,
) -> str:
    """Build the Writer's prompt: task + done prerequisites + prior feedback."""
    # include the descriptions of already-completed prerequisites for context.
    done_deps = [
        execution_plan.tasks[d].description
        for d in spec.dependencies
        if d in execution_plan.tasks
        and execution_plan.tasks[d].status.value == "completed"
    ]
    contract = contract or _compact_contract_from_spec(spec, criteria)
    return _json_handoff(
        {
            "request_kind": "writer_task",
            "writer_task_contract": contract.model_dump(mode="json"),
            "writer_plan": (
                writer_plan.model_dump(mode="json")
                if writer_plan is not None
                else None
            ),
            "completed_prerequisites": done_deps,
            "review_feedback": (
                feedback.model_dump(mode="json") if feedback is not None else None
            ),
            "retry_policy": (
                "Previous work remains on disk. Treat WriterPlan as a working plan, "
                "make targeted edits for each stable-ID finding, and report one "
                "FindingResolution per attempted finding."
                if feedback is not None
                else ""
            ),
            "output_contract": {
                "status": "completed|blocked",
                "summary": "string",
                "verification": ["VerificationRecord"],
                "finding_resolutions": ["FindingResolution"],
                "blocking_reasons": ["string"],
            },
        }
    )


def _writer_contract(
    proposal: ImprovementProposal,
    spec: ImprovementTask,
    criteria: dict[str, AcceptanceCriterion],
) -> WriterTaskContract:
    selected = proposal.selected_change_contract
    if selected is not None and selected.contract_id == spec.id:
        return selected.as_writer_contract()
    return _compact_contract_from_spec(spec, criteria)


def _compact_contract_from_spec(
    spec: ImprovementTask,
    criteria: dict[str, AcceptanceCriterion],
) -> WriterTaskContract:
    """Compatibility compiler for legacy proposals without a Change Contract."""

    checks = [
        criteria[criterion_id]
        for criterion_id in spec.acceptance_criteria_ids
        if criterion_id in criteria
    ][:3]
    return WriterTaskContract(
        contract_id=spec.id,
        objective=spec.description,
        capability_delta=spec.capability_change,
        implementation_direction=spec.capability_change,
        requirements=spec.required_behaviors[:5],
        constraints=[
            *spec.implementation_constraints,
            *spec.invariants,
        ][:4],
        non_goals=spec.prohibited_shortcuts[:3],
        suggested_components=spec.affected_components[:6],
        required_safety_properties=spec.required_safety_properties,
        acceptance_checks=[
            WriterAcceptanceCheck(
                id=item.id,
                description=item.description,
                verification=item.verification,
            )
            for item in checks
        ],
    )


def _registry_with_proposal(
    registry: Any,
    proposal: ImprovementProposal,
) -> ToolRegistry:
    """Scope an immutable Proposal lookup to this one Writer/Reviewer Loop."""

    scoped = ToolRegistry()
    scoped.register_all(
        [
            tool
            for name in registry.list_names()
            if (tool := registry.get(name)) is not None
        ]
    )
    scoped.register(proposal_lookup_tool(proposal))
    return scoped


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
        "Suggested components: "
        + (", ".join(spec.affected_components) or "(inspect the repository)"),
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


def _writer_handoff_error(text: str) -> str:
    try:
        output = WriterHandoff.model_validate_json(text)
    except ValueError as exc:
        return str(exc)
    if not output.summary.strip():
        return "WriterHandoff summary must explain the outcome"
    if output.status == "completed" and output.blocking_reasons:
        return "completed contradicts non-empty blocking_reasons"
    if output.status == "blocked" and not output.blocking_reasons:
        return "blocked requires concrete blocking_reasons"
    resolution_ids = [item.finding_id for item in output.finding_resolutions]
    if len(resolution_ids) != len(set(resolution_ids)):
        return "finding_resolutions may reference each finding only once"
    return ""


def _writer_plan_error(text: str) -> str:
    try:
        output = WriterPlan.model_validate_json(text)
    except ValueError as exc:
        return str(exc)
    if not output.approach.strip():
        return "WriterPlan approach must explain the implementation strategy"
    return ""


def _json_handoff(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
