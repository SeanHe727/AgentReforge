"""An agentic Reviewer — reviews one task's diff by actually running things.

Unlike the plain text Reviewer, this one drives the ReAct loop with read-only
tools + bash: it reads the changed files, checks they import/compile, runs smoke
or targeted commands to see the behavior really works, checks for hardcoded/faked
values and out-of-scope edits, and covers every point on the Orchestrator's
review checklist. It cannot edit product code (its registry has no write tools),
matching "run a smoke test command rather than write files". Its verdict keeps
the same `Review(approved, feedback)` shape, so the executor is unchanged.
"""

from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..agent.query import query
from ..agent.reviewer import Review
from ..llm.parse import parse_json_model
from ..observability import traceable
from ..orchestration.handoff import repair_handoff_output
from ..tools.base import Tool, ToolContext, ToolResult
from ..tools.registry import ToolRegistry
from .models import Finding

REVIEWER_PROMPT = """You are an agentic code Reviewer, like a careful senior engineer.
A Writer just implemented ONE task. Its authoritative hand-off is the frozen Task
plus the task-scoped Git diff and current repository state. Any Writer note is
optional context, not evidence. Review ONLY this task's change, by actually checking
(not guessing), from these perspectives — each applies to every task, so a well-done
task passes them all:

1. Important functions & interfaces — expected inputs are handled, return values and
   errors are correct, and important edge cases behave sensibly.
2. Dataflow & schema — data flows correctly through the changed code; types and
   structures are consistent and actually consumed.
3. Shared contract — audit EVERY numbered required behavior, implementation
   constraint, invariant, and prohibited shortcut in the supplied contract. The code
   must genuinely produce the declared capability change, not merely resemble it.
   Trace behavioral claims through actual control flow. Prompt prose, comments,
   flags that are never consumed, and advisory text appended before returning do not
   implement a promised retry, continuation, or verification action.
   For a behavioral capability Contract, a documentation-only diff is not an
   implementation. Verify that the selected `implementation_direction` changes the
   executable agent mechanism. A generic startup command or an agent's confident
   final summary cannot prove that mechanism changed.
   If control flow depends on an agent/model outcome, require an explicit structured
   status or signal. Reject broad keyword matching over free-form final prose when a
   word such as "verify", "blocking", or "done" can bypass the real state transition.
   The optional `read_proposal` tool exposes frozen whole-picture context. Use it only
   when needed to verify global goals, non-goals, evidence, or dependencies; it cannot
   redefine the Shared Task Contract.
   On a repair pass, `prior_review` contains the previous stable-ID findings and
   the Writer handoff may contain matching `finding_resolutions`. Verify each
   resolution against the current cumulative diff. Preserve the same finding ID
   when the same defect remains; omit it only when the code and checks resolve it.
4. Code quality — readable, consistent with the codebase, no obvious bugs, no
   unrelated edits, no hardcoded values.
   This repository is the reusable AGENT UNDER IMPROVEMENT. Evaluation-task flags,
   schemas, algorithms, filenames, and business behavior belong to its disposable
   TASK WORKSPACE. Reject an implementation that hardcodes those task symptoms into
   the agent instead of implementing the Contract's reusable capability.
   Determine which repository owned the failing artifact in the cited trajectory.
   Reject a diff that adds a generated task's concrete CLI flag, schema, or business
   interface to the Agent Repository, even when the Contract calls it "contract
   fidelity"; that is task overfitting, not a reusable agent improvement.
   Audit every changed path and behavioral hunk for causal scope. Affected components
   are not a hard path allowlist, but each change must map to a numbered Contract
   clause or be strictly necessary integration. Reject unrelated behavioral hardening
   (including an undeclared safety feature) as scope creep; being useful is not enough.
5. Targeted verification — run focused unit or function-level checks over the important
   changed behavior. Do not duplicate the Deliverer's end-to-end startup/goal run, and
   do not turn generated-code quality into a component unit-test requirement.
6. Declared safety properties — when the Task requires `path_confinement`, actually
   run a relative traversal attempt such as `..` and confirm it is rejected. A
   read-only operation is not automatically confined, and comments/docstrings are not
   runtime evidence.

How to check: read the changed files; confirm they import / compile; run targeted
or smoke commands with the bash tool (e.g. `python3 -c "..."`) to see it really
works. Every bash command runs against a fresh disposable COPY of the candidate.
`$AGENTREFORGE_REVIEW_SCRATCH` is writable for temporary tests and probes. The copied
candidate source and evaluation files are immutable evidence: if a command changes
them, that command's result is discarded. You must report a needed code/eval change
to the Writer; never repair the candidate or acceptance conditions yourself.
Judge the supplied task-scoped diff as the expected implementation change. Do not
require a pristine worktree, treat that diff itself as a runtime artifact, or demand
that the Writer prove Delivery commands preserve repository state. Loop-level
integration, command side effects, and pre/post tree equality belong to the later
DeliveryCoordinator, which checks immutable Git snapshots deterministically.

Judge by IMPACT, not perfection. REJECT only for a REAL defect: a requirement in
"what this task should do" is not met, or there is a genuine bug (crash, deadlock,
wrong result, security issue, or brittle logic that breaks in NORMAL use). If the core
requirements are correctly met and only minor, non-blocking nits remain (contrived
edge cases, naming, style, nice-to-haves), APPROVE — you may mention the nits, but do
NOT block on them. Do not invent new requirements beyond the task.

When done, return exactly one JSON object:
{
  "verdict": "approve" or "needs_fix",
  "blocking_findings": [
    {"id": "F1", "severity": "major", "location": "file:line",
     "description": "real defect",
     "evidence": "observed code/runtime fact", "required_fix": "concrete fix"}
  ],
  "non_blocking_findings": [],
  "summary": "concise assessment"
}
Only real, evidenced defects belong in blocking_findings. Naming, style, optional
documentation, and speculative edge cases are non-blocking."""


class ReviewerOutput(BaseModel):
    verdict: Literal["approve", "needs_fix"]
    blocking_findings: list[Finding] = Field(default_factory=list)
    non_blocking_findings: list[Finding] = Field(default_factory=list)
    summary: str = ""


def review_registry(full: ToolRegistry, *, source_root: str) -> ToolRegistry:
    """Read original source; execute checks only in fresh disposable copies."""

    allowed = {
        "read_file",
        "list_dir",
        "glob",
        "grep",
        "search_code",
        "read_proposal",
    }
    scoped = ToolRegistry()
    scoped.register_all(
        [t for name in full.list_names() if name in allowed and (t := full.get(name))]
    )
    bash = full.get("bash")
    if bash is not None:
        scoped.register(_isolated_review_bash(bash, source_root=source_root))
    return scoped


def _isolated_review_bash(bash: Tool, *, source_root: str) -> Tool:
    """Wrap bash so it cannot mutate candidate code or frozen evaluation files."""

    source = Path(source_root).resolve()

    async def run(args: dict[str, Any], context: ToolContext) -> ToolResult:
        with tempfile.TemporaryDirectory(prefix="agentreforge-review-") as temp:
            root = Path(temp)
            candidate = root / "candidate"
            scratch = root / "scratch"
            scratch.mkdir()
            shutil.copytree(
                source,
                candidate,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".agentreforge",
                    "__pycache__",
                    ".pytest_cache",
                    "*.pyc",
                ),
            )
            before = _review_tree_snapshot(candidate)
            command_args = dict(args)
            command = str(command_args.get("command") or "")
            command_args["command"] = (
                f"export AGENTREFORGE_REVIEW_SCRATCH={shlex.quote(str(scratch))}; "
                f"{command}"
            )
            isolated_context = ToolContext(
                cwd=str(candidate),
                memory=context.memory,
                code_index=context.code_index,
                approval_callback=context.approval_callback,
            )
            result = await bash.handler(command_args, isolated_context)
            changed = _changed_review_paths(before, _review_tree_snapshot(candidate))
            if changed:
                return ToolResult(
                    "Error: Reviewer verification mutated the disposable candidate "
                    "or evaluation files, so its runtime result is invalid. Write probes "
                    "under $AGENTREFORGE_REVIEW_SCRATCH and report required product "
                    "changes to the Writer. Changed paths: "
                    + ", ".join(changed[:20]),
                    is_error=True,
                )
            return result

    return Tool(
        name="bash",
        description=(
            "Run one targeted verification command in a fresh disposable copy of the "
            "candidate. Candidate code/eval files must remain unchanged. Write temporary "
            "tests or probes only under $AGENTREFORGE_REVIEW_SCRATCH."
        ),
        parameters=bash.parameters,
        handler=run,
        is_read_only=True,
    )


def _review_tree_snapshot(root: Path) -> dict[str, bytes]:
    ignored = {".git", ".agentreforge", "__pycache__", ".pytest_cache"}
    snapshot: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        snapshot[str(path.relative_to(root))] = path.read_bytes()
    return snapshot


def _changed_review_paths(
    before: dict[str, bytes],
    after: dict[str, bytes],
) -> list[str]:
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )


class AgenticReviewer:
    def __init__(
        self,
        *,
        client: Any,
        registry: ToolRegistry,
        cwd: str,
        max_turns: int = 6,
    ):
        self.client = client
        # Read the original; execute commands only against disposable copies.
        self.registry = review_registry(registry, source_root=cwd)
        self.cwd = cwd
        self.max_turns = max_turns

    @traceable(name="reviewer.review", run_type="chain")
    async def review(self, task: str, result: str) -> Review:
        """Drive a ReAct review of the task's diff; return an approved/feedback verdict."""
        # The brief and Git diff are authoritative. Writer prose is optional context.
        message = json.dumps(
            {
                "request_kind": "task_review",
                "task_contract": _json_or_text(task),
                "writer_artifact": _json_or_text(result),
            },
            ensure_ascii=False,
        )
        # run the loop; a query error surfaces as a rejection (don't crash the run).
        text = ""
        async for event in query(
            client=self.client,
            registry=self.registry,
            system_prompt=REVIEWER_PROMPT,
            user_message=message,
            cwd=self.cwd,
            max_turns=self.max_turns,
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                return Review(
                    approved=False,
                    feedback=f"reviewer error: {event['error']}",
                    handoff_failed=True,
                )
        output_error = _review_output_error(text)
        if output_error:
            repaired = await repair_handoff_output(
                self.client,
                producer="Reviewer",
                invalid_output=text,
                validation_error=output_error,
                contract=(
                    "One ReviewerOutput JSON object with verdict approve|needs_fix, "
                    "blocking_findings, non_blocking_findings, and summary."
                ),
                context=json.dumps(
                    {
                        "request_kind": "repair_task_review_handoff",
                        "task_contract": _json_or_text(task),
                        "previous_review": text,
                    },
                    ensure_ascii=False,
                ),
                validate=_review_output_error,
            )
            if repaired.error:
                return Review(
                    approved=False,
                    feedback=repaired.error,
                    handoff_failed=True,
                )
            text = repaired.text
        output = parse_json_model(text, ReviewerOutput)
        approved = output.verdict == "approve" and not output.blocking_findings
        blocking_findings = [
            finding.model_copy(
                update={"id": finding.id or f"F{index}"}
            )
            for index, finding in enumerate(output.blocking_findings, start=1)
        ]
        feedback = "\n".join(
            f"{finding.id} — {finding.location}: {finding.description}; "
            f"fix: {finding.required_fix}"
            for finding in blocking_findings
        )
        advisory = "\n".join(
            f"non-blocking — {finding.location}: {finding.description}"
            for finding in output.non_blocking_findings
        )
        return Review(
            approved=approved,
            feedback=feedback or "\n".join(
                item for item in (output.summary, advisory) if item
            ),
            structured_findings=[
                finding.model_dump(mode="json") for finding in blocking_findings
            ],
        )


def _review_output_error(text: str) -> str:
    try:
        output = parse_json_model(text, ReviewerOutput)
    except ValueError as exc:
        return str(exc)
    if output.verdict == "approve" and output.blocking_findings:
        return "approve contradicts non-empty blocking_findings"
    if output.verdict == "needs_fix" and not output.blocking_findings:
        return "needs_fix requires at least one blocking finding"
    return ""


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"unstructured_legacy_text": value}
