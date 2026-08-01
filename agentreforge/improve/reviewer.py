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

import re
from typing import Any

from pydantic import BaseModel, Field

from ..agent.query import query
from ..agent.reviewer import Review
from ..llm.parse import parse_json_model
from ..observability import traceable
from ..orchestration.handoff import repair_handoff_output
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
4. Code quality — readable, consistent with the codebase, no obvious bugs, no
   unrelated edits, no hardcoded values.
5. Targeted verification — run focused unit or function-level checks over the important
   changed behavior. Do not duplicate the Deliverer's end-to-end startup/goal run, and
   do not turn generated-code quality into a component unit-test requirement.
6. Declared safety properties — when the Task requires `path_confinement`, actually
   run a relative traversal attempt such as `..` and confirm it is rejected. A
   read-only operation is not automatically confined, and comments/docstrings are not
   runtime evidence.

How to check: read the changed files; confirm they import / compile; run targeted
or smoke commands with the bash tool (e.g. `python3 -c "..."`) to see it really
works. You may run throwaway smoke commands but MUST NOT modify product code.
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
    {"severity": "major", "location": "file:line", "description": "real defect",
     "evidence": "observed code/runtime fact", "required_fix": "concrete fix"}
  ],
  "non_blocking_findings": [],
  "summary": "concise assessment"
}
Only real, evidenced defects belong in blocking_findings. Naming, style, optional
documentation, and speculative edge cases are non-blocking."""


class ReviewerOutput(BaseModel):
    verdict: str
    blocking_findings: list[Finding] = Field(default_factory=list)
    non_blocking_findings: list[Finding] = Field(default_factory=list)
    summary: str = ""


def review_registry(full: ToolRegistry) -> ToolRegistry:
    """A read-only-plus-bash registry: the Reviewer can inspect and run, not edit."""
    allowed = {"read_file", "list_dir", "glob", "grep", "search_code", "bash"}
    scoped = ToolRegistry()
    scoped.register_all(
        [t for name in full.list_names() if name in allowed and (t := full.get(name))]
    )
    return scoped


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
        # scope to read + bash so the Reviewer can run checks but not edit code.
        self.registry = review_registry(registry)
        self.cwd = cwd
        self.max_turns = max_turns

    @traceable(name="reviewer.review", run_type="chain")
    async def review(self, task: str, result: str) -> Review:
        """Drive a ReAct review of the task's diff; return an approved/feedback verdict."""
        # The brief and Git diff are authoritative. Writer prose is optional context.
        message = (
            f"What this task should do:\n{task}\n\n"
            f"Authoritative task change:\n{result}"
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
                context=f"Task:\n{task}\n\nPrevious review:\n{text}",
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
        feedback = "\n".join(
            f"{finding.location}: {finding.description}; fix: {finding.required_fix}"
            for finding in output.blocking_findings
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
        )


def _review_output_error(text: str) -> str:
    try:
        output = parse_json_model(text, ReviewerOutput)
    except ValueError as exc:
        return str(exc)
    if output.verdict not in {"approve", "needs_fix"}:
        return "ReviewerOutput verdict must be approve or needs_fix"
    if output.verdict == "approve" and output.blocking_findings:
        return "approve contradicts non-empty blocking_findings"
    if output.verdict == "needs_fix" and not output.blocking_findings:
        return "needs_fix requires at least one blocking finding"
    return ""


def _parse(text: str, task: str = "") -> Review:
    """Approve an explicit verdict or an unambiguous all-clause PASS checklist."""
    cleaned = (text or "").strip()
    upper = cleaned.upper()
    required_ids = _required_review_clause_ids(task)
    mentioned_ids = set(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", cleaned))
    missing_ids = [clause_id for clause_id in required_ids if clause_id not in mentioned_ids]
    explicit_approval = "APPROVED" in upper and "REJECTED" not in upper
    passed_ids = {
        clause_id
        for clause_id in required_ids
        if re.search(rf"\[{re.escape(clause_id)}\]\s+PASS\b", cleaned, re.IGNORECASE)
    }
    has_failure = "REJECTED" in upper or bool(
        re.search(r"\[[A-Za-z0-9_.:-]+\]\s+FAIL\b", cleaned, re.IGNORECASE)
    )
    checklist_approval = (
        bool(required_ids)
        and len(passed_ids) == len(required_ids)
        and not missing_ids
        and not has_failure
    )
    approved = explicit_approval or checklist_approval
    if explicit_approval and missing_ids:
        return Review(
            approved=False,
            feedback=(
                "Reviewer approval was incomplete: no explicit PASS/FAIL evidence for "
                f"contract clauses {', '.join(missing_ids)}. Re-review every clause."
            ),
        )
    if approved:
        return Review(approved=True)
    lines = cleaned.splitlines()
    feedback = "\n".join(lines[1:]).strip() or cleaned or "Rejected without specific feedback."
    return Review(approved=False, feedback=feedback)


def _required_review_clause_ids(task: str) -> list[str]:
    marker = "Required review clause ids:"
    for line in task.splitlines():
        if line.startswith(marker):
            return [
                clause_id.strip()
                for clause_id in line.removeprefix(marker).split(",")
                if clause_id.strip()
            ]
    return []
