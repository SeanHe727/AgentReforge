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

from typing import Any

from ..agent.query import query
from ..agent.reviewer import Review
from ..observability import traceable
from ..tools.registry import ToolRegistry

REVIEWER_PROMPT = """You are an agentic code Reviewer, like a careful senior engineer.
A Writer just implemented ONE task. Review ONLY this task's change, by actually
checking (not guessing), from these perspectives — each applies to every task, so
a well-done task passes them all:

1. Inputs & outputs — expected inputs handled, outputs correct, edge cases (bad or
   empty input) handled sensibly.
2. Dataflow & schema — data flows correctly through the code; types/structures are
   consistent and used as intended.
3. Goal realized — the code genuinely accomplishes what THIS task should do (not a
   hardcoded/faked shortcut that only appears to work).
4. Code quality — readable, consistent with the codebase, no obvious bugs, no
   out-of-scope edits, no hardcoded values.

How to check: read the changed files; confirm they import / compile; run targeted
or smoke commands with the bash tool (e.g. `python3 -c "..."`) to see it really
works. You may run throwaway smoke commands but MUST NOT modify product code.

Judge by IMPACT, not perfection. REJECT only for a REAL defect: a requirement in
"what this task should do" is not met, or there is a genuine bug (crash, deadlock,
wrong result, security issue, or brittle logic that breaks in NORMAL use). If the core
requirements are correctly met and only minor, non-blocking nits remain (contrived
edge cases, naming, style, nice-to-haves), APPROVE — you may mention the nits, but do
NOT block on them. Do not invent new requirements beyond the task.

When done, reply on the FIRST line with exactly APPROVED or REJECTED. If REJECTED,
add a second line with concrete, actionable feedback for the Writer to fix."""


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
        # the brief is what THIS task should do + the Writer's change to review.
        message = (
            f"What this task should do:\n{task}\n\n"
            f"Writer output + diff (this task only):\n{result}"
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
                return Review(approved=False, feedback=f"reviewer error: {event['error']}")
        return _parse(text)


def _parse(text: str) -> Review:
    """Only an explicit APPROVED (without REJECTED) approves; else it's a rejection."""
    cleaned = (text or "").strip()
    upper = cleaned.upper()
    if "APPROVED" in upper and "REJECTED" not in upper:
        return Review(approved=True)
    lines = cleaned.splitlines()
    feedback = "\n".join(lines[1:]).strip() or cleaned or "Rejected without specific feedback."
    return Review(approved=False, feedback=feedback)
