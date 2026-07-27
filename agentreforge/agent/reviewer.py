from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LlmClient
from ..llm.collect import collect_text
from ..types import Message

REVIEWER_PROMPT = """You are a strict reviewer on an agent team.
Given a task and a worker's result, decide if the result correctly and fully
satisfies the task.
Reply on the FIRST line with exactly APPROVED or REJECTED.
If REJECTED, add a second line with concrete, actionable feedback for a retry."""


@dataclass
class Review:
    approved: bool
    feedback: str = ""


class Reviewer:
    def __init__(self, client: LlmClient):
        self.client = client

    async def review(self, task: str, result: str) -> Review:
        # ask the LLM to judge the result against the task (no tools), get text
        text = await collect_text(
            self.client,
            [Message(role="user", content=f"Task:\n{task}\n\nWorker result:\n{result}")],
            system_prompt=REVIEWER_PROMPT,
        )
        return self._parse(text)

    def _parse(self, text: str) -> Review:
        # parse the verdict text into a Review. Conservative: only an explicit
        # APPROVED (without REJECTED) approves; anything else is a rejection.
        cleaned = (text or "").strip()
        upper = cleaned.upper()
        approved = "APPROVED" in upper and "REJECTED" not in upper
        if approved:
            return Review(approved=True)
        lines = cleaned.splitlines()
        feedback = "\n".join(lines[1:]).strip() or cleaned or "Rejected without specific feedback."
        return Review(approved=False, feedback=feedback)
