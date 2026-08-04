from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..llm.base import LlmClient
from ..llm.collect import collect_text
from ..llm.parse import parse_json_model
from ..orchestration.handoff import repair_handoff_output
from ..types import Message

REVIEWER_PROMPT = """You are a strict reviewer on an agent team.
Given a task and a worker's result, decide if the result correctly and fully
satisfies the task.
Return ONLY one JSON object:
{"verdict": "approve|reject", "feedback": [str], "summary": str}"""


class ReviewerDecision(BaseModel):
    verdict: Literal["approve", "reject"]
    feedback: list[str] = Field(default_factory=list)
    summary: str = ""


@dataclass
class Review:
    approved: bool
    feedback: str = ""
    handoff_failed: bool = False
    structured_findings: list[dict[str, Any]] = field(default_factory=list)


class Reviewer:
    def __init__(self, client: LlmClient):
        self.client = client

    async def review(self, task: str, result: str) -> Review:
        # ask the LLM to judge the result against the task (no tools), get text
        text = await collect_text(
            self.client,
            [
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "request_kind": "review",
                            "task": _json_value(task),
                            "worker_result": _json_value(result),
                        },
                        ensure_ascii=False,
                    ),
                )
            ],
            system_prompt=REVIEWER_PROMPT,
        )
        output_error = _review_output_error(text)
        if output_error:
            repaired = await repair_handoff_output(
                self.client,
                producer="Reviewer",
                invalid_output=text,
                validation_error=output_error,
                contract=(
                    "One ReviewerDecision JSON object with verdict approve|reject, "
                    "feedback:list[str], and summary:string."
                ),
                context=json.dumps(
                    {
                        "request_kind": "review",
                        "task": _json_value(task),
                        "worker_result": _json_value(result),
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
        return self._parse(text)

    def _parse(self, text: str) -> Review:
        output = parse_json_model(text, ReviewerDecision)
        if output.verdict == "approve":
            return Review(approved=True)
        feedback = "\n".join(output.feedback).strip() or output.summary
        return Review(
            approved=False,
            feedback=feedback or "Rejected without specific feedback.",
        )


def _review_output_error(text: str) -> str:
    try:
        output = parse_json_model(text, ReviewerDecision)
    except ValueError as exc:
        return str(exc)
    if output.verdict == "reject" and not (output.feedback or output.summary.strip()):
        return "reject requires concrete feedback"
    return ""


def _json_value(value: str) -> Any:
    """Keep nested handoffs as objects; preserve genuine prose as an attributed field."""

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {"unstructured_context": value}
