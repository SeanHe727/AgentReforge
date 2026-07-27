"""Shared helper: run a non-streaming LLM call and collect its full text.

Both the Planner and the Reviewer just want the model's complete text answer
(no tools, no streaming display). This consumes the event stream, accumulates
text_delta into one string, and raises on error. Extracted to avoid duplicating
the same loop in every "one-shot text" caller (DRY).
"""

from __future__ import annotations

from ..types import Message
from .base import LlmClient


async def collect_text(client: LlmClient, messages: list[Message], *, system_prompt: str) -> str:
    text = ""
    async for event in client.chat(messages, system_prompt=system_prompt):
        if event.get("type") == "text_delta":
            text += str(event.get("text") or "")
        elif event.get("type") == "error":
            raise event["error"]
    return text
