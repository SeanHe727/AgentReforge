"""Shared data types for agent-reforge.

Kept deliberately small for now. We add fields (multimodal content, etc.)
in later phases when the features that need them arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """One message in a conversation.

    role: "system" | "user" | "assistant" | "tool"
    content: the text of the message
    tool_calls: on an assistant message, the tool calls it proposed
    tool_call_id: on a tool message, which tool call this result answers
    """

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
