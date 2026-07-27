from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    tool_use_id: str | None = None


@dataclass
class ToolContext:
    cwd: str
    # optional capabilities handed to tools; grows over phases. Typed loosely
    # to avoid import cycles.
    memory: Any = None
    code_index: Any = None
    # HITL: called before running a non-read-only tool; returns True to approve.
    approval_callback: Any = None


# A handler is an async function taking (args, context) and returning a ToolResult.
ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    handler: ToolHandler
    # read-only tools have no side effects: safe to run in parallel and don't
    # need approval. write/exec tools set this False (used by concurrency & safety).
    is_read_only: bool = True

    def definition(self) -> dict[str, Any]:
        # The "outbound" schema we send to the LLM so it knows this tool exists.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def object_schema(
    properties: dict[str, dict[str, Any]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    # Build a JSON-Schema object for tool parameters. The "type: object" +
    # properties + required shape is fixed; only properties/required vary.
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }
