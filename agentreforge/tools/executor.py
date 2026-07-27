"""
Execute the tool-calls handler one by one, with defense-in-depth safety:
  1. guard (hard rules): command blacklist + path sandbox -> block outright
  2. HITL (soft): non-read-only tools need user approval
  3. execute, and audit-log every dangerous outcome
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..policy import AuditLog, guard_tool_call
from ..tools.base import ToolContext, ToolResult
from ..tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.audit = AuditLog()

    async def execute_all(
        self, tool_calls: list[dict[str, Any]], context: ToolContext
    ) -> list[ToolResult]:
        # execute the tools one by one in the tool_calls
        results: list[ToolResult] = []
        for call in tool_calls:
            results.append(await self._execute_single(call, context))
        return results

    async def _execute_single(self, call: dict[str, Any], context: ToolContext) -> ToolResult:
        tool_call_id = str(call.get("id") or "")
        function = call.get("function") or {}
        name = str(function.get("name") or "")

        # find tool by name; unknown tool -> error result (not a raise).
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(
                content=f"Error: tool '{name}' not found. "
                f"Available: {', '.join(self.registry.list_names())}",
                is_error=True,
                tool_use_id=tool_call_id,
            )

        # parse arguments (JSON string -> dict).
        args = _parse_arguments(function.get("arguments"))

        # 1. GUARD (hard rule): block obviously-dangerous commands / out-of-sandbox
        #    paths outright — the user never even gets asked.
        denied = guard_tool_call(name, args, context.cwd)
        if denied:
            self.audit.record(tool_name=name, args=args, outcome="blocked", cwd=context.cwd)
            return ToolResult(
                content=f"Denied by safety policy: {denied}",
                is_error=True,
                tool_use_id=tool_call_id,
            )

        # 2. HITL (soft): a non-read-only (write/dangerous) tool must be approved.
        #    read-only tools skip this. If denied, don't run it; tell the model.
        if not tool.is_read_only and context.approval_callback is not None:
            approved = context.approval_callback({"name": name, "args": args})
            if asyncio.iscoroutine(approved):
                approved = await approved
            if not approved:
                self.audit.record(tool_name=name, args=args, outcome="denied", cwd=context.cwd)
                return ToolResult(
                    content=f"Tool '{name}' was denied by the user; it was not run.",
                    is_error=True,
                    tool_use_id=tool_call_id,
                )

        # 3. run the handler; any exception becomes an error result so the loop
        #    (and the model) can see it instead of crashing.
        try:
            result = await tool.handler(args, context)
        except Exception as exc:  # noqa: BLE001 - tool errors must flow back to the model
            if not tool.is_read_only:
                self.audit.record(tool_name=name, args=args, outcome="error", cwd=context.cwd)
            return ToolResult(
                content=f"Error: tool '{name}' failed: {exc}",
                is_error=True,
                tool_use_id=tool_call_id,
            )

        # audit only mutating tools (read-only ops are safe and noisy to log).
        if not tool.is_read_only:
            self.audit.record(tool_name=name, args=args, outcome="executed", cwd=context.cwd)
        result.tool_use_id = tool_call_id
        return result


def _parse_arguments(raw: Any) -> dict[str, Any]:
    # parse the arguments from a JSON string into a dict, defensively.
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
