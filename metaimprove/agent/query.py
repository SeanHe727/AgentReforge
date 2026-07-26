"""
Full ReAct Loop:
    1. extract the tool calls from events
    2. batch the tool calls & execute
    3. return all the results
    4. next possible loop
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..llm.base import LlmClient
from ..observability import traceable
from ..tools.base import ToolContext
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from ..types import Message

# never log these to the tracer — `client` holds the API key; the rest are noise.
_TRACE_DROP = {"client", "registry", "approval_callback", "memory", "code_index", "history"}


def _redact_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in inputs.items() if k not in _TRACE_DROP}


@traceable(name="agent.query", run_type="chain", process_inputs=_redact_inputs)
async def query(
    *,
    client: LlmClient,
    registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    cwd: str,
    history: list[Message] | None = None,
    memory: Any = None,
    code_index: Any = None,
    approval_callback: Any = None,
    max_turns: int = 10,
) -> AsyncIterator[dict[str, Any]]:
    messages: list[Message] = [*(history or []), Message(role="user", content=user_message)]
    tool_definitions = registry.definitions()
    executor = ToolExecutor(registry)
    # bundle capabilities (incl. HITL approval) for tools to use.
    context = ToolContext(
        cwd=cwd, memory=memory, code_index=code_index, approval_callback=approval_callback
    )

    total_tokens = 0
    turn = 0
    while turn < max_turns:  # for each turn
        turn += 1
        text = ""
        tool_states: dict[int, dict[str, Any]] = {}

        # for each event from one LLM call
        async for event in client.chat(messages, tool_definitions, system_prompt=system_prompt):
            etype = event.get("type")
            if etype == "text_delta":
                text += event["text"]
                yield event
            elif etype == "tool_call_delta":
                # accumulate streamed fragments into full tool calls
                _merge_tool_delta(tool_states, event["tool_call"])
            elif etype == "usage":
                total_tokens += event["usage"]["input_tokens"] + event["usage"]["output_tokens"]
                yield event
            elif etype == "error":
                yield event
                return

        # tool-callings summarize
        tool_calls = _finalize_tool_calls(tool_states)

        # record this assistant turn (with any proposed tool calls) into history
        messages.append(Message(role="assistant", content=text, tool_calls=tool_calls))

        # no tool calls -> the model gave its final answer
        if not tool_calls:
            break

        # actual tool-calling execute, then feed each result back
        results = await executor.execute_all(tool_calls, context)
        for result in results:
            yield {
                "type": "tool_result",
                "name": _name_by_id(tool_calls, result.tool_use_id or ""),
                "content": result.content,
                "is_error": result.is_error,
            }
            messages.append(
                Message(role="tool", content=result.content, tool_call_id=result.tool_use_id)
            )
        # loop back: ask the model again, now that it can see the results

    yield {"type": "done", "turns": turn, "total_tokens": total_tokens, "messages": messages}


def _merge_tool_delta(states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    # Accumulate streamed tool-call fragments by their index into full calls.
    index = int(delta.get("index") or 0)
    state = states.setdefault(
        index,
        {
            "id": delta.get("id") or f"tool_{index}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        state["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        state["function"]["name"] = function["name"]
    if function.get("arguments"):
        state["function"]["arguments"] += function["arguments"]


def _finalize_tool_calls(states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    # Return complete calls (those that got a name), ordered by index.
    return [states[i] for i in sorted(states) if states[i]["function"]["name"]]


def _name_by_id(calls: list[dict[str, Any]], tool_use_id: str) -> str:
    for call in calls:
        if call.get("id") == tool_use_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"
