"""Subprocess adapter for the AgentReforge demo target.

This module runs in a fresh interpreter so it observes the candidate worktree's
``demo_agent`` package rather than AgentReforge's process state. It converts the
demo agent's internal tool calls into the normalized JSONL trajectory contract.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from .trajectory import tool_result_is_error

_CONTENT_CAP = 2_000
_ARGUMENT_CAP = 2_000
_SENSITIVE_KEYS = {"api_key", "authorization", "password", "secret", "token"}
_ACTION_TOOLS = {
    "write_file",
    "edit_file",
    "delete_file",
    "run_bash",
    "bash",
    "execute_command",
}


def _write_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _safe_arguments(value: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if any(secret in str(key).casefold() for secret in _SENSITIVE_KEYS):
            safe[str(key)] = "[REDACTED]"
            continue
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        safe[str(key)] = text[:_ARGUMENT_CAP]
    return safe


def _load_agent(target: Path):
    sys.path.insert(0, str(target))
    from demo_agent import agent  # noqa: PLC0415

    return agent


def run_scenario(
    *,
    target: Path,
    workspace: Path,
    prompt: str,
    trajectory_path: Path,
    max_turns: int = 20,
    max_actions: int = 8,
) -> int:
    agent = _load_agent(target)
    original_execute = agent.execute
    original_chat = getattr(agent, "chat", None)
    sequence = 0
    pending_calls: list[dict[str, Any]] = []
    turns_used = 0
    actions_used = 0
    prior_message_count = 0

    def observed_chat(messages: list[dict], tools: list[dict] | None = None, **kwargs):
        nonlocal prior_message_count, sequence, turns_used
        input_delta = messages[prior_message_count:]
        prior_message_count = len(messages)
        turns_used += 1
        message = original_chat(messages, tools=tools, **kwargs)
        sequence += 1
        calls = message.get("tool_calls") or []
        pending_calls.extend(call for call in calls if isinstance(call, dict))
        _write_event(
            trajectory_path,
            {
                "type": "agent_turn",
                "sequence": sequence,
                "actor": "target_agent",
                "status": "completed",
                "turn": turns_used,
                "input_messages": [_safe_message(item) for item in input_delta],
                "content": str(message.get("content") or "")[:_CONTENT_CAP],
                "tool_calls": [
                    {
                        "id": str(call.get("id") or ""),
                        "name": str((call.get("function") or {}).get("name") or ""),
                        "arguments": _safe_call_arguments(
                            (call.get("function") or {}).get("arguments")
                        ),
                    }
                    for call in calls
                    if isinstance(call, dict)
                ],
            },
        )
        return message

    def observed_execute(name: str, arguments: dict, cwd: str) -> str:
        nonlocal actions_used, sequence
        sequence += 1
        tool_call_id = ""
        for index, call in enumerate(pending_calls):
            if str((call.get("function") or {}).get("name") or "") == name:
                tool_call_id = str(call.get("id") or "")
                pending_calls.pop(index)
                break
        is_action = name in _ACTION_TOOLS
        if is_action and actions_used >= max_actions:
            result = (
                "error: target action budget exhausted "
                f"({actions_used}/{max_actions}); inspect existing evidence and finalize"
            )
            _write_event(
                trajectory_path,
                {
                    "type": "tool_result",
                    "sequence": sequence,
                    "actor": "tool",
                    "status": "error",
                    "name": name,
                    "tool_call_id": tool_call_id,
                    "arguments": _safe_arguments(arguments),
                    "is_error": True,
                    "budget_blocked": True,
                    "action_steps_used": actions_used,
                    "action_steps_limit": max_actions,
                    "content": result,
                },
            )
            return result
        if is_action:
            actions_used += 1
        try:
            result = original_execute(name, arguments, cwd)
        except Exception as exc:
            _write_event(
                trajectory_path,
                {
                    "type": "tool_result",
                    "sequence": sequence,
                    "actor": "tool",
                    "status": "error",
                    "name": name,
                    "tool_call_id": tool_call_id,
                    "arguments": _safe_arguments(arguments),
                    "is_error": True,
                    "action_step": actions_used if is_action else None,
                    "content": f"{type(exc).__name__}: {exc}"[:_CONTENT_CAP],
                },
            )
            raise
        _write_event(
            trajectory_path,
            {
                "type": "tool_result",
                "sequence": sequence,
                "actor": "tool",
                "status": "error" if tool_result_is_error(result) else "completed",
                "name": name,
                "tool_call_id": tool_call_id,
                "arguments": _safe_arguments(arguments),
                "is_error": tool_result_is_error(result),
                "action_step": actions_used if is_action else None,
                "content": result[:_CONTENT_CAP],
            },
        )
        return result

    if original_chat is not None:
        agent.chat = observed_chat
    agent.execute = observed_execute
    try:
        run_parameters = inspect.signature(agent.run_task).parameters
        if "max_steps" in run_parameters:
            final = agent.run_task(prompt, str(workspace), max_steps=max_turns)
        else:
            final = agent.run_task(prompt, str(workspace))
    except Exception as exc:
        _write_event(
            trajectory_path,
            {
                "type": "error",
                "sequence": sequence + 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if original_chat is not None:
            agent.chat = original_chat
        agent.execute = original_execute
    _write_event(
        trajectory_path,
        {
            "type": "done",
            "sequence": sequence + 1,
            "actor": "target_agent",
            "status": "completed",
            "outcome": (
                "incomplete"
                if final.strip().casefold()
                in {
                    "(stopped: reached max steps)",
                    "(stopped: maximum steps reached)",
                    "(stopped: step budget exhausted)",
                }
                else "completed"
            ),
            "reason_code": (
                "turn_budget_exhausted"
                if final.strip().casefold()
                in {
                    "(stopped: reached max steps)",
                    "(stopped: maximum steps reached)",
                    "(stopped: step budget exhausted)",
                }
                else "target_finished"
            ),
            "final_response": final,
            "budget": {
                "agent_turns_used": turns_used,
                "agent_turns_limit": max_turns,
                "action_steps_used": actions_used,
                "action_steps_limit": max_actions,
            },
        },
    )
    print(final)
    return 0


def _safe_call_arguments(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}") if isinstance(value, str) else value
    except json.JSONDecodeError:
        parsed = {"_raw": str(value)}
    return _safe_arguments(parsed if isinstance(parsed, dict) else {"_value": parsed})


def _safe_message(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"role": "unknown", "content": str(value)[:_CONTENT_CAP]}
    return {
        "role": str(value.get("role") or ""),
        "content": str(value.get("content") or "")[:_CONTENT_CAP],
        "tool_call_id": str(value.get("tool_call_id") or ""),
    }


def probe_path_confinement(
    *,
    target: Path,
    workspace: Path,
    outside_path: str,
    sentinel: str,
) -> int:
    agent = _load_agent(target)
    result = agent.execute("read_file", {"path": outside_path}, str(workspace))
    if sentinel in result:
        print("unsafe: traversal exposed the outside sentinel")
        return 1
    if result.startswith("error:"):
        print(f"safe: traversal blocked ({result})")
        return 0
    print(f"unsafe: traversal did not produce a blocking error ({result[:200]})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--probe-path-confinement", action="store_true")
    parser.add_argument("--outside-path")
    parser.add_argument("--sentinel")
    args = parser.parse_args()

    if args.probe_path_confinement:
        if not args.outside_path or not args.sentinel:
            parser.error("path-confinement probe requires outside-path and sentinel")
        return probe_path_confinement(
            target=args.target.resolve(),
            workspace=args.workspace.resolve(),
            outside_path=args.outside_path,
            sentinel=args.sentinel,
        )
    if args.prompt is None or args.trajectory is None:
        parser.error("scenario run requires prompt and trajectory")
    return run_scenario(
        target=args.target.resolve(),
        workspace=args.workspace.resolve(),
        prompt=args.prompt,
        trajectory_path=args.trajectory.resolve(),
        max_turns=args.max_turns,
        max_actions=args.max_actions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
