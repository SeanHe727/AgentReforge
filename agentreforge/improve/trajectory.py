"""Target-agent trajectory persistence as structured JSONL.

The improvement pipeline needs "how the agent actually behaved" as evidence
(tool calls, errors, token use). Our event stream is otherwise ephemeral, so
`log_trajectory` tees each event to a per-project JSONL trace file while passing
it through to the normal consumer. The Orchestrator later reads these traces to
ground proposals in concrete evidence.

This module records the TARGET AGENT, not AgentReforge's own workflow.  The
separate AgentReforge run/loop audit lives in ``improve.records``.

Records the original target task, observable tool arguments/results, final
response, errors, and usage. Stored under ~/.agentreforge/traces/<project_key>/,
isolated per project — reusing snapshot's project key so both agree on "which project".
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..snapshot.service import _project_key

_MAX_CONTENT = 2000  # truncate tool output in traces to keep them small
_MAX_ARGUMENT = 2000
_SENSITIVE_KEYS = {"api_key", "authorization", "password", "secret", "token"}


def _trace_dir(cwd: str | Path, store_root: str | Path | None = None) -> Path:
    root = Path(store_root or Path.home() / ".agentreforge" / "traces")
    return root / _project_key(Path(cwd).resolve())


async def log_trajectory(
    events: AsyncIterator[dict[str, Any]],
    *,
    cwd: str,
    session_id: str | None = None,
    task_prompt: str = "",
    store_root: str | Path | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Backward-compatible alias for logging one TARGET AGENT run."""

    async for event in log_target_trajectory(
        events,
        cwd=cwd,
        session_id=session_id,
        task_prompt=task_prompt,
        store_root=store_root,
    ):
        yield event


async def log_target_trajectory(
    events: AsyncIterator[dict[str, Any]],
    *,
    cwd: str,
    session_id: str | None = None,
    task_prompt: str = "",
    store_root: str | Path | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Tee one target-agent event stream into its own evidence record."""

    session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = _trace_dir(cwd, store_root) / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        started = {
            "ts": datetime.now(UTC).isoformat(),
            "event_id": f"{session_id}:event:0",
            "trajectory_kind": "target_agent",
            "run_id": session_id,
            "session_id": session_id,
            "type": "target_run_started",
            "task_prompt": task_prompt,
        }
        f.write(json.dumps(started, ensure_ascii=False) + "\n")
        event_index = 1
        async for event in events:
            record = _record(event, session_id, event_index)
            event_index += 1
            if record is not None:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            yield event  # pass through to the original consumer


def _record(
    event: dict[str, Any], session_id: str, event_index: int = 0
) -> dict[str, Any] | None:
    etype = event.get("type")
    base = {
        "ts": datetime.now(UTC).isoformat(),
        "event_id": f"{session_id}:event:{event_index}",
        "trajectory_kind": "target_agent",
        "run_id": session_id,
        "session_id": session_id,
        "type": etype,
    }
    if etype == "tool_result":
        content = str(event.get("content") or "")
        return base | {
            "name": event.get("name"),
            "arguments": _safe_arguments(event.get("arguments")),
            "is_error": bool(event.get("is_error")),
            "content": content[:_MAX_CONTENT],
        }
    if etype == "usage":
        return base | {"usage": event.get("usage")}
    if etype == "done":
        return base | {
            "turns": event.get("turns"),
            "total_tokens": event.get("total_tokens"),
            "outcome": "completed",
            "final_response": _final_response(event)[:_MAX_CONTENT],
        }
    if etype == "error":
        return base | {"error": str(event.get("error"))}
    return None  # skip text_delta and anything without evidence value


def _safe_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if any(secret in str(key).lower() for secret in _SENSITIVE_KEYS):
            safe[str(key)] = "[REDACTED]"
            continue
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        safe[str(key)] = text[:_MAX_ARGUMENT]
    return safe


def _final_response(event: dict[str, Any]) -> str:
    messages = event.get("messages") or []
    for message in reversed(messages):
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "assistant" and content:
            return str(content)
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def load_trajectory(
    cwd: str, session_id: str, store_root: str | Path | None = None
) -> list[dict[str, Any]]:
    path = _trace_dir(cwd, store_root) / f"{session_id}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def list_trajectories(cwd: str, store_root: str | Path | None = None) -> list[str]:
    d = _trace_dir(cwd, store_root)
    return sorted((p.stem for p in d.glob("*.jsonl")), reverse=True) if d.exists() else []


def load_recent_trajectory(
    cwd: str, *, sessions: int = 3, max_records: int = 200, store_root: str | Path | None = None
) -> list[dict[str, Any]]:
    """Aggregate the evidence events from the most recent `sessions` runs.

    This is what the Analyzer gets grounded on: 'how the agent recently behaved',
    newest sessions first, capped so a long history can't flood the prompt.
    """
    # newest sessions first, then their events oldest-to-newest within each.
    out: list[dict[str, Any]] = []
    for session_id in list_trajectories(cwd, store_root)[:sessions]:
        out.extend(load_trajectory(cwd, session_id, store_root))
        if len(out) >= max_records:
            break
    return out[:max_records]


# Explicit names for new call sites. Legacy names remain available for callers.
load_target_trajectory = load_trajectory
load_recent_target_trajectory = load_recent_trajectory
