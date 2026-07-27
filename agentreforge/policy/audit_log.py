"""Audit log: append-only JSONL record of dangerous tool operations.

Every mutating/blocked/denied operation is written as one JSON line to
~/.agentreforge/audit.jsonl. JSONL (one JSON object per line) is the industry-standard
format for append-only logs: cheap to append, easy to grep/stream/parse.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path.home() / ".agentreforge" / "audit.jsonl"


class AuditLog:
    def __init__(self, path: str | Path = _DEFAULT_PATH):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, tool_name: str, args: dict[str, Any], outcome: str, cwd: str) -> None:
        # outcome: "blocked" | "denied" | "executed" | "error"
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "tool": tool_name,
            "args": args,
            "outcome": outcome,
            "cwd": cwd,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
