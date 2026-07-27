"""Safety policy: command guard + path guard + audit log (defense in depth).

guard_tool_call is the single entry point the executor calls before running a
tool: it dispatches to the command guard (for shell commands) and the path guard
(for file paths) and returns a denial reason, or None if the call is allowed.
"""

from __future__ import annotations

from typing import Any

from .audit_log import AuditLog
from .command_guard import check_command
from .path_guard import check_path

__all__ = ["AuditLog", "guard_tool_call"]


def guard_tool_call(name: str, args: dict[str, Any], cwd: str) -> str | None:
    # hard-rule checks; returns a denial reason, or None if allowed.
    command = args.get("command")
    if isinstance(command, str):
        reason = check_command(command)
        if reason:
            return reason
    path = args.get("path")
    if isinstance(path, str):
        reason = check_path(path, cwd)
        if reason:
            return reason
    return None
