"""A deliberately incomplete rule-based command agent."""

from __future__ import annotations


class MiniAgent:
    """Dispatch a one-line text command and return a string result."""

    def handle(self, command: str) -> str:
        parts = command.strip().split()
        if not parts:
            return ""
        name, args = parts[0], parts[1:]

        if name == "echo":
            return " ".join(args)
        if name == "add":
            return str(int(args[0]) + int(args[1]))
        if name == "reverse":
            return " ".join(args)[::-1]
        return f"unknown command: {name}"
