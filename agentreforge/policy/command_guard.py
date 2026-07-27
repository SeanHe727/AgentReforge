"""Command guard: hard-deny obviously dangerous shell commands.

This is the "hard rule" layer of defense-in-depth: patterns matched here are
refused outright (the user never even gets a y/N prompt). Reasonable-but-risky
commands fall through and are handled by HITL approval instead.

The patterns are sensible hardcoded defaults; in production they'd typically be
a configurable allow/deny list.
"""

from __future__ import annotations

import re

# Each entry: (compiled regex, human reason). Kept conservative — only match
# things that are almost never legitimate.
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r"),
        "recursive force-delete (rm -rf)",
    ),
    (re.compile(r":\s*\(\s*\)\s*\{.*\|.*&\s*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\bsudo\b"), "privilege escalation (sudo)"),
    (re.compile(r"\bmkfs\b"), "filesystem format (mkfs)"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw write to a device (dd of=/dev/...)"),
    (re.compile(r">\s*/dev/(sd|nvme|disk)"), "write to a disk device"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/"), "recursive chmod 777 on root"),
    (
        re.compile(r"\b(curl|wget)\b.*\|\s*(sudo\s+)?(sh|bash|zsh)\b"),
        "pipe remote script into a shell",
    ),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "shutdown/reboot the machine"),
]


def check_command(command: str) -> str | None:
    # returns a denial reason if the command is dangerous, else None (allowed).
    text = command.strip()
    if not text:
        return None
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            return f"blocked dangerous command: {reason}"
    return None
