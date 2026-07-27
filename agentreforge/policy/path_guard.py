"""Path guard: sandbox file operations to the project directory.

Prevents an agent from reading/writing outside the working directory (e.g.
write_file("/etc/passwd") or read_file("../../secrets")). The boundary is
COMPUTED from the project root, not a hardcoded list.
"""

from __future__ import annotations

from pathlib import Path


def check_path(path: str, cwd: str) -> str | None:
    # returns a denial reason if `path` escapes the sandbox (cwd), else None.
    root = Path(cwd).resolve()
    # if `path` is absolute, (root / path) resolves to that absolute path;
    # if relative, it's resolved under root. Either way we then check it's inside.
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return f"blocked path outside the sandbox: '{path}' is not under {root}"
    return None
