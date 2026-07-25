"""Built-in tools.

read_file / list_dir / glob / grep are read-only (no side effects). write_file
and bash (added later) mutate state and set is_read_only=False.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from .base import Tool, ToolContext, ToolResult, object_schema

# Cap how much we read so a huge file can't blow up the context window.
_MAX_CHARS = 30_000
_MAX_MATCHES = 200
_BASH_TIMEOUT = 30.0


async def _read_file(args: dict, context: ToolContext) -> ToolResult:
    # 1. get the path argument and resolve it against the working directory.
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return ToolResult(content="Error: 'path' is required.", is_error=True)
    path = (Path(context.cwd) / raw_path).resolve()

    # 2. read the file, turning expected failures into error results (not raises),
    #    so the model can see what went wrong and react.
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult(content=f"Error: file not found: {raw_path}", is_error=True)
    except IsADirectoryError:
        return ToolResult(content=f"Error: '{raw_path}' is a directory, not a file.", is_error=True)
    except UnicodeDecodeError:
        return ToolResult(content=f"Error: '{raw_path}' is not a UTF-8 text file.", is_error=True)
    except OSError as exc:
        return ToolResult(content=f"Error reading '{raw_path}': {exc}", is_error=True)

    # 3. truncate if too long, and return the content.
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n... [truncated]"
    return ToolResult(content=text)


async def _list_dir(args: dict, context: ToolContext) -> ToolResult:
    # Default to the working directory when no path is given.
    raw_path = str(args.get("path") or ".").strip() or "."
    path = (Path(context.cwd) / raw_path).resolve()

    if not path.exists():
        return ToolResult(content=f"Error: path not found: {raw_path}", is_error=True)
    if not path.is_dir():
        return ToolResult(content=f"Error: '{raw_path}' is not a directory.", is_error=True)

    # Directories first (trailing /), then files with their size. Sorted for stability.
    lines = []
    for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            size = entry.stat().st_size
            lines.append(f"{entry.name} ({size} bytes)")
    body = "\n".join(lines) if lines else "(empty directory)"
    return ToolResult(content=body)


async def _glob(args: dict, context: ToolContext) -> ToolResult:
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return ToolResult(content="Error: 'pattern' is required.", is_error=True)

    base = Path(context.cwd)
    # Path.glob understands ** for recursive matches, e.g. "**/*.py".
    matches = [str(p.relative_to(base)) for p in base.glob(pattern) if p.is_file()]
    matches.sort()
    if not matches:
        return ToolResult(content=f"No files match pattern: {pattern}")
    if len(matches) > _MAX_MATCHES:
        matches = matches[:_MAX_MATCHES] + ["... [truncated]"]
    return ToolResult(content="\n".join(matches))


async def _grep(args: dict, context: ToolContext) -> ToolResult:
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return ToolResult(content="Error: 'pattern' is required.", is_error=True)
    raw_path = str(args.get("path") or ".").strip() or "."

    # Compile the regex up front; a bad pattern is a user error, not a crash.
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(content=f"Error: invalid regex: {exc}", is_error=True)

    cwd = Path(context.cwd).resolve()
    base = (cwd / raw_path).resolve()
    # Search a single file, or every file under a directory.
    files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]

    results: list[str] = []
    for file in files:
        try:
            for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
                if regex.search(line):
                    rel = file.relative_to(cwd)
                    results.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(results) >= _MAX_MATCHES:
                        results.append("... [truncated]")
                        return ToolResult(content="\n".join(results))
        except (UnicodeDecodeError, OSError):
            # Skip binary or unreadable files instead of failing the whole search.
            continue

    if not results:
        return ToolResult(content=f"No matches for: {pattern}")
    return ToolResult(content="\n".join(results))


async def _write_file(args: dict, context: ToolContext) -> ToolResult:
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return ToolResult(content="Error: 'path' is required.", is_error=True)
    content = args.get("content")
    if not isinstance(content, str):
        return ToolResult(content="Error: 'content' must be a string.", is_error=True)

    path = (Path(context.cwd) / raw_path).resolve()
    try:
        # create parent directories if needed, then write.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return ToolResult(content=f"Error writing '{raw_path}': {exc}", is_error=True)
    return ToolResult(content=f"Wrote {len(content)} chars to {raw_path}")


async def _edit_file(args: dict, context: ToolContext) -> ToolResult:
    # Anchored replace: swap an EXACT substring for another. Because the match
    # must be found (and unique unless replace_all), the model structurally cannot
    # silently drop the rest of the file — the failure mode `write_file` allows.
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return ToolResult(content="Error: 'path' is required.", is_error=True)
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return ToolResult(
            content="Error: 'old_string' and 'new_string' must be strings.", is_error=True
        )
    if old == new:
        return ToolResult(
            content="Error: 'old_string' and 'new_string' are identical.", is_error=True
        )
    replace_all = bool(args.get("replace_all") or False)

    path = (Path(context.cwd) / raw_path).resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult(
            content=f"Error: file not found: {raw_path}. Use write_file to create a NEW file.",
            is_error=True,
        )
    except (IsADirectoryError, UnicodeDecodeError, OSError) as exc:
        return ToolResult(content=f"Error reading '{raw_path}': {exc}", is_error=True)

    count = text.count(old)
    if count == 0:
        return ToolResult(
            content=f"Error: old_string not found in {raw_path}. It must match the file "
            "EXACTLY (whitespace included). Read the file and copy the exact text.",
            is_error=True,
        )
    if count > 1 and not replace_all:
        return ToolResult(
            content=f"Error: old_string is not unique in {raw_path} ({count} occurrences). "
            "Add surrounding context to make it unique, or set replace_all=true.",
            is_error=True,
        )

    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return ToolResult(content=f"Error writing '{raw_path}': {exc}", is_error=True)
    where = f"{count} occurrences" if replace_all else "1 occurrence"
    return ToolResult(content=f"Edited {raw_path} ({where} replaced).")


async def _bash(args: dict, context: ToolContext) -> ToolResult:
    command = str(args.get("command") or "").strip()
    if not command:
        return ToolResult(content="Error: 'command' is required.", is_error=True)

    # Launch the command in a shell, in the working directory, capturing output.
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=context.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        # wait for it, but kill it if it runs too long (hung / interactive).
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_BASH_TIMEOUT)
    except TimeoutError:
        process.kill()
        return ToolResult(
            content=f"Error: command timed out after {_BASH_TIMEOUT:.0f}s: {command}",
            is_error=True,
        )

    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")
    exit_code = process.returncode

    # Assemble a readable report; non-zero exit code marks it as an error.
    parts = [f"$ {command}", f"exit code: {exit_code}"]
    if out.strip():
        parts.append(f"stdout:\n{out.strip()}")
    if err.strip():
        parts.append(f"stderr:\n{err.strip()}")
    body = "\n".join(parts)
    if len(body) > _MAX_CHARS:
        body = body[:_MAX_CHARS] + "\n... [truncated]"
    return ToolResult(content=body, is_error=exit_code != 0)


async def _search_code(args: dict, context: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolResult(content="Error: 'query' is required.", is_error=True)
    if context.code_index is None:
        return ToolResult(content="Error: code index is not available.", is_error=True)
    results = context.code_index.search(query)
    if not results:
        return ToolResult(content=f"No code matches for: {query}")
    body = "\n".join(f"{r.path}:{r.line}: {r.snippet}" for r in results)
    return ToolResult(content=body)


async def _save_memory(args: dict, context: ToolContext) -> ToolResult:
    content = str(args.get("content") or "").strip()
    if not content:
        return ToolResult(content="Error: 'content' is required.", is_error=True)
    if context.memory is None:
        return ToolResult(content="Error: long-term memory is not available.", is_error=True)
    memory_id = context.memory.save(content)
    return ToolResult(content=f"Saved memory #{memory_id}.")


# tool definition
read_file_tool = Tool(
    name="read_file",
    description=(
        "Read the contents of a text file at a path relative to the working "
        "directory. Use this to inspect files before answering questions about them."
    ),
    parameters=object_schema(
        {"path": {"type": "string", "description": "File path relative to the working directory"}},
        required=["path"],
    ),
    handler=_read_file,
)

list_dir_tool = Tool(
    name="list_dir",
    description="List the entries of a directory (relative to the working directory).",
    parameters=object_schema(
        {"path": {"type": "string", "description": "Directory path; defaults to '.'"}},
    ),
    handler=_list_dir,
)

glob_tool = Tool(
    name="glob",
    description="Find files matching a glob pattern, e.g. '**/*.py' or 'src/*.txt'.",
    parameters=object_schema(
        {"pattern": {"type": "string", "description": "Glob pattern, supports ** for recursion"}},
        required=["pattern"],
    ),
    handler=_glob,
)

grep_tool = Tool(
    name="grep",
    description=(
        "Search file contents for a regular expression. Returns matching lines as "
        "'path:line: text'. Searches a file or recursively under a directory."
    ),
    parameters=object_schema(
        {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
            "path": {
                "type": "string",
                "description": "File or directory to search; defaults to '.'",
            },
        },
        required=["pattern"],
    ),
    handler=_grep,
)


write_file_tool = Tool(
    name="write_file",
    description=(
        "Write text content to a file (relative to the working directory), "
        "creating parent directories and overwriting any existing file."
    ),
    parameters=object_schema(
        {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Full text content to write"},
        },
        required=["path", "content"],
    ),
    handler=_write_file,
    is_read_only=False,  # write operation: has side effects
)


edit_file_tool = Tool(
    name="edit_file",
    description=(
        "Edit an EXISTING file by replacing an exact substring (old_string) with "
        "new_string. PREFER this over write_file for changing existing files: it "
        "only touches the matched region and leaves the rest of the file intact. "
        "old_string must match the file exactly and be unique (or set replace_all). "
        "Use write_file only to create a brand-new file."
    ),
    parameters=object_schema(
        {
            "path": {"type": "string", "description": "File to edit"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence (default false: require a unique match)",
            },
        },
        required=["path", "old_string", "new_string"],
    ),
    handler=_edit_file,
    is_read_only=False,  # write operation: has side effects
)


bash_tool = Tool(
    name="bash",
    description=(
        "Run a shell command in the working directory and return its stdout, "
        "stderr, and exit code. Use for builds, tests, git, and quick inspection."
    ),
    parameters=object_schema(
        {"command": {"type": "string", "description": "The shell command to run"}},
        required=["command"],
    ),
    handler=_bash,
    is_read_only=False,  # can run anything: has side effects and is dangerous
)


save_memory_tool = Tool(
    name="save_memory",
    description=(
        "Save a durable fact to long-term memory (persists across sessions). "
        "Use for stable user preferences, project conventions, and key decisions."
    ),
    parameters=object_schema(
        {"content": {"type": "string", "description": "The fact to remember, one sentence"}},
        required=["content"],
    ),
    handler=_save_memory,
    is_read_only=False,  # writes to the memory store
)


search_code_tool = Tool(
    name="search_code",
    description=(
        "Search the project's code index by keyword. Returns matching lines as "
        "'path:line: snippet'. Faster than grep for repeated lookups across the repo."
    ),
    parameters=object_schema(
        {"query": {"type": "string", "description": "Keywords to search for in code"}},
        required=["query"],
    ),
    handler=_search_code,
)


def get_builtin_tools() -> list[Tool]:
    return [
        read_file_tool,
        list_dir_tool,
        glob_tool,
        grep_tool,
        write_file_tool,
        edit_file_tool,
        bash_tool,
        save_memory_tool,
        search_code_tool,
    ]
