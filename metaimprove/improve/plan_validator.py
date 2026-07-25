"""Deterministic task-DAG validation, exposed as a tool the Orchestrator calls.

Before emitting its proposal, the Orchestrator runs its task graph through
`validate_plan` (via the `validate_plan` tool). Invalid graphs — duplicate ids,
dangling dependencies, cycles, no runnable root — are reported back so the model
FIXES them and re-validates, instead of the pipeline discovering the mess later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..tools.base import Tool, ToolContext, ToolResult, object_schema


@dataclass
class PlanValidation:
    valid: bool
    duplicate_ids: list[str] = field(default_factory=list)
    unknown_dependencies: list[str] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    no_runnable_root: bool = False

    def summary(self) -> str:
        if self.valid:
            return "valid: unique ids, all dependencies exist, acyclic, has a runnable root."
        parts = []
        if self.duplicate_ids:
            parts.append(f"duplicate ids: {', '.join(self.duplicate_ids)}")
        if self.unknown_dependencies:
            parts.append(f"unknown dependencies: {', '.join(self.unknown_dependencies)}")
        if self.cycles:
            parts.append(f"cycles: {'; '.join(' -> '.join(c) for c in self.cycles)}")
        if self.no_runnable_root:
            parts.append("no runnable root (every task depends on another)")
        return "INVALID — " + "; ".join(parts)


def validate_plan(tasks: list[dict[str, Any]]) -> PlanValidation:
    """Check a task DAG: unique non-empty ids, existing deps, acyclic, has a root."""
    ids = [str(t.get("id", "")).strip() for t in tasks]

    # duplicate or empty ids.
    seen: set[str] = set()
    duplicates: list[str] = []
    for tid in ids:
        if tid == "" or (tid in seen and tid not in duplicates):
            duplicates.append(tid or "(empty)")
        seen.add(tid)

    # dependencies that don't reference an existing task.
    id_set = {tid for tid in ids if tid}
    deps: dict[str, list[str]] = {}
    unknown: list[str] = []
    for t in tasks:
        tid = str(t.get("id", "")).strip()
        d = [str(x).strip() for x in (t.get("dependencies") or [])]
        deps[tid] = d
        unknown.extend(dep for dep in d if dep not in id_set)

    # cycle detection (DFS over the dependency edges).
    cycles = _find_cycles(id_set, deps)

    # a runnable root = at least one task with no dependencies (when there are tasks).
    no_root = bool(id_set) and all(deps.get(tid) for tid in id_set)

    valid = not (duplicates or unknown or cycles or no_root)
    return PlanValidation(valid, duplicates, sorted(set(unknown)), cycles, no_root)


def _find_cycles(id_set: set[str], deps: dict[str, list[str]]) -> list[list[str]]:
    """Return any dependency cycles (each as the path of ids forming it)."""
    cycles: list[list[str]] = []
    visited: set[str] = set()

    def walk(node: str, stack: list[str]) -> None:
        if node in stack:
            cycles.append(stack[stack.index(node):] + [node])
            return
        if node in visited or node not in id_set:
            return
        stack.append(node)
        for dep in deps.get(node, []):
            walk(dep, stack)
        stack.pop()
        visited.add(node)

    for tid in id_set:
        walk(tid, [])
    # de-duplicate cycles that start at different nodes but are the same loop.
    unique: list[list[str]] = []
    seen_sets: list[set[str]] = []
    for c in cycles:
        s = set(c)
        if s not in seen_sets:
            seen_sets.append(s)
            unique.append(c)
    return unique


async def _validate_plan_handler(args: dict, context: ToolContext) -> ToolResult:
    # accept the tasks either as a list or a JSON string.
    raw = args.get("tasks")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolResult(content=f"Error: 'tasks' is not valid JSON: {exc}", is_error=True)
    if not isinstance(raw, list):
        return ToolResult(
            content="Error: 'tasks' must be a list of {id, dependencies}.", is_error=True
        )
    result = validate_plan(raw)
    return ToolResult(content=result.summary(), is_error=not result.valid)


validate_plan_tool = Tool(
    name="validate_plan",
    description=(
        "Validate a task DAG before finalizing a proposal. Pass 'tasks' as a list "
        "of {id, dependencies:[...]}. Returns whether it is valid (unique ids, all "
        "dependencies exist, acyclic, has a runnable root) or the specific problems "
        "to fix. Always call this on your tasks before emitting the proposal."
    ),
    parameters=object_schema(
        {"tasks": {"type": "array", "description": "List of {id, dependencies:[...]}"}},
        required=["tasks"],
    ),
    handler=_validate_plan_handler,
)
