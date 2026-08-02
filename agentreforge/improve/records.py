"""Durable records for target-agent evidence and AgentReforge execution.

There are deliberately two different histories:

* ``TargetRunRecord`` describes what the TARGET AGENT (for example, a coder)
  was asked to do and what observable actions/results it produced.  The
  Orchestrator uses this as diagnostic evidence.
* ``ReforgeLoopRecord`` / ``RecursiveRunRecord`` describe what AgentReforge
  itself planned, wrote, reviewed, delivered, and committed.  They are the
  audit trail for the improvement workflow and the feedback for later loops.

Keeping the contracts separate prevents an AgentReforge delivery failure from
being mistaken for a target-agent capability failure (and vice versa).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TargetRunSummary(BaseModel):
    """A compact, deterministic view of one observed target-agent run."""

    run_id: str
    task_prompt: str = ""
    outcome: str = "unknown"
    final_response: str = ""
    tool_calls: int = 0
    tool_errors: int = 0
    tools_used: list[str] = Field(default_factory=list)
    error_messages: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_source: str = "external"
    target_commit: str = ""
    is_current: bool = False


class ComponentRecord(BaseModel):
    """One AgentReforge component's observable output within a loop."""

    component: Literal[
        "orchestrator",
        "writer",
        "reviewer",
        # Kept only so historical records remain readable. New records store
        # execution evidence inside the Deliverer component.
        "acceptance_runner",
        "deliverer",
        "delivery_coordinator",
        "policy",
    ]
    status: str
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class CapabilityCompletionScope(BaseModel):
    """The narrow capability claim one delivered Loop is allowed to close."""

    candidate: str
    capability_gap: str = ""
    mechanism: str = ""
    expected_capability_delta: str = ""
    evidence_scope: list[str] = Field(default_factory=list)
    # Delivery demonstrates the frozen behavior in its scenarios.  It does not
    # automatically establish a causal pre/post improvement delta.
    verification_level: Literal["implemented", "behavior_verified", "delta_demonstrated"] = (
        "implemented"
    )


class ReforgeLoopRecord(BaseModel):
    """The materialized audit record for one AgentReforge improvement loop."""

    record_kind: Literal["reforge_loop"] = "reforge_loop"
    run_id: str
    loop_id: str
    loop: int
    base_commit: str
    stage: str
    components: list[ComponentRecord] = Field(default_factory=list)
    proposal_summary: str = ""
    diagnosis: dict[str, Any] = Field(default_factory=dict)
    changed_paths: list[str] = Field(default_factory=list)
    diff_ref: str = ""
    commit: str = ""
    completed: bool = False
    achievements: list[str] = Field(default_factory=list)
    completion_scopes: list[CapabilityCompletionScope] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)
    failure_kind: str = "none"
    attempt_fingerprint: str = ""
    error: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ReforgeLoopSummary(BaseModel):
    """Bounded next-loop context; the full component record remains drill-down data."""

    loop_id: str
    loop: int
    stage: str
    proposal_summary: str = ""
    capability_gaps: list[str] = Field(default_factory=list)
    selected_candidates: list[str] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)
    component_status: dict[str, str] = Field(default_factory=dict)
    commit: str = ""
    completed: bool = False
    achievements: list[str] = Field(default_factory=list)
    completion_scopes: list[CapabilityCompletionScope] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)
    failure_kind: str = "none"
    attempt_fingerprint: str = ""
    error: str = ""


class RecursiveRunRecord(BaseModel):
    """Stable top-level identity for one recursive run; branch is metadata."""

    record_kind: Literal["reforge_recursive_run"] = "reforge_recursive_run"
    run_id: str
    intent: str
    target_repo: str
    branch: str
    base_commit: str
    manifest: dict[str, Any] = Field(default_factory=dict)
    target_run_refs: list[str] = Field(default_factory=list)
    loop_refs: list[str] = Field(default_factory=list)
    status: str = "running"
    final_commit: str = ""
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str = ""


def jsonable(value: Any) -> Any:
    """Convert Pydantic/dataclass component outputs into JSON-safe structures."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ImprovementRecordStore:
    """Filesystem-backed source of truth for AgentReforge run/loop records."""

    def __init__(self, repo_root: str | Path, run_id: str):
        self.root = Path(repo_root) / ".agentreforge" / "records" / run_id
        self.run_path = self.root / "run.json"
        self.loops_dir = self.root / "loops"

    def start(self, record: RecursiveRunRecord) -> None:
        self.loops_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.run_path, record.model_dump(mode="json"))

    def append_loop(self, record: ReforgeLoopRecord, diff: str = "") -> ReforgeLoopRecord:
        self.loops_dir.mkdir(parents=True, exist_ok=True)
        loop_dir = self.loops_dir / f"loop_{record.loop}"
        loop_dir.mkdir(parents=True, exist_ok=True)
        if diff:
            diff_path = loop_dir / "diff.patch"
            diff_path.write_text(diff, encoding="utf-8")
            record.diff_ref = str(diff_path.relative_to(self.root))
        record_path = loop_dir / "record.json"
        self._write_json(record_path, record.model_dump(mode="json"))
        return record

    def finish(self, record: RecursiveRunRecord) -> None:
        record.completed_at = datetime.now(UTC).isoformat()
        self._write_json(self.run_path, record.model_dump(mode="json"))

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
