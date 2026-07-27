"""Deterministic context preparation for the improvement Orchestrator.

The ContextBuilder is mandatory pipeline infrastructure, not an optional LLM
tool.  It gives the Orchestrator complete current-run facts while keeping raw
evidence addressable by stable references.  Historical retrieval/RAG can be
layered on top later without becoming the source of truth for current state.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .records import ReforgeLoopRecord, ReforgeLoopSummary, TargetRunSummary

_SKIP_DIRS = {
    ".git",
    ".meta-improve",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
_MANIFESTS = {
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "requirements.txt",
}
_ENTRYPOINT_NAMES = {
    "__main__.py",
    "cli.py",
    "main.py",
    "app.py",
    "index.ts",
    "index.js",
}
_MAX_REPO_FILES = 250
_MAX_RESPONSE = 2_000
_MAX_EVIDENCE_CONTENT = 800
_MAX_EVIDENCE_RECORDS = 80


class RepositoryContext(BaseModel):
    root: str
    manifests: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    test_paths: list[str] = Field(default_factory=list)
    top_level: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    language_counts: dict[str, int] = Field(default_factory=dict)


class OrchestratorContext(BaseModel):
    """Everything that must be present before Orchestrator reasoning starts."""

    improvement_intent: str
    target_agent_runs: list[TargetRunSummary] = Field(default_factory=list)
    # AgentReforge's own history; never mixed into target_agent_runs.
    previous_reforge_loops: list[ReforgeLoopSummary] = Field(default_factory=list)
    repository: RepositoryContext
    run_manifest: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[dict[str, Any]] = Field(default_factory=list)
    # Full target events remain available to a drill-down tool but are excluded
    # from the initial prompt, which receives only the bounded catalog above.
    raw_target_evidence: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        exclude=True,
    )
    raw_reforge_loops: dict[str, ReforgeLoopRecord] = Field(
        default_factory=dict,
        exclude=True,
    )


class OrchestratorContextBuilder:
    def __init__(self, cwd: str):
        self.cwd = Path(cwd).resolve()

    def build(
        self,
        *,
        intent: str,
        target_trajectory: list[dict[str, Any]],
        previous_reforge_loops: list[ReforgeLoopRecord],
        run_manifest: dict[str, Any],
    ) -> OrchestratorContext:
        summaries, evidence = summarize_target_trajectory(target_trajectory)
        raw_evidence = {}
        for index, record in enumerate(target_trajectory):
            run_id = str(record.get("run_id") or record.get("session_id") or "unknown")
            event_id = str(record.get("event_id") or f"{run_id}:event:{index}")
            raw_evidence[event_id] = record
        loop_summaries = [_summarize_reforge_loop(loop) for loop in previous_reforge_loops]
        return OrchestratorContext(
            improvement_intent=intent,
            target_agent_runs=summaries,
            previous_reforge_loops=loop_summaries,
            repository=self._repository_context(),
            run_manifest=run_manifest,
            evidence_catalog=evidence,
            raw_target_evidence=raw_evidence,
            raw_reforge_loops={loop.loop_id: loop for loop in previous_reforge_loops},
        )

    def _repository_context(self) -> RepositoryContext:
        files: list[str] = []
        languages: Counter[str] = Counter()
        for path in sorted(self.cwd.rglob("*")):
            if any(part in _SKIP_DIRS for part in path.relative_to(self.cwd).parts):
                continue
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.cwd))
            files.append(rel)
            if path.suffix:
                languages[path.suffix.lower()] += 1
            if len(files) >= _MAX_REPO_FILES:
                break

        top_level = []
        for path in sorted(self.cwd.iterdir(), key=lambda item: item.name):
            if path.name in _SKIP_DIRS:
                continue
            top_level.append(f"{path.name}/" if path.is_dir() else path.name)

        return RepositoryContext(
            root=str(self.cwd),
            manifests=[path for path in files if Path(path).name in _MANIFESTS],
            entrypoints=[path for path in files if Path(path).name in _ENTRYPOINT_NAMES],
            test_paths=[
                path
                for path in files
                if "test" in Path(path).name.lower()
                or any(part in {"test", "tests"} for part in Path(path).parts)
            ][:50],
            top_level=top_level[:100],
            files=files,
            language_counts=dict(languages.most_common()),
        )


def summarize_target_trajectory(
    records: list[dict[str, Any]],
) -> tuple[list[TargetRunSummary], list[dict[str, Any]]]:
    """Build lossless-enough target-run summaries without LLM interpretation."""

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        run_id = str(record.get("run_id") or record.get("session_id") or "unknown")
        grouped.setdefault(run_id, []).append((index, record))

    summaries: list[TargetRunSummary] = []
    evidence_catalog: list[dict[str, Any]] = []
    for run_id, events in grouped.items():
        task_prompt = ""
        final_response = ""
        outcome = "unknown"
        tools: list[str] = []
        tool_errors = 0
        errors: list[str] = []
        refs: list[str] = []

        for index, event in events:
            event_id = str(event.get("event_id") or f"{run_id}:event:{index}")
            event_type = str(event.get("type") or "")
            if event_type in {"target_run_started", "run_started"}:
                task_prompt = str(event.get("task_prompt") or task_prompt)
            elif event_type == "tool_result":
                tools.append(str(event.get("name") or "unknown"))
                tool_errors += int(bool(event.get("is_error")))
            elif event_type == "error":
                errors.append(str(event.get("error") or "unknown error"))
                outcome = "error"
            elif event_type == "done":
                final_response = str(event.get("final_response") or final_response)
                outcome = str(event.get("outcome") or "completed")

            if event_type in {"target_run_started", "tool_result", "error", "done"}:
                refs.append(event_id)
                evidence_catalog.append(
                    {
                        "event_id": event_id,
                        "trajectory_kind": "target_agent",
                        "run_id": run_id,
                        "type": event_type,
                        "tool": event.get("name"),
                        "arguments": event.get("arguments"),
                        "is_error": event.get("is_error"),
                        "content": str(
                            event.get("content")
                            or event.get("error")
                            or event.get("final_response")
                            or ""
                        )[:_MAX_EVIDENCE_CONTENT],
                    }
                )

        summaries.append(
            TargetRunSummary(
                run_id=run_id,
                task_prompt=task_prompt,
                outcome=outcome,
                final_response=final_response[:_MAX_RESPONSE],
                tool_calls=len(tools),
                tool_errors=tool_errors,
                tools_used=list(dict.fromkeys(tools)),
                error_messages=errors,
                evidence_refs=refs,
            )
        )
    return summaries, evidence_catalog[:_MAX_EVIDENCE_RECORDS]


def _summarize_reforge_loop(record: ReforgeLoopRecord) -> ReforgeLoopSummary:
    findings = record.diagnosis.get("findings") or []
    capability_gaps = [
        str(finding.get("capability_gap"))
        for finding in findings
        if isinstance(finding, dict) and finding.get("capability_gap")
    ]
    return ReforgeLoopSummary(
        loop_id=record.loop_id,
        loop=record.loop,
        stage=record.stage,
        proposal_summary=record.proposal_summary,
        capability_gaps=capability_gaps,
        selected_candidate=str(record.diagnosis.get("selected_candidate") or ""),
        changed_paths=record.changed_paths,
        component_status={
            component.component: component.status for component in record.components
        },
        commit=record.commit,
        completed=record.completed,
        remaining_gaps=record.remaining_gaps,
        error=record.error,
    )
