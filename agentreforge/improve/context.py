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

from .models import (
    BacklogCandidate,
    CandidateDiagnosis,
    CandidateHistory,
    CandidateIntervention,
    CandidatePriority,
    CandidateScope,
)
from .records import ReforgeLoopRecord, ReforgeLoopSummary, TargetRunSummary
from .trajectory import tool_result_is_error

_SKIP_DIRS = {
    ".git",
    ".agentreforge",
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
_TARGET_RUN_SEMANTICS = (
    "Each target_agent_run is a trial of target_commit on a disposable task "
    "workspace. The generated artifact and its external evaluation are evidence "
    "about that target agent version, even when task files differ from the target "
    "agent repository."
)


def _same_git_commit(left: str, right: str) -> bool:
    """Compare full or abbreviated Git object IDs without accepting tiny prefixes."""

    if not left or not right:
        return False
    normalized_left = left.casefold()
    normalized_right = right.casefold()
    if normalized_left == normalized_right:
        return True
    return (
        min(len(normalized_left), len(normalized_right)) >= 7
        and (
            normalized_left.startswith(normalized_right)
            or normalized_right.startswith(normalized_left)
        )
    )


class RepositoryContext(BaseModel):
    root: str
    manifests: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    test_paths: list[str] = Field(default_factory=list)
    top_level: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    language_counts: dict[str, int] = Field(default_factory=dict)


class CurrentRunAlert(BaseModel):
    """A salient factual index of unresolved evidence on the current commit."""

    run_id: str
    outcome: str
    task_prompt: str
    step_budget_exhausted: bool
    evaluation_passed: bool | None
    evaluation_summary: str
    evidence_refs: list[str] = Field(default_factory=list)


class OrchestratorContext(BaseModel):
    """Everything that must be present before Orchestrator reasoning starts."""

    improvement_intent: str
    target_run_semantics: str = _TARGET_RUN_SEMANTICS
    target_agent_runs: list[TargetRunSummary] = Field(default_factory=list)
    # This is an attention index, not a policy gate or ranking decision. It
    # prevents terminal current-version evidence from being buried beneath
    # many successful or stale runs.
    current_run_alerts: list[CurrentRunAlert] = Field(default_factory=list)
    # AgentReforge's own history; never mixed into target_agent_runs.
    previous_reforge_loops: list[ReforgeLoopSummary] = Field(default_factory=list)
    # Dynamic hypothesis list reconstructed from every previous proposal.  The
    # Orchestrator re-ranks and revises it; it is not a frozen implementation plan.
    improvement_backlog: dict[str, BacklogCandidate] = Field(default_factory=dict)
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
        summaries, evidence = summarize_target_trajectory(
            target_trajectory,
            current_commit=str(run_manifest.get("loop_base") or ""),
        )
        raw_evidence = {}
        for index, record in enumerate(target_trajectory):
            run_id = str(record.get("run_id") or record.get("session_id") or "unknown")
            event_id = str(record.get("event_id") or f"{run_id}:event:{index}")
            raw_evidence[event_id] = record
        loop_summaries = [_summarize_reforge_loop(loop) for loop in previous_reforge_loops]
        return OrchestratorContext(
            improvement_intent=intent,
            target_agent_runs=summaries,
            current_run_alerts=[
                CurrentRunAlert(
                    run_id=summary.run_id,
                    outcome=summary.outcome,
                    task_prompt=summary.task_prompt,
                    step_budget_exhausted=summary.step_budget_exhausted,
                    evaluation_passed=summary.evaluation_passed,
                    evaluation_summary=summary.evaluation_summary,
                    evidence_refs=summary.evidence_refs,
                )
                for summary in summaries
                if summary.is_current
                and (
                    summary.outcome
                    in {"failed_verification", "incomplete", "error", "stopped"}
                    or summary.stopped_early
                    or summary.evaluation_passed is False
                )
            ],
            previous_reforge_loops=loop_summaries,
            improvement_backlog=_build_improvement_backlog(previous_reforge_loops),
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
    *,
    current_commit: str = "",
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
        reported_outcome = "unknown"
        saw_error = False
        stopped_early = False
        step_budget_exhausted = False
        evaluation_passed: bool | None = None
        evaluation_summary = ""
        tools: list[str] = []
        tool_errors = 0
        errors: list[str] = []
        refs: list[str] = []
        evidence_source = "external"
        target_commit = ""

        for index, event in events:
            event_id = str(event.get("event_id") or f"{run_id}:event:{index}")
            event_type = str(event.get("type") or "")
            evidence_source = str(event.get("evidence_source") or evidence_source)
            target_commit = str(event.get("target_commit") or target_commit)
            if event_type in {"target_run_started", "run_started"}:
                task_prompt = str(event.get("task_prompt") or task_prompt)
            elif event_type == "tool_result":
                tools.append(str(event.get("name") or "unknown"))
                content = str(event.get("content") or "")
                normalized_error = tool_result_is_error(
                    content,
                    declared=bool(event.get("is_error")),
                )
                tool_errors += int(normalized_error)
            elif event_type == "error":
                errors.append(str(event.get("error") or "unknown error"))
                saw_error = True
            elif event_type == "evaluation_result":
                evaluation_passed = bool(event.get("passed"))
                evaluation_summary = str(
                    event.get("content") or "external verification returned no details"
                )[:_MAX_EVIDENCE_CONTENT]
                if not evaluation_passed:
                    errors.append(evaluation_summary)
            elif event_type == "done":
                final_response = str(event.get("final_response") or final_response)
                reported_outcome = str(event.get("outcome") or "completed")
                normalized_response = final_response.casefold()
                step_budget_exhausted = any(
                    marker in normalized_response
                    for marker in (
                        "reached max steps",
                        "maximum steps reached",
                        "step budget exhausted",
                    )
                )
                stopped_early = step_budget_exhausted or reported_outcome in {
                    "incomplete",
                    "stopped",
                }

            if event_type in {
                "target_run_started",
                "tool_result",
                "evaluation_result",
                "error",
                "done",
            }:
                refs.append(event_id)
                evidence_catalog.append(
                    {
                        "event_id": event_id,
                        "trajectory_kind": "target_agent",
                        "run_id": run_id,
                        "type": event_type,
                        "tool": event.get("name"),
                        "arguments": event.get("arguments"),
                        "is_error": (
                            normalized_error
                            if event_type == "tool_result"
                            else not bool(event.get("passed"))
                            if event_type == "evaluation_result"
                            else event.get("is_error")
                        ),
                        "evidence_source": event.get("evidence_source"),
                        "target_commit": event.get("target_commit"),
                        "content": str(
                            event.get("content")
                            or event.get("error")
                            or event.get("final_response")
                            or ""
                        )[:_MAX_EVIDENCE_CONTENT],
                    }
                )

        if saw_error:
            outcome = "error"
        elif evaluation_passed is False:
            outcome = "failed_verification"
        elif stopped_early:
            outcome = "incomplete"
        else:
            outcome = reported_outcome

        summaries.append(
            TargetRunSummary(
                run_id=run_id,
                task_prompt=task_prompt,
                outcome=outcome,
                final_response=final_response[:_MAX_RESPONSE],
                stopped_early=stopped_early,
                step_budget_exhausted=step_budget_exhausted,
                evaluation_passed=evaluation_passed,
                evaluation_summary=evaluation_summary,
                tool_calls=len(tools),
                tool_errors=tool_errors,
                tools_used=list(dict.fromkeys(tools)),
                error_messages=errors,
                evidence_refs=refs,
                evidence_source=evidence_source,
                target_commit=target_commit,
                is_current=_same_git_commit(target_commit, current_commit),
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
    candidate_backlog = record.diagnosis.get("candidate_backlog") or {}
    if isinstance(candidate_backlog, dict):
        capability_gaps.extend(
            str(item.get("diagnosis", {}).get("capability_gap"))
            for item in candidate_backlog.values()
            if isinstance(item, dict)
            and isinstance(item.get("diagnosis"), dict)
            and item["diagnosis"].get("capability_gap")
        )
    return ReforgeLoopSummary(
        loop_id=record.loop_id,
        loop=record.loop,
        stage=record.stage,
        proposal_summary=record.proposal_summary,
        capability_gaps=capability_gaps,
        selected_candidates=[
            str(value)
            for value in (record.diagnosis.get("selected_candidates") or [])
            if value
        ] or (
            [str(record.diagnosis["selected_candidate_id"])]
            if record.diagnosis.get("selected_candidate_id")
            else
            [str(record.diagnosis["selected_candidate"])]
            if record.diagnosis.get("selected_candidate")
            else []
        ),
        changed_paths=record.changed_paths,
        component_status={
            component.component: component.status for component in record.components
        },
        commit=record.commit,
        completed=record.completed,
        achievements=record.achievements,
        completion_scopes=record.completion_scopes,
        remaining_gaps=record.remaining_gaps,
        failure_kind=record.failure_kind,
        attempt_fingerprint=record.attempt_fingerprint,
        error=record.error,
    )


def _build_improvement_backlog(
    records: list[ReforgeLoopRecord],
) -> dict[str, BacklogCandidate]:
    """Materialize prior Candidate hypotheses without treating them as truth.

    Stable ids carry structured items forward to their latest observation. Broader
    capability-level equivalence remains an Orchestrator judgment because deterministic
    string normalization cannot safely merge different mechanisms.
    """

    backlog: dict[str, BacklogCandidate] = {}
    for record in sorted(records, key=lambda item: item.loop):
        structured_backlog = record.diagnosis.get("candidate_backlog") or {}
        if isinstance(structured_backlog, dict) and structured_backlog:
            completion_by_candidate = {
                scope.candidate: scope for scope in record.completion_scopes
            }
            selected_id = str(record.diagnosis.get("selected_candidate_id") or "")
            for candidate_id, raw in structured_backlog.items():
                if not isinstance(raw, dict):
                    continue
                diagnosis = raw.get("diagnosis") or {}
                intervention = raw.get("intervention") or {}
                if not isinstance(diagnosis, dict) or not isinstance(
                    intervention, dict
                ):
                    continue
                if str(candidate_id) == selected_id and record.completed:
                    status = "behavior_verified"
                    reason = (
                        "selected, delivered, and verified in the recorded evidence scope"
                    )
                elif str(candidate_id) == selected_id:
                    status = "attempt_failed"
                    reason = record.error or record.failure_kind
                else:
                    status = str(raw.get("status") or "open")
                    reason = str(
                        (raw.get("history") or {}).get("disposition_reason") or ""
                    )
                history = raw.get("history") or {}
                if not isinstance(history, dict):
                    history = {}
                completion = completion_by_candidate.get(str(candidate_id))
                backlog[str(candidate_id)] = BacklogCandidate(
                    id=str(candidate_id),
                    status=status,
                    title=str(raw.get("title") or candidate_id),
                    diagnosis=CandidateDiagnosis.model_validate(diagnosis),
                    intervention=CandidateIntervention.model_validate(intervention),
                    priority=CandidatePriority.model_validate(raw.get("priority") or {}),
                    scope=CandidateScope.model_validate(raw.get("scope") or {}),
                    dependencies=[
                        str(value) for value in (raw.get("dependencies") or [])
                    ],
                    conflicts_with=[
                        str(value) for value in (raw.get("conflicts_with") or [])
                    ],
                    history=CandidateHistory(
                        first_seen_loop=int(history.get("first_seen_loop") or record.loop),
                        last_reviewed_loop=record.loop,
                        previous_attempts=[
                            str(value)
                            for value in (history.get("previous_attempts") or [])
                        ],
                        verification_scope=(
                            completion.evidence_scope
                            if completion
                            else [
                                str(value)
                                for value in (history.get("verification_scope") or [])
                            ]
                        ),
                        verification_level=(
                            completion.verification_level
                            if completion
                            else str(history.get("verification_level") or "none")
                        ),
                        disposition_reason=reason,
                    ),
                )
            continue

        candidates = record.diagnosis.get("candidates") or []
        selected = {
            str(value)
            for value in (record.diagnosis.get("selected_candidates") or [])
            if value
        }
        if not selected and record.diagnosis.get("selected_candidate"):
            selected.add(str(record.diagnosis["selected_candidate"]))

        completion_by_candidate = {
            scope.candidate: scope for scope in record.completion_scopes
        }
        for raw in candidates:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            candidate = str(raw["name"])
            capability_gap = str(raw.get("capability_gap") or "")
            mechanism = str(raw.get("mechanism") or "")
            expected_delta = str(raw.get("expected_capability_delta") or "")
            if candidate in selected and record.completed:
                status = "behavior_verified"
                reason = "selected, delivered, and verified in the recorded evidence scope"
            elif candidate in selected:
                status = "attempt_failed"
                reason = record.error or record.failure_kind
            elif raw.get("rejected_reason"):
                status = "deferred"
                reason = str(raw["rejected_reason"])
            else:
                status = "open"
                reason = "considered but not selected"
            completion = completion_by_candidate.get(candidate)

            backlog[candidate] = BacklogCandidate(
                id=candidate,
                status=status,
                title=candidate,
                diagnosis=CandidateDiagnosis(
                    capability_gap=capability_gap or candidate,
                    evidence_refs=[
                        str(value)
                        for value in (raw.get("evidence_refs") or [])
                        if value
                    ],
                ),
                intervention=CandidateIntervention(
                    level=str(raw.get("level") or "workflow"),
                    mechanism=mechanism or candidate,
                    expected_capability_delta=expected_delta or candidate,
                ),
                priority=CandidatePriority(
                    benefit=int(raw.get("benefit") or 1),
                    risk=int(raw.get("risk") or 1),
                    effort=int(raw.get("effort") or 1),
                    confidence=min(
                        1.0,
                        max(0.0, float(raw.get("evidence_strength") or 1) / 5.0),
                    ),
                    rank_reason=str(raw.get("rejected_reason") or ""),
                ),
                history=CandidateHistory(
                    first_seen_loop=record.loop,
                    last_reviewed_loop=record.loop,
                    previous_attempts=(
                        [record.attempt_fingerprint]
                        if candidate in selected and record.attempt_fingerprint
                        else []
                    ),
                    verification_scope=completion.evidence_scope if completion else [],
                    verification_level=(
                        completion.verification_level if completion else "none"
                    ),
                    disposition_reason=reason,
                ),
            )
    return backlog
