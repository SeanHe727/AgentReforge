"""Minimal policy routing plus generated-artifact denial."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Literal

from .models import ImprovementProposal


@dataclass
class GatePolicy:
    # Generated runtime artifacts are never product changes and cannot be
    # human-approved into a delivered candidate.
    forbidden_artifacts: list[str] = field(
        default_factory=lambda: [
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            "coverage.xml",
            ".coder_state.json",
        ]
    )


@dataclass
class GateDecision:
    decision: Literal["proceed", "abstain", "needs_human", "deny"]
    reasons: list[str]


def evaluate(proposal: ImprovementProposal, policy: GatePolicy | None = None) -> GateDecision:
    """Route only explicit Orchestrator intent; scores are advisory metadata."""

    if proposal.decision == "abstain":
        return GateDecision("abstain", [proposal.decision_reason or "Orchestrator abstained"])
    if proposal.decision == "needs_human":
        return GateDecision(
            "needs_human",
            [proposal.decision_reason or "Orchestrator requested human judgment"],
        )
    return GateDecision("proceed", ["Orchestrator proposed a runnable improvement"])


def evaluate_changes(
    _proposal: ImprovementProposal,
    changed_paths: list[str],
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Reject runtime artifacts; repository scope is enforced by the worktree."""
    policy = policy or GatePolicy()
    forbidden = _forbidden_artifact_hits(changed_paths, policy.forbidden_artifacts)
    if forbidden:
        return GateDecision(
            "deny",
            [f"actual diff contains generated artifacts: {', '.join(forbidden)}"],
        )
    return GateDecision("proceed", ["actual diff contains no forbidden runtime artifacts"])


def _forbidden_artifact_hits(paths: list[str], patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        clean = path.removeprefix("./")
        parts = clean.split("/")
        if any(
            fnmatch.fnmatch(clean, pattern)
            or fnmatch.fnmatch(parts[-1], pattern)
            or pattern in parts
            for pattern in patterns
        ):
            hits.append(path)
    return hits
