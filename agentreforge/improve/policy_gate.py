"""Deterministic Policy Gate — the reliability crown jewel.

Takes an ImprovementProposal and decides proceed / abstain / needs_human using
FIXED, EXPLAINABLE rules. The LLM's proposed decision and benefit/risk/effort
scores are evidence/inputs only; this gate has the final say (design invariant:
"no LLM judgment as the sole hard acceptance signal").

Thresholds live in an adjustable GatePolicy so users can tune how conservative
the gate is (e.g. raise the risk ceiling to allow riskier auto-changes).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Literal

from .models import ImprovementProposal


@dataclass
class GatePolicy:
    max_auto_risk: int = 3  # risk above this -> escalate to human
    min_benefit: int = 3  # benefit below this -> abstain (not worth it)
    min_confidence: float = 0.5  # confidence below this -> abstain
    # critical parts of the improvement ENGINE itself: touching them needs a human.
    protected_paths: list[str] = field(
        default_factory=lambda: [
            "agentreforge/improve/",
            "agentreforge/orchestration/",
            "pyproject.toml",
            ".github/",
        ]
    )
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
    policy = policy or GatePolicy()

    # 1. ABSTAIN conditions first: weak / ungrounded proposals aren't worth acting
    #    on OR escalating to a human.
    abstain_reasons: list[str] = []
    if not proposal.evidence:
        abstain_reasons.append("no inspectable evidence")
    if proposal.benefit < policy.min_benefit:
        abstain_reasons.append(f"benefit {proposal.benefit} < min {policy.min_benefit}")
    if proposal.confidence < policy.min_confidence:
        abstain_reasons.append(f"confidence {proposal.confidence} < min {policy.min_confidence}")
    if abstain_reasons:
        return GateDecision("abstain", abstain_reasons)

    # 2. NEEDS_HUMAN conditions: worth considering, but must be supervised.
    human_reasons: list[str] = []
    if proposal.risk > policy.max_auto_risk:
        human_reasons.append(f"risk {proposal.risk} > max auto {policy.max_auto_risk}")
    hit = _protected_hits(proposal, policy.protected_paths)
    if hit:
        human_reasons.append(f"touches protected paths: {', '.join(hit)}")
    if not _is_auto_verifiable(proposal):
        human_reasons.append(
            "required acceptance criteria are not all executable command checks"
        )
    if human_reasons:
        return GateDecision("needs_human", human_reasons)

    # 3. otherwise the expected value is favorable and it's auto-verifiable.
    return GateDecision(
        "proceed",
        [f"benefit {proposal.benefit} >= {policy.min_benefit}, risk {proposal.risk} "
         f"<= {policy.max_auto_risk}, grounded and auto-verifiable"],
    )


def _is_auto_verifiable(proposal: ImprovementProposal) -> bool:
    required = [criterion for criterion in proposal.acceptance_criteria if criterion.required]
    return bool(required) and all(
        criterion.verification == "command" and criterion.command.strip()
        for criterion in required
    )


def evaluate_changes(
    proposal: ImprovementProposal,
    changed_paths: list[str],
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Check the real Writer output against the approved scope before delivery."""
    policy = policy or GatePolicy()
    forbidden = _forbidden_artifact_hits(changed_paths, policy.forbidden_artifacts)
    protected = _protected_path_hits(changed_paths, policy.protected_paths)
    outside = [
        path for path in changed_paths if not _matches_scope(path, proposal.allowed_write_paths)
    ]
    reasons: list[str] = []
    if forbidden:
        return GateDecision(
            "deny",
            [f"actual diff contains generated artifacts: {', '.join(forbidden)}"],
        )
    if protected:
        reasons.append(f"actual diff touches protected paths: {', '.join(protected)}")
    if outside:
        reasons.append(f"actual diff exceeds allowed_write_paths: {', '.join(outside)}")
    if reasons:
        return GateDecision("needs_human", reasons)
    return GateDecision("proceed", ["actual diff stays within the approved write scope"])


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


def _protected_hits(proposal: ImprovementProposal, protected: list[str]) -> list[str]:
    return _protected_path_hits(
        proposal.allowed_write_paths or proposal.affected_components,
        protected,
    )


def _protected_path_hits(paths: list[str], protected: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if any(_paths_overlap(path, protected_path) for protected_path in protected)
    ]


def _paths_overlap(left: str, right: str) -> bool:
    left = left.removeprefix("./")
    right = right.removeprefix("./")
    left_prefix = _literal_prefix(left)
    right_prefix = _literal_prefix(right)
    return (
        left.startswith(right)
        or right.startswith(left)
        or fnmatch.fnmatch(left, right)
        or fnmatch.fnmatch(right, left)
        or (
            bool(left_prefix)
            and bool(right_prefix)
            and (
                left_prefix.startswith(right_prefix)
                or right_prefix.startswith(left_prefix)
            )
        )
    )


def _literal_prefix(pattern: str) -> str:
    positions = [pattern.find(char) for char in "*?[" if char in pattern]
    return pattern[: min(positions)] if positions else pattern


def _matches_scope(path: str, scopes: list[str]) -> bool:
    path = path.removeprefix("./")
    for scope in scopes:
        scope = scope.removeprefix("./")
        if scope.endswith("/") and path.startswith(scope):
            return True
        if path == scope or fnmatch.fnmatch(path, scope):
            return True
    return False
