"""Deterministic Policy Gate — the reliability crown jewel.

Takes an ImprovementProposal and decides proceed / abstain / needs_human using
FIXED, EXPLAINABLE rules. The LLM's proposed decision and benefit/risk/effort
scores are evidence/inputs only; this gate has the final say (design invariant:
"no LLM judgment as the sole hard acceptance signal").

Thresholds live in an adjustable GatePolicy so users can tune how conservative
the gate is (e.g. raise the risk ceiling to allow riskier auto-changes).
"""

from __future__ import annotations

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
            "metaimprove/improve/",
            "metaimprove/orchestration/",
            "pyproject.toml",
            ".github/",
        ]
    )


@dataclass
class GateDecision:
    decision: Literal["proceed", "abstain", "needs_human"]
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
        human_reasons.append("no delivery_run — the Deliverer can't auto-verify it")
    if human_reasons:
        return GateDecision("needs_human", human_reasons)

    # 3. otherwise the expected value is favorable and it's auto-verifiable.
    return GateDecision(
        "proceed",
        [f"benefit {proposal.benefit} >= {policy.min_benefit}, risk {proposal.risk} "
         f"<= {policy.max_auto_risk}, grounded and auto-verifiable"],
    )


def _protected_hits(proposal: ImprovementProposal, protected: list[str]) -> list[str]:
    return [
        comp
        for comp in proposal.affected_components
        if any(p in comp for p in protected)
    ]


def _is_auto_verifiable(proposal: ImprovementProposal) -> bool:
    # auto-verifiable = the Deliverer has something to run: a non-empty delivery_run.
    return bool(proposal.delivery_run)
