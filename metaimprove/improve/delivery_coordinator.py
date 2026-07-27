"""Composition boundary for deterministic acceptance and high-level delivery review."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..observability import traceable
from .acceptance_runner import AcceptanceRunner, RunResult
from .deliverer import Deliverer
from .models import ImprovementProposal


@dataclass
class Delivery:
    """Aggregate delivery decision consumed by the Pipeline."""

    passed: bool
    runs: list[RunResult] = field(default_factory=list)
    hard_gate_ok: bool = True
    acceptance_failures: list[str] = field(default_factory=list)
    goal_accepted: bool = False
    goal_review: str = ""
    reasons: list[str] = field(default_factory=list)
    # Worktree integrity is populated by Pipeline._deliver_immutable because
    # candidate snapshots/version isolation belong to the Pipeline.
    integrity_ok: bool = True
    mutation_diff: str = ""
    verified_tree: str = ""


class DeliveryCoordinator:
    """The only component that combines AcceptanceRunner and Deliverer verdicts."""

    def __init__(
        self,
        *,
        runner: AcceptanceRunner,
        deliverer: Deliverer,
    ):
        self.runner = runner
        self.deliverer = deliverer

    @classmethod
    def from_client(
        cls,
        *,
        client: Any,
        governance: str,
    ) -> DeliveryCoordinator:
        return cls(
            runner=AcceptanceRunner(governance=governance),
            deliverer=Deliverer(client=client),
        )

    @traceable(name="delivery.coordinate", run_type="chain")
    async def deliver(
        self,
        proposal: ImprovementProposal,
        *,
        cwd: str,
        loop_diff: str = "",
    ) -> Delivery:
        acceptance = await self.runner.run(proposal, cwd=cwd)
        goal = await self.deliverer.review(proposal, loop_diff=loop_diff)

        reasons = list(acceptance.failures)
        if not acceptance.passed:
            reasons.insert(0, "deterministic acceptance hard gate failed")
        if not goal.accepted:
            reasons.append("high-level goal realization review rejected")
        passed = acceptance.passed and goal.accepted
        return Delivery(
            passed=passed,
            runs=acceptance.runs,
            hard_gate_ok=acceptance.passed,
            acceptance_failures=acceptance.failures,
            goal_accepted=goal.accepted,
            goal_review=goal.text,
            reasons=(
                reasons
                or ["deterministic hard gate + high-level goal realization review passed"]
            ),
        )
