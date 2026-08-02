"""Deliverer facade: agent-driven execution plus universal delivery gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..observability import traceable
from .acceptance_runner import AcceptanceRunner, RunResult, ScenarioRunResult
from .deliverer import Deliverer
from .models import ImprovementProposal


@dataclass
class Delivery:
    """Aggregate delivery decision consumed by the Pipeline."""

    passed: bool
    runs: list[RunResult] = field(default_factory=list)
    scenario_runs: list[ScenarioRunResult] = field(default_factory=list)
    delivery_gate_ok: bool = True
    acceptance_failures: list[str] = field(default_factory=list)
    goal_accepted: bool = False
    goal_review: str = ""
    handoff_failed: bool = False
    failure_kind: str = "none"
    reasons: list[str] = field(default_factory=list)
    # Worktree integrity is populated by Pipeline._deliver_immutable because
    # candidate snapshots/version isolation belong to the Pipeline.
    integrity_ok: bool = True
    mutation_diff: str = ""
    verified_tree: str = ""


class DeliveryCoordinator:
    """Protect the candidate while the Deliverer actively gathers evidence."""

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
        # Production Deliverer owns action selection and invokes the trusted
        # execution substrate as tools. The compatibility path keeps injected
        # test/legacy judges working while callers migrate.
        agentic_deliver = getattr(self.deliverer, "deliver", None)
        if callable(agentic_deliver):
            attempt = await agentic_deliver(
                proposal,
                cwd=cwd,
                loop_diff=loop_diff,
                runner=self.runner,
            )
            acceptance = attempt.evidence
            goal = attempt.review
        else:
            acceptance = await self.runner.run(proposal, cwd=cwd)
            goal = await self.deliverer.review(
                proposal,
                loop_diff=loop_diff,
                acceptance=acceptance,
            )

        reasons = list(acceptance.failures)
        if not acceptance.passed:
            reasons.insert(0, "universal delivery/commit gate failed")
        if goal.handoff_failed:
            reasons.append("Deliverer output hand-off failed")
        elif not goal.accepted:
            reasons.append(
                "high-level goal realization review rejected "
                f"(root cause: {goal.failure_kind})"
            )
        passed = acceptance.passed and goal.accepted
        failure_kind = goal.failure_kind
        if not passed and failure_kind == "none":
            if goal.accepted and _has_failed_safety_probe(acceptance.failures):
                # The product goal was realized, but the frozen plan attached an
                # incompatible safety contract. Replanning is safer than asking
                # Writer to implement an unrelated capability.
                failure_kind = "plan_gap"
            else:
                failure_kind = (
                    "environment_failure"
                    if not acceptance.passed
                    else "implementation_defect"
                )
        return Delivery(
            passed=passed,
            runs=acceptance.runs,
            scenario_runs=acceptance.scenario_runs,
            delivery_gate_ok=acceptance.passed,
            acceptance_failures=acceptance.failures,
            goal_accepted=goal.accepted,
            goal_review=goal.text,
            handoff_failed=goal.handoff_failed,
            failure_kind=failure_kind,
            reasons=(
                reasons
                or ["universal delivery gates clear + Deliverer accepted"]
            ),
        )


def _has_failed_safety_probe(failures: list[str]) -> bool:
    return any("system safety probe" in failure for failure in failures)
