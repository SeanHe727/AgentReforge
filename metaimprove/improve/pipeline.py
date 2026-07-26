"""The improvement Pipeline — the total assembly that wires every stage.

It sequences the reliability-first flow end to end (design doc section 11):

    intent + trajectory
      -> Orchestrator            : evidence-grounded proposal + review checklist +
                                   delivery run/checklist (validated task DAG)
      -> deterministic Gate      : proceed / abstain / needs_human
      -> HITL approval           : a human approves the INTENT (governance-gated)
      -> Worktree (isolation)    : an isolated branch off a pinned base commit
         -> freeze proposal      : hash + approval stamp
         -> Writer + Reviewer    : bounded per-task implementation loop
         -> Deliverer            : run the candidate once, hard-gate + checklist review
         -> repair loop          : a rejected delivery feeds back to the Writer
         -> report to a file     : human-readable markdown
         -> merge back OR keep   : default keeps the worktree for a human to merge

Everything the pipeline needs is injectable (LLM client, tools, approver,
deliverer) so it stays testable and the policy stays out of the mechanism.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..llm.base import LlmClient
from ..observability import traceable
from ..tools.builtins import get_builtin_tools
from ..tools.registry import ToolRegistry
from . import policy_gate
from .acceptance import validate_acceptance
from .deliverer import Deliverer, Delivery
from .models import (
    ExecutionBlocker,
    FrozenProposal,
    ImprovementProposal,
)
from .orchestrator import Orchestrator
from .plan_validator import validate_plan
from .policy_gate import GateDecision, GatePolicy
from .run_config import (
    DetailLevel,
    GovernanceMode,
    RecursionPolicy,
    profile_for,
)
from .worktree import GitError, WorktreeSession, _run_git
from .writer_reviewer import ExecutionOutcome, WriterReviewer


@dataclass
class Approval:
    """A human's verdict on the proposal's intent, produced at the spec stage."""

    approved: bool
    approved_by: str = "unknown"
    notes: str = ""


# An approver reviews the proposal + the gate's decision and returns an Approval.
# Async so a real one can prompt a human / call a UI.
Approver = Callable[[ImprovementProposal, GateDecision], Awaitable[Approval]]


async def auto_approve(proposal: ImprovementProposal, decision: GateDecision) -> Approval:
    """Default non-interactive approver: approve anything the gate didn't abstain
    on. Real deployments inject a human-in-the-loop approver instead."""
    return Approval(approved=decision.decision != "abstain", approved_by="auto")


@dataclass
class ImprovementVersion:
    # a lightweight record of one delivered improvement loop (design doc section 10).
    loop: int
    base_commit: str
    branch: str
    verified_commit: str | None
    proposal_hash: str
    evaluation_hash: str


@dataclass
class PipelineResult:
    # where the run ended and everything it produced, for the caller + report.
    stage: str  # abstained|rejected|needs_human|blocked|failed|rejected_delivery|delivered|merged
    proposal: ImprovementProposal | None = None
    gate: GateDecision | None = None
    approval: Approval | None = None
    frozen: FrozenProposal | None = None
    outcome: ExecutionOutcome | None = None
    delivery: Delivery | None = None
    version: ImprovementVersion | None = None
    loop: int = 0
    repairs: int = 0
    blocker: ExecutionBlocker | None = None
    diff: str = ""
    merged_commit: str | None = None
    report_path: str | None = None
    error: str = ""
    # run-level (set on the final result the caller sees).
    manifest: dict[str, Any] = field(default_factory=dict)
    run_report_path: str | None = None


class ImprovementPipeline:
    def __init__(
        self,
        *,
        client: LlmClient,
        cwd: str,
        registry: ToolRegistry | None = None,
        policy: GatePolicy | None = None,
        level: DetailLevel | str = DetailLevel.STANDARD,
        governance: GovernanceMode | str = GovernanceMode.SUPERVISED,
        recursion: RecursionPolicy | None = None,
        approver: Approver = auto_approve,
        deliverer: Deliverer | None = None,
        auto_merge: bool = False,
        keep_worktree: bool = True,
    ):
        self.client = client
        self.cwd = cwd
        # one full (write-enabled) registry; the Orchestrator filters itself down
        # to read-only tools internally.
        self.registry = registry or _default_registry()
        self.policy = policy
        # detail level -> agent effort; governance -> HITL (independent).
        self.profile = profile_for(level)
        self.governance = GovernanceMode(governance)
        self.recursion = recursion or RecursionPolicy()
        self.approver = approver
        self.deliverer = deliverer or Deliverer(client=client, governance=self.governance.value)
        self.auto_merge = auto_merge
        self.keep_worktree = keep_worktree

    @traceable(name="improve.run", run_type="chain")
    async def run(
        self, *, intent: str, trajectory: list[dict[str, Any]] | None = None
    ) -> PipelineResult:
        """One Recursive Run: ONE worktree/branch, up to max_loops delivered loops,
        each a single commit; merge (or keep the branch) at the end."""
        trajectory = trajectory or []
        # ONE worktree/branch for the whole run; loops accumulate commits on it.
        wt = WorktreeSession(self.cwd, base="HEAD", keep=True)
        await wt.__aenter__()
        manifest = await self._preflight(wt, intent, len(trajectory))
        last_success: PipelineResult | None = None
        result: PipelineResult | None = None
        loop_history: list[str] = []  # what earlier loops tried, fed to the Orchestrator
        all_results: list[PipelineResult] = []
        try:
            for loop_i in range(self.recursion.max_loops):
                result = await self._run_loop(wt, intent, trajectory, loop_history, loop_i)
                all_results.append(result)
                loop_history.append(_loop_note(result))
                delivered = result.delivery is not None and result.delivery.passed
                # a non-delivered loop ends the run; return an earlier win if any.
                if not delivered:
                    return await self._finalize(wt, last_success or result, manifest, all_results)
                last_success = result
                if loop_i + 1 >= self.recursion.max_loops:
                    break
            final = last_success or result or PipelineResult("abstained")
            return await self._finalize(wt, final, manifest, all_results)
        finally:
            await wt.__aexit__(None, None, None)

    async def _preflight(self, wt: WorktreeSession, intent: str, traj: int) -> dict[str, Any]:
        """A Run Manifest: what this Recursive Run is operating on and with."""
        # a worktree is created from a COMMIT, so it excludes uncommitted changes.
        status = await _run_git("status", "--porcelain", cwd=self.cwd)
        return {
            "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
            "intent": intent[:200],
            "target": self.cwd,
            "candidate_branch": wt.branch,
            "base_commit": (wt.base_commit or "")[:12],
            "original_dirty": bool(status.stdout.strip()),
            "governance": self.governance.value,
            "level": self.profile.__dict__,
            "max_loops": self.recursion.max_loops,
            "max_repairs": self.recursion.max_repairs,
            "auto_merge": self.auto_merge,
            "keep_worktree": self.keep_worktree,
            "trajectory_records": traj,
        }

    @traceable(name="improve.loop", run_type="chain")
    async def _run_loop(
        self,
        wt: WorktreeSession,
        intent: str,
        trajectory: list[dict[str, Any]],
        loop_history: list[str],
        loop_i: int,
    ) -> PipelineResult:
        """One Improvement Loop on the shared worktree: analyze -> gate -> approve
        -> Writer/Reviewer -> Deliverer. A delivered loop makes exactly one commit;
        a failed loop rolls the worktree back to loop_base."""
        loop_base = await wt.head()  # this loop's base commit
        wt_path = str(wt.path)

        # 1. Orchestrator analyzes the current state, given what earlier loops tried.
        orch = Orchestrator(self.client, self.registry, wt_path)
        try:
            proposal = await orch.analyze(intent, trajectory, loop_history)
            # PIPELINE is the authority on DAG validity (the tool is advisory).
            proposal = await self._enforce_valid_proposal(orch, proposal)
        except ValueError as exc:
            return PipelineResult(stage="failed", loop=loop_i, error=str(exc))

        # 2. Deterministic gate.
        gate = policy_gate.evaluate(proposal, self.policy)
        if gate.decision == "abstain":
            return PipelineResult(stage="abstained", proposal=proposal, gate=gate, loop=loop_i)
        if gate.decision == "needs_human" and self.governance == GovernanceMode.AUTONOMOUS:
            return PipelineResult(stage="needs_human", proposal=proposal, gate=gate, loop=loop_i)

        # 3. HITL approval.
        if self.governance == GovernanceMode.AUTONOMOUS:
            approval = await auto_approve(proposal, gate)
        else:
            approval = await self.approver(proposal, gate)
        if not approval.approved:
            return PipelineResult(
                stage="rejected", proposal=proposal, gate=gate, approval=approval, loop=loop_i
            )

        # 4. Freeze the approved proposal, then implement.
        frozen = _freeze_proposal(proposal, approval, target_commit=loop_base)
        writer = WriterReviewer(
            client=self.client,
            registry=self.registry,
            max_rounds=self.profile.max_rounds,
            max_task_turns=self.profile.max_task_turns,
        )

        # 5. Only a COMPLETE, non-blocked implementation may reach the Deliverer.
        outcome = await writer.run(proposal=proposal, worktree=wt)
        incomplete = self._incomplete_result(outcome, gate, approval, frozen, proposal, loop_i)
        if incomplete is not None:
            await wt.reset_hard(loop_base)  # roll back the failed loop
            return incomplete

        # 6. Check the REAL diff against the pre-approved write scope.  The proposal
        # gate above checks intent; this gate checks what the Writer actually did.
        changed_paths = await wt.changed_since(loop_base)
        actual_gate = policy_gate.evaluate_changes(proposal, changed_paths, self.policy)
        if actual_gate.decision == "needs_human":
            if self.governance == GovernanceMode.AUTONOMOUS:
                await wt.reset_hard(loop_base)
                return PipelineResult(
                    stage="needs_human", proposal=proposal, gate=actual_gate,
                    approval=approval, frozen=frozen, outcome=outcome, loop=loop_i,
                    error="; ".join(actual_gate.reasons),
                )
            scope_approval = await self.approver(proposal, actual_gate)
            if not scope_approval.approved:
                await wt.reset_hard(loop_base)
                return PipelineResult(
                    stage="rejected", proposal=proposal, gate=actual_gate,
                    approval=scope_approval, frozen=frozen, outcome=outcome, loop=loop_i,
                )

        # 7. Deliverer sees loop_base -> current candidate.  Its commands are
        # verification-only: any repository mutation rejects the delivery.
        delivery, loop_diff = await self._deliver_immutable(
            proposal, wt, loop_base, wt_path
        )
        repairs = 0
        while (
            not delivery.passed
            and delivery.integrity_ok
            and repairs < self.recursion.max_repairs
        ):
            repairs += 1
            outcome = await writer.repair(worktree=wt, instruction=_repair_instruction(delivery))
            if outcome.blocker is not None or not outcome.completed:
                break
            repaired_gate = policy_gate.evaluate_changes(
                proposal, await wt.changed_since(loop_base), self.policy
            )
            if repaired_gate.decision == "needs_human":
                if self.governance == GovernanceMode.AUTONOMOUS:
                    delivery = Delivery(
                        passed=False,
                        hard_gate_ok=False,
                        reasons=repaired_gate.reasons,
                    )
                    loop_diff = await wt.diff_since(loop_base)
                    break
                repair_approval = await self.approver(proposal, repaired_gate)
                if not repair_approval.approved:
                    delivery = Delivery(
                        passed=False,
                        hard_gate_ok=False,
                        reasons=["repair scope was not approved", *repaired_gate.reasons],
                    )
                    loop_diff = await wt.diff_since(loop_base)
                    break
            delivery, loop_diff = await self._deliver_immutable(
                proposal, wt, loop_base, wt_path
            )

        if not delivery.passed:
            # record the rejected diff, then roll back so it can't pollute the next loop.
            result = PipelineResult(
                stage="rejected_delivery", proposal=proposal, gate=gate, approval=approval,
                frozen=frozen, outcome=outcome, delivery=delivery, diff=loop_diff,
                loop=loop_i, repairs=repairs,
            )
            result.report_path = self._write_report(result)
            await wt.reset_hard(loop_base)
            return result

        # 8. Delivered: one commit for the whole loop.
        try:
            loop_commit = await wt.commit(
                f"improve loop {loop_i}: {proposal.summary}",
                expected_tree=delivery.verified_tree,
            )
        except GitError as exc:
            await wt.reset_hard(loop_base)
            return PipelineResult(
                stage="failed", proposal=proposal, gate=gate, approval=approval,
                frozen=frozen, outcome=outcome, delivery=delivery, diff=loop_diff,
                loop=loop_i, repairs=repairs, error=str(exc),
            )
        version = ImprovementVersion(
            loop=loop_i, base_commit=loop_base, branch=wt.branch,
            verified_commit=loop_commit, proposal_hash=frozen.proposal_hash, evaluation_hash="",
        )
        result = PipelineResult(
            stage="delivered", proposal=proposal, gate=gate, approval=approval, frozen=frozen,
            outcome=outcome, delivery=delivery, version=version, loop=loop_i,
            repairs=repairs, diff=loop_diff,
        )
        result.report_path = self._write_report(result)
        return result

    async def _finalize(
        self,
        wt: WorktreeSession,
        result: PipelineResult,
        manifest: dict[str, Any],
        all_results: list[PipelineResult],
    ) -> PipelineResult:
        """End the Recursive Run: merge if asked + delivered, else keep the branch
        (so delivered commits stay reachable) or clean up if nothing was delivered."""
        delivered = result.delivery is not None and result.delivery.passed
        if self.auto_merge and delivered and result.proposal is not None:
            msg = f"improve: {result.proposal.summary}"
            result.merged_commit = await wt.merge_back(message=msg)
            result.stage = "merged"
            await wt.remove()  # safely merged into main; nothing left to keep
        elif not delivered:
            await wt.remove()  # nothing delivered — clean up entirely
        elif not self.keep_worktree:
            # delivered but not merged: drop the dir but KEEP the branch (no dangling).
            await wt.remove(keep_branch=True)
        # a run-level summary: manifest + every loop's commit/stage (traceable).
        result.manifest = manifest
        result.run_report_path = self._write_run_report(manifest, all_results, result)
        return result

    def _write_run_report(
        self, manifest: dict[str, Any], all_results: list[PipelineResult], final: PipelineResult
    ) -> str:
        reports_dir = Path(self.cwd) / ".meta-improve" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# Recursive Run {manifest['run_id']}", "", "## Manifest"]
        lines += [f"- **{k}:** {v}" for k, v in manifest.items()]
        lines += ["", "## Loops"]
        for r in all_results:
            commit = (r.version.verified_commit or "")[:12] if r.version else "—"
            reasons = "; ".join(r.delivery.reasons) if r.delivery else (r.error or "")
            lines.append(f"- loop {r.loop} [{r.stage}] commit {commit} — {reasons}")
        if final.merged_commit:
            lines.append(f"\n**Merged to main:** {final.merged_commit[:12]}")
        elif final.version:
            lines.append(f"\n**Candidate branch kept:** `{final.version.branch}`")
        path = reports_dir / f"{manifest['run_id']}_run.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    async def _enforce_valid_proposal(
        self, orch: Orchestrator, proposal: ImprovementProposal, max_tries: int = 2
    ) -> ImprovementProposal:
        """Hard-validate both the task graph and the executable acceptance contract."""
        tries = 0
        while tries <= max_tries:
            dag = validate_plan([t.model_dump() for t in proposal.tasks])
            acceptance = validate_acceptance(proposal)
            problems = []
            if not dag.valid:
                problems.append(f"invalid task DAG — {dag.summary()}")
            if not acceptance.valid:
                problems.append(f"invalid acceptance contract — {acceptance.summary()}")
            if not problems:
                return proposal
            if tries >= max_tries:
                raise ValueError(
                    f"proposal invalid after {tries} revision(s): {'; '.join(problems)}"
                )
            tries += 1
            proposal = await orch.revise(proposal, "; ".join(problems))
        raise AssertionError("unreachable")

    async def _deliver_immutable(
        self,
        proposal: ImprovementProposal,
        wt: WorktreeSession,
        loop_base: str,
        wt_path: str,
    ) -> tuple[Delivery, str]:
        """Deliver one exact candidate tree and reject verification side effects."""
        candidate_tree = await wt.snapshot()
        reviewed_diff = await wt.diff_since(loop_base)
        delivery = await self.deliverer.deliver(
            proposal, cwd=wt_path, loop_diff=reviewed_diff
        )
        delivered_tree = await wt.snapshot()
        final_diff = await wt.diff_since(loop_base)
        if delivered_tree != candidate_tree:
            delivery.passed = False
            delivery.integrity_ok = False
            delivery.mutation_diff = await wt.diff_since(candidate_tree)
            delivery.reasons.insert(
                0,
                "delivery mutated the candidate worktree; verification commands must be read-only",
            )
        else:
            delivery.verified_tree = candidate_tree
        return delivery, final_diff

    def _incomplete_result(
        self,
        outcome: ExecutionOutcome,
        gate: GateDecision,
        approval: Approval,
        frozen: FrozenProposal,
        proposal: ImprovementProposal,
        loop_i: int,
    ) -> PipelineResult | None:
        """A blocked or not-fully-completed implementation must not reach delivery."""
        if outcome.blocker is not None:
            stage, error = "blocked", ""
        elif not outcome.completed:
            stage, error = "failed", "implementation did not complete all tasks"
        else:
            return None
        result = PipelineResult(
            stage=stage, proposal=proposal, gate=gate, approval=approval, frozen=frozen,
            outcome=outcome, blocker=outcome.blocker, diff=outcome.diff, loop=loop_i, error=error,
        )
        result.report_path = self._write_report(result)
        return result

    def _write_report(self, result: PipelineResult) -> str:
        reports_dir = Path(self.cwd) / ".meta-improve" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = reports_dir / f"{stamp}_{result.stage}.md"
        path.write_text(render_report(result), encoding="utf-8")
        return str(path)


# --- helpers --------------------------------------------------------------------


def _default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    return registry


def _freeze_proposal(
    proposal: ImprovementProposal,
    approval: Approval,
    *,
    target_commit: str,
    version: int = 1,
) -> FrozenProposal:
    """Hash + stamp an approved proposal so the intent can't silently drift."""
    digest = hashlib.sha256(proposal.model_dump_json().encode("utf-8")).hexdigest()
    return FrozenProposal(
        proposal_id=digest[:12],
        proposal_version=version,
        proposal_hash=digest,
        approved_by=approval.approved_by,
        approved_at=datetime.now(UTC).isoformat(),
        target_commit=target_commit,
        baseline_run_id="",
        proposal=proposal,
    )


def _loop_note(result: PipelineResult) -> str:
    """A one-line summary of a finished loop, fed to the next loop's Orchestrator."""
    summary = result.proposal.summary[:200] if result.proposal else "(no proposal)"
    note = f"loop {result.loop} [{result.stage}]: {summary}"
    # why it ended as it did, so the next loop can plan around it.
    reasons = list(result.delivery.reasons) if result.delivery else []
    if result.error:
        reasons.append(result.error)
    if reasons:
        note += f" — {'; '.join(reasons)}"
    return note


def _repair_instruction(delivery: Delivery) -> str:
    """Turn a rejected delivery into a concrete repair brief for the Writer."""
    lines = [
        "The candidate did not pass delivery. Fix the PRODUCT CODE so it does. "
        "What was run and what the Deliverer found:",
    ]
    # the failing run commands + their output.
    for r in delivery.runs:
        if r.exit_code != 0:
            tail = r.output[-800:] if r.output else ""
            lines.append(f"\n$ {r.command} (exit {r.exit_code})\n{tail}")
    # the Deliverer's checklist review (why it rejected).
    if delivery.checklist_review:
        lines.append(f"\nDeliverer review:\n{delivery.checklist_review}")
    return "\n".join(lines)


def render_report(result: PipelineResult) -> str:
    """A human-readable markdown report of the whole run."""
    p = result.proposal
    lines = [
        f"# Improvement report — loop {result.loop} — {result.stage}",
        "",
        f"- **Intent summary:** {p.summary if p else '(none)'}",
    ]
    if result.version:
        v = result.version
        lines.append(
            f"- **Loop {v.loop}:** base {v.base_commit[:12]} -> "
            f"commit {(v.verified_commit or '(none)')[:12]} on `{v.branch}`"
        )
    if result.error:
        lines.append(f"- **Error:** {result.error}")
    if result.gate:
        lines.append(f"- **Gate:** {result.gate.decision} — {'; '.join(result.gate.reasons)}")
    if result.approval:
        lines.append(
            f"- **Approval:** {'approved' if result.approval.approved else 'rejected'} "
            f"by {result.approval.approved_by}"
        )
    if result.frozen:
        lines.append(
            f"- **Frozen proposal:** {result.frozen.proposal_id} "
            f"(hash {result.frozen.proposal_hash[:12]}), base {result.frozen.target_commit[:12]}"
        )
    if result.delivery:
        d = result.delivery
        verdict = "passed" if d.passed else "rejected"
        lines.append(f"- **Delivery:** {verdict} — {'; '.join(d.reasons)}")
    if result.repairs:
        lines.append(f"- **Repair rounds:** {result.repairs}")
    if result.merged_commit:
        lines.append(f"- **Merged:** {result.merged_commit[:12]}")

    if result.outcome:
        lines += ["", "## Implementation"]
        for t in result.outcome.task_outcomes:
            lines.append(f"- `{t.task_id}`: {t.status} in {t.rounds} round(s)")
    if result.blocker:
        b = result.blocker
        lines += [
            "",
            "## Blocker",
            f"- Task `{b.task_id}` did not converge in {b.review_rounds} rounds.",
            f"- Recommendation: **{b.recommendation}**",
            f"- Affected tasks: {', '.join(b.affected_tasks) or '(none)'}",
        ]
        for f in b.unresolved_findings:
            lines.append(f"  - [{f.severity}] {f.description}")

    # the Deliverer's single candidate run: what was executed and its output.
    if result.delivery and result.delivery.runs:
        lines += ["", "## Delivery run"]
        for r in result.delivery.runs:
            tail = r.output[-1500:] if r.output else ""
            lines += [f"\n`$ {r.command}` (exit {r.exit_code})", "```", tail, "```"]
        if result.delivery.checklist_review:
            lines += ["", "### Checklist review", result.delivery.checklist_review]

    if result.diff:
        diff = result.diff if len(result.diff) < 8000 else result.diff[:8000] + "\n...(truncated)"
        lines += ["", "## Diff", "```diff", diff, "```"]

    return "\n".join(lines) + "\n"
