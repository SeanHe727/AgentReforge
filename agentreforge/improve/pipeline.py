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
         -> Deliverer            : run the candidate once, delivery gate + goal review
         -> repair loop          : a rejected delivery feeds back to the Writer
         -> report to a file     : human-readable markdown
         -> merge back OR keep   : default keeps the worktree for a human to merge

Everything the pipeline needs is injectable (LLM client, tools, approver,
deliverer) so it stays testable and the policy stays out of the mechanism.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..llm.base import LlmClient
from ..observability import traceable
from ..rag.code_index import CodeIndex
from ..tools.builtins import get_builtin_tools
from ..tools.registry import ToolRegistry
from . import policy_gate
from .acceptance import validate_acceptance
from .context import OrchestratorContextBuilder
from .delivery_coordinator import Delivery, DeliveryCoordinator
from .history_index import ImprovementHistoryIndex
from .models import (
    ExecutionBlocker,
    FrozenProposal,
    ImprovementProposal,
)
from .orchestrator import Orchestrator
from .plan_validator import validate_plan
from .policy_gate import GateDecision, GatePolicy
from .records import (
    ComponentRecord,
    ImprovementRecordStore,
    RecursiveRunRecord,
    ReforgeLoopRecord,
    jsonable,
)
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
    return Approval(
        approved=decision.decision in ("proceed", "needs_human"),
        approved_by="auto",
    )


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
    stage: str  # includes rejected_policy|partially_delivered plus loop terminal stages
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
    terminal_stage: str = ""
    terminal_loop: int | None = None
    terminal_report_path: str | None = None
    terminal_error: str = ""


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
        delivery_coordinator: DeliveryCoordinator | None = None,
        # Compatibility/injection seam: any object implementing ``deliver(...)``.
        deliverer: Any = None,
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
        if delivery_coordinator is not None and deliverer is not None:
            raise ValueError("pass delivery_coordinator or deliverer, not both")
        self.delivery_coordinator = (
            delivery_coordinator
            or deliverer
            or DeliveryCoordinator.from_client(
                client=client,
                governance=self.governance.value,
            )
        )
        self.auto_merge = auto_merge
        self.keep_worktree = keep_worktree

    @traceable(name="improve.run", run_type="chain")
    async def run(
        self,
        *,
        intent: str,
        target_trajectory: list[dict[str, Any]] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        """One Recursive Run: ONE worktree/branch, up to max_loops delivered loops,
        each a single commit; merge (or keep the branch) at the end."""
        if target_trajectory is not None and trajectory is not None:
            raise ValueError("pass target_trajectory, not both trajectory names")
        # ``trajectory`` remains a compatibility alias, but internally the name is
        # explicit so target-agent evidence cannot be confused with Reforge history.
        target_trajectory = target_trajectory if target_trajectory is not None else trajectory
        target_trajectory = list(target_trajectory or [])
        # ONE worktree/branch for the whole run; loops accumulate commits on it.
        wt = WorktreeSession(self.cwd, base="HEAD", keep=True)
        await wt.__aenter__()
        try:
            manifest = await self._preflight(wt, intent, len(target_trajectory))
            target_trajectory = _tag_baseline_trajectory(
                target_trajectory,
                commit=wt.base_commit or "",
            )
            run_record = RecursiveRunRecord(
                run_id=manifest["run_id"],
                intent=intent,
                target_repo=self.cwd,
                branch=wt.branch,
                base_commit=wt.base_commit or "",
                manifest=manifest,
                target_run_refs=_target_run_refs(target_trajectory),
            )
            record_store = ImprovementRecordStore(self.cwd, run_record.run_id)
            record_store.start(run_record)
            history_index = ImprovementHistoryIndex(
                Path(self.cwd) / ".agentreforge" / "improvement_history.db"
            )
            history_index.rebuild(self.cwd)
        except Exception:
            await wt.__aexit__(None, None, None)
            raise
        last_success: PipelineResult | None = None
        result: PipelineResult | None = None
        loop_history: list[ReforgeLoopRecord] = []
        all_results: list[PipelineResult] = []
        try:
            for loop_i in range(self.recursion.max_loops):
                loop_base = await wt.head()
                result = await self._run_loop(
                    wt,
                    intent,
                    target_trajectory,
                    loop_history,
                    manifest,
                    history_index,
                    loop_i,
                )
                all_results.append(result)
                loop_record = _materialize_loop_record(
                    run_record.run_id, loop_base, result
                )
                loop_record = record_store.append_loop(loop_record, result.diff)
                history_index.index_loop(
                    loop_record,
                    self.cwd,
                    record_path=str(
                        record_store.loops_dir
                        / f"loop_{loop_record.loop}"
                        / "record.json"
                    ),
                )
                loop_history.append(loop_record)
                run_record.loop_refs.append(
                    f"loops/loop_{loop_record.loop}/record.json"
                )
                delivered = result.delivery is not None and result.delivery.passed
                if delivered:
                    last_success = result
                    fresh_trajectory = _delivered_target_trajectory(
                        run_record.run_id,
                        result,
                    )
                    target_trajectory.extend(fresh_trajectory)
                    for ref in _target_run_refs(fresh_trajectory):
                        if ref not in run_record.target_run_refs:
                            run_record.target_run_refs.append(ref)
                elif result.stage == "abstained":
                    # Abstention is intentional convergence, not a failed hand-off.
                    if last_success is not None:
                        result = _as_converged(last_success, result)
                    break
                elif result.stage == "rejected":
                    # An explicit human rejection is authority to stop this run.
                    if last_success is not None:
                        result = _as_partial(last_success, result)
                    break
                elif _has_next_loop(loop_i, self.recursion.max_loops):
                    # A failed hand-off or delivery/commit gate rejects THIS candidate,
                    # not the recursive run. The materialized LoopOutcome above is fed
                    # to the next Orchestrator, and _run_loop has already rolled back
                    # any uncommitted candidate changes.
                    continue
                elif last_success is not None:
                    # Recovery budget exhausted after at least one verified version.
                    result = _as_partial(last_success, result)
                    break

                if not _has_next_loop(loop_i, self.recursion.max_loops):
                    break
            final = last_success or result or PipelineResult("abstained")
            if result is not None and result.stage in {"partially_delivered", "converged"}:
                final = result
            finalized = await self._finalize(wt, final, manifest, all_results)
            run_record.status = finalized.stage
            run_record.final_commit = (
                finalized.merged_commit
                or (finalized.version.verified_commit if finalized.version else "")
                or ""
            )
            record_store.finish(run_record)
            return finalized
        except Exception:
            run_record.status = "failed"
            record_store.finish(run_record)
            raise
        finally:
            await wt.__aexit__(None, None, None)

    async def _preflight(self, wt: WorktreeSession, intent: str, traj: int) -> dict[str, Any]:
        """A Run Manifest: what this Recursive Run is operating on and with."""
        # a worktree is created from a COMMIT, so it excludes uncommitted changes.
        status = await _run_git("status", "--porcelain", cwd=self.cwd)
        return {
            "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f"),
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
            "target_trajectory_records": traj,
        }

    @traceable(name="improve.loop", run_type="chain")
    async def _run_loop(
        self,
        wt: WorktreeSession,
        intent: str,
        target_trajectory: list[dict[str, Any]],
        loop_history: list[ReforgeLoopRecord],
        run_manifest: dict[str, Any],
        history_index: ImprovementHistoryIndex,
        loop_i: int,
    ) -> PipelineResult:
        """One Improvement Loop on the shared worktree: analyze -> gate -> approve
        -> Writer/Reviewer -> Deliverer. A delivered loop makes exactly one commit;
        a failed loop rolls the worktree back to loop_base."""
        loop_base = await wt.head()  # this loop's base commit
        wt_path = str(wt.path)

        # 1. Orchestrator analyzes the current state, given what earlier loops tried.
        code_index = CodeIndex(
            root=wt_path,
            db_path=Path(self.cwd) / ".agentreforge" / "orchestrator_code_index.db",
        )
        code_index.rebuild()
        context = OrchestratorContextBuilder(wt_path).build(
            intent=intent,
            target_trajectory=target_trajectory,
            previous_reforge_loops=loop_history,
            run_manifest=run_manifest | {"current_loop": loop_i, "loop_base": loop_base},
        )
        orch = Orchestrator(
            self.client,
            self.registry,
            wt_path,
            code_index=code_index,
            history_index=history_index,
        )
        try:
            proposal = await orch.analyze(
                intent,
                target_trajectory,
                loop_history,
                context=context,
            )
            # PIPELINE is the authority on DAG validity (the tool is advisory).
            proposal = await self._enforce_valid_proposal(
                orch,
                proposal,
                loop_history=loop_history,
            )
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
        if actual_gate.decision == "deny":
            result = PipelineResult(
                stage="rejected_policy", proposal=proposal, gate=actual_gate,
                approval=approval, frozen=frozen, outcome=outcome,
                diff=await wt.diff_since(loop_base), loop=loop_i,
                error="; ".join(actual_gate.reasons),
            )
            result.report_path = self._write_report(result)
            await wt.reset_hard(loop_base)
            return result
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
        while _delivery_is_writer_repairable(
            delivery,
            repairs=repairs,
            max_repairs=self.recursion.max_repairs,
        ):
            repairs += 1
            repair_outcome = await writer.repair(
                worktree=wt,
                instruction=_repair_instruction(delivery),
                allowed_write_paths=proposal.allowed_write_paths,
            )
            for task_outcome in repair_outcome.task_outcomes:
                task_outcome.repair_iteration = repairs
            outcome = _combine_execution_outcomes(outcome, repair_outcome)
            if repair_outcome.blocker is not None or not repair_outcome.completed:
                break
            repaired_gate = policy_gate.evaluate_changes(
                proposal, await wt.changed_since(loop_base), self.policy
            )
            if repaired_gate.decision == "deny":
                delivery = Delivery(
                    passed=False,
                    delivery_gate_ok=False,
                    reasons=repaired_gate.reasons,
                )
                loop_diff = await wt.diff_since(loop_base)
                break
            if repaired_gate.decision == "needs_human":
                if self.governance == GovernanceMode.AUTONOMOUS:
                    delivery = Delivery(
                        passed=False,
                        delivery_gate_ok=False,
                        reasons=repaired_gate.reasons,
                    )
                    loop_diff = await wt.diff_since(loop_base)
                    break
                repair_approval = await self.approver(proposal, repaired_gate)
                if not repair_approval.approved:
                    delivery = Delivery(
                        passed=False,
                        delivery_gate_ok=False,
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
        if (
            self.auto_merge
            and delivered
            and result.stage in {"delivered", "converged"}
            and result.proposal is not None
        ):
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
        reports_dir = Path(self.cwd) / ".agentreforge" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# Recursive Run {manifest['run_id']}", "", "## Manifest"]
        lines += [f"- **{k}:** {v}" for k, v in manifest.items()]
        lines += ["", "## Loops"]
        for r in all_results:
            commit = (r.version.verified_commit or "")[:12] if r.version else "—"
            reasons = "; ".join(r.delivery.reasons) if r.delivery else (r.error or "")
            lines += [
                "",
                f"### Loop {r.loop} — {r.stage}",
                f"- **Objective:** {r.proposal.summary if r.proposal else '(none)'}",
                f"- **Commit:** {commit}",
                f"- **Result:** {reasons or r.stage}",
            ]
            changed_files = _changed_files_from_diff(r.diff)
            lines.append(
                "- **Changed files:** "
                + (", ".join(f"`{path}`" for path in changed_files) or "(none)")
            )
            if r.outcome and r.outcome.task_outcomes:
                lines.append("- **Tasks:**")
                task_specs = {
                    task.id: task
                    for task in (r.proposal.tasks if r.proposal else [])
                }
                for outcome in r.outcome.task_outcomes:
                    spec = task_specs.get(outcome.task_id)
                    description = spec.description if spec else outcome.writer_summary
                    description = " ".join(description.split())
                    if len(description) > 240:
                        description = description[:237] + "..."
                    lines.append(
                        f"  - `{outcome.task_id}` [{outcome.status}]: {description}"
                    )
        if final.stage == "partially_delivered":
            lines += [
                "",
                "## Terminal state",
                f"- **Last successful loop:** {final.loop}",
                f"- **Terminal loop:** {final.terminal_loop}",
                f"- **Terminal stage:** {final.terminal_stage}",
            ]
            if final.terminal_error:
                lines.append(f"- **Terminal error:** {final.terminal_error}")
        elif final.stage == "converged":
            lines += [
                "",
                "## Convergence",
                f"- **Last delivered loop:** {final.loop}",
                f"- **Stop loop:** {final.terminal_loop}",
                "- **Reason:** Orchestrator abstained; no further evidence-backed "
                "improvement was justified.",
            ]
        if final.merged_commit:
            lines.append(f"\n**Merged to main:** {final.merged_commit[:12]}")
        elif final.version:
            lines.append(f"\n**Candidate branch kept:** `{final.version.branch}`")
        path = reports_dir / f"{manifest['run_id']}_run.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    async def _enforce_valid_proposal(
        self,
        orch: Orchestrator,
        proposal: ImprovementProposal,
        *,
        loop_history: list[ReforgeLoopRecord] | None = None,
        max_tries: int = 3,
    ) -> ImprovementProposal:
        """Hard-validate both the task graph and the executable acceptance contract."""
        # Abstention is a valid terminal decision, not an implementation plan. Requiring
        # a dummy Task or delivery command would encourage the model to invent work.
        if proposal.decision == "abstain":
            return proposal
        tries = 0
        while tries <= max_tries:
            dag = validate_plan([t.model_dump() for t in proposal.tasks])
            acceptance = validate_acceptance(proposal)
            problems = []
            if not dag.valid:
                problems.append(f"invalid task DAG — {dag.summary()}")
            if not acceptance.valid:
                problems.append(f"invalid acceptance contract — {acceptance.summary()}")
            analysis_problems = _analysis_problems(proposal)
            if analysis_problems:
                problems.append(
                    "incomplete orchestrator analysis — " + "; ".join(analysis_problems)
                )
            repeated_attempts = _negative_attempt_problems(
                proposal,
                loop_history or [],
            )
            if repeated_attempts:
                problems.append(
                    "repeated failed attempt — " + "; ".join(repeated_attempts)
                )
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
        delivery = await self.delivery_coordinator.deliver(
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
        reports_dir = Path(self.cwd) / ".agentreforge" / "reports"
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


def _target_run_refs(records: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for record in records:
        run_id = str(record.get("run_id") or record.get("session_id") or "")
        if run_id and run_id not in refs:
            refs.append(run_id)
    return refs


def _analysis_problems(proposal: ImprovementProposal) -> list[str]:
    """Validate the single-Candidate/single-Task Orchestrator hand-off."""

    analysis = proposal.analysis
    problems = []
    candidates = {candidate.name: candidate for candidate in analysis.candidates}
    selected = analysis.selected_candidates
    if len(selected) != len(set(selected)):
        problems.append("selected_candidates must not contain duplicates")
    unknown = [name for name in selected if name not in candidates]
    if unknown:
        problems.append(
            "selected_candidates must name declared candidates: " + ", ".join(unknown)
        )
    if proposal.decision != "abstain" and len(selected) != 1:
        problems.append("a proceeding Loop must select exactly one Candidate")
    if proposal.decision != "abstain" and len(proposal.tasks) != 1:
        problems.append("a Loop must contain exactly one implementation Task")
    missing_owner = [
        task.id
        for task in proposal.tasks
        if task.candidate not in selected
    ]
    if missing_owner:
        problems.append(
            "every Task must name an owning selected Candidate: " + ", ".join(missing_owner)
        )
    return problems


def _combine_execution_outcomes(
    original: ExecutionOutcome,
    repair: ExecutionOutcome,
) -> ExecutionOutcome:
    """Preserve the original task history while appending each delivery repair."""

    return ExecutionOutcome(
        completed=original.completed and repair.completed,
        task_outcomes=[*original.task_outcomes, *repair.task_outcomes],
        blocker=repair.blocker or original.blocker,
        final_commit=repair.final_commit or original.final_commit,
        diff=repair.diff or original.diff,
    )


def _materialize_loop_record(
    run_id: str,
    loop_base: str,
    result: PipelineResult,
) -> ReforgeLoopRecord:
    """Create the next-loop feedback and durable Reforge audit deterministically."""

    proposal = result.proposal
    components: list[ComponentRecord] = []
    if proposal is not None:
        components.append(
            ComponentRecord(
                component="orchestrator",
                status=proposal.decision,
                summary=proposal.summary,
                details={
                    "proposal": proposal.model_dump(mode="json"),
                    "problem_statement": proposal.problem_statement,
                    "analysis": proposal.analysis.model_dump(mode="json"),
                    "tasks": [task.model_dump(mode="json") for task in proposal.tasks],
                    "evidence": [item.model_dump(mode="json") for item in proposal.evidence],
                    "allowed_write_paths": proposal.allowed_write_paths,
                },
            )
        )
    if result.gate is not None:
        components.append(
            ComponentRecord(
                component="policy",
                status=result.gate.decision,
                summary="; ".join(result.gate.reasons),
                details=jsonable(result.gate),
            )
        )
    if result.outcome is not None:
        task_details = []
        reviews = []
        for task in result.outcome.task_outcomes:
            task_details.append(
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "rounds": task.rounds,
                    "phase": task.phase,
                    "repair_iteration": task.repair_iteration,
                    "writer_summary": task.writer_summary,
                    "attempts": task.attempts,
                    "commit": task.commit,
                }
            )
            reviews.append(
                {
                    "task_id": task.task_id,
                    "verdict": task.review.verdict,
                    "summary": task.review.summary,
                    "findings": [
                        finding.model_dump(mode="json") for finding in task.review.findings
                    ],
                }
            )
        components.extend(
            [
                ComponentRecord(
                    component="writer",
                    status="completed" if result.outcome.completed else "incomplete",
                    summary=f"{len(task_details)} task outcome(s)",
                    details={"tasks": task_details, "blocker": jsonable(result.outcome.blocker)},
                ),
                ComponentRecord(
                    component="reviewer",
                    status=(
                        "accepted"
                        if reviews and all(item["verdict"] == "accept" for item in reviews)
                        else "mixed"
                    ),
                    summary=f"{len(reviews)} task review(s)",
                    details={"task_reviews": reviews},
                ),
            ]
        )
    if result.delivery is not None:
        components.extend(
            [
                ComponentRecord(
                    component="acceptance_runner",
                    status="passed" if result.delivery.delivery_gate_ok else "failed",
                    summary="; ".join(result.delivery.acceptance_failures),
                    details={
                        "runs": jsonable(result.delivery.runs),
                        "scenario_runs": jsonable(result.delivery.scenario_runs),
                        "failures": result.delivery.acceptance_failures,
                    },
                ),
                ComponentRecord(
                    component="deliverer",
                    status="accepted" if result.delivery.goal_accepted else "rejected",
                    summary=result.delivery.goal_review,
                    details={"goal_review": result.delivery.goal_review},
                ),
                ComponentRecord(
                    component="delivery_coordinator",
                    status="accepted" if result.delivery.passed else "rejected",
                    summary="; ".join(result.delivery.reasons),
                    details={
                        "delivery_gate_ok": result.delivery.delivery_gate_ok,
                        "goal_accepted": result.delivery.goal_accepted,
                        "handoff_failed": result.delivery.handoff_failed,
                        "failure_kind": result.delivery.failure_kind,
                        "integrity_ok": result.delivery.integrity_ok,
                        "reasons": result.delivery.reasons,
                    },
                ),
            ]
        )

    diagnosis = proposal.analysis.model_dump(mode="json") if proposal else {}
    achievements = []
    remaining = []
    delivered = result.delivery is not None and result.delivery.passed
    if proposal is not None:
        if delivered:
            selected = set(proposal.analysis.selected_candidates)
            achievements = [
                (
                    f"{candidate.name}: {candidate.expected_capability_delta}"
                    if candidate.expected_capability_delta
                    else candidate.name
                )
                for candidate in proposal.analysis.candidates
                if candidate.name in selected
            ]
        else:
            remaining = [
                finding.capability_gap
                for finding in proposal.analysis.findings
                if finding.capability_gap
            ]
    commit = (
        result.version.verified_commit
        if result.version and result.version.verified_commit
        else ""
    )
    return ReforgeLoopRecord(
        run_id=run_id,
        loop_id=f"{run_id}/loop_{result.loop}",
        loop=result.loop,
        base_commit=loop_base,
        stage=result.stage,
        components=components,
        proposal_summary=proposal.summary if proposal else "",
        diagnosis=diagnosis,
        changed_paths=_changed_paths_from_diff(result.diff),
        commit=commit,
        completed=delivered,
        achievements=achievements,
        remaining_gaps=remaining,
        failure_kind=(
            result.delivery.failure_kind
            if result.delivery is not None and not result.delivery.passed
            else "none"
        ),
        attempt_fingerprint=(
            _proposal_attempt_fingerprint(proposal) if proposal is not None else ""
        ),
        error=_loop_failure_summary(result),
    )


def _proposal_attempt_fingerprint(proposal: ImprovementProposal) -> str:
    """Stable identity for one Candidate plus its frozen verification strategy."""

    payload = {
        "candidates": sorted(
            name.strip().casefold()
            for name in proposal.analysis.selected_candidates
        ),
        "delivery_run": sorted(proposal.delivery_run),
        "scenarios": sorted(
            (
                tuple(scenario.command),
                scenario.requires_trajectory,
                tuple(
                    sorted(
                        (condition.name, condition.state)
                        for condition in scenario.executable_conditions
                    )
                ),
                tuple(sorted(scenario.expected_behaviors)),
                tuple(sorted(scenario.forbidden_behaviors)),
            )
            for scenario in proposal.delivery_scenarios
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _tag_baseline_trajectory(
    records: list[dict[str, Any]],
    *,
    commit: str,
) -> list[dict[str, Any]]:
    """Bind pre-run observations to the commit they actually describe."""

    return [
        {
            **record,
            "evidence_source": str(record.get("evidence_source") or "baseline"),
            "target_commit": str(record.get("target_commit") or commit),
        }
        for record in records
    ]


def _delivered_target_trajectory(
    recursive_run_id: str,
    result: PipelineResult,
) -> list[dict[str, Any]]:
    """Promote delivered Scenario evidence into the next Loop's target history."""

    if (
        result.delivery is None
        or not result.delivery.passed
        or result.version is None
        or not result.version.verified_commit
    ):
        return []
    records: list[dict[str, Any]] = []
    commit = result.version.verified_commit
    for scenario in result.delivery.scenario_runs:
        target_run_id = (
            f"{recursive_run_id}:loop:{result.loop}:scenario:{scenario.scenario_id}"
        )
        records.append(
            {
                "event_id": f"{target_run_id}:event:0",
                "trajectory_kind": "target_agent",
                "evidence_source": "delivered_scenario",
                "target_commit": commit,
                "run_id": target_run_id,
                "session_id": target_run_id,
                "type": "target_run_started",
                "task_prompt": scenario.prompt,
                "scenario_id": scenario.scenario_id,
            }
        )
        saw_done = False
        for index, event in enumerate(scenario.trajectory, start=1):
            normalized = {
                **event,
                "event_id": f"{target_run_id}:event:{index}",
                "trajectory_kind": "target_agent",
                "evidence_source": "delivered_scenario",
                "target_commit": commit,
                "run_id": target_run_id,
                "session_id": target_run_id,
                "scenario_id": scenario.scenario_id,
            }
            if normalized.get("type") == "done":
                saw_done = True
                normalized.setdefault("outcome", "completed")
                normalized.setdefault("final_response", scenario.output)
            records.append(normalized)
        if not saw_done:
            records.append(
                {
                    "event_id": (
                        f"{target_run_id}:event:{len(scenario.trajectory) + 1}"
                    ),
                    "trajectory_kind": "target_agent",
                    "evidence_source": "delivered_scenario",
                    "target_commit": commit,
                    "run_id": target_run_id,
                    "session_id": target_run_id,
                    "type": "done",
                    "outcome": "completed",
                    "final_response": scenario.output,
                    "scenario_id": scenario.scenario_id,
                }
            )
    return records


def _negative_attempt_problems(
    proposal: ImprovementProposal,
    loop_history: list[ReforgeLoopRecord],
) -> list[str]:
    """Reject an unchanged verification strategy after a recorded failed attempt."""

    fingerprint = _proposal_attempt_fingerprint(proposal)
    selected = set(proposal.analysis.selected_candidates)
    problems: list[str] = []
    for record in loop_history:
        if record.completed or record.failure_kind == "none":
            continue
        previous_selected = set(
            str(value)
            for value in (record.diagnosis.get("selected_candidates") or [])
        )
        same_candidate = bool(selected & previous_selected)
        if record.attempt_fingerprint == fingerprint:
            problems.append(
                f"{record.loop_id} already tried the same Candidate and verification "
                f"strategy ({record.failure_kind})"
            )
            continue
        missing_trajectory = (
            record.failure_kind == "verification_gap"
            and "trajectory" in record.error.casefold()
        )
        if (
            same_candidate
            and missing_trajectory
            and any(scenario.requires_trajectory for scenario in proposal.delivery_scenarios)
        ):
            problems.append(
                f"{record.loop_id} proved required trajectory unavailable for this "
                "Candidate; use an adapter/observable scenario or abstain"
            )
        failed_safety = {
            safety
            for safety in _record_required_safety(record)
            if f"adapter:safety:{safety}" in record.error
        }
        repeated_safety = failed_safety & _proposal_required_safety(proposal)
        if same_candidate and repeated_safety:
            problems.append(
                f"{record.loop_id} already failed the same safety contract for this "
                f"Candidate: {', '.join(sorted(repeated_safety))}; remove the unrelated "
                "requirement, select the safety capability itself, or abstain"
            )
    return problems


def _proposal_required_safety(proposal: ImprovementProposal) -> set[str]:
    return {
        str(safety)
        for task in proposal.tasks
        for safety in task.required_safety_properties
    }


def _record_required_safety(record: ReforgeLoopRecord) -> set[str]:
    for component in record.components:
        if component.component != "orchestrator":
            continue
        proposal = component.details.get("proposal")
        if not isinstance(proposal, dict):
            continue
        tasks = proposal.get("tasks")
        if not isinstance(tasks, list):
            continue
        return {
            str(safety)
            for task in tasks
            if isinstance(task, dict)
            for safety in (task.get("required_safety_properties") or [])
        }
    return set()


def _has_next_loop(loop_i: int, max_loops: int) -> bool:
    """Whether the recursive-run budget permits another Orchestrator loop."""

    return loop_i + 1 < max_loops


def _loop_failure_summary(result: PipelineResult) -> str:
    """Put the actionable failed-gate reason in the next loop's bounded context."""

    if result.error:
        return result.error
    if result.blocker is not None:
        findings = [
            finding.description
            for finding in result.blocker.unresolved_findings
            if finding.description
        ]
        detail = "; ".join(findings) or "Writer/Reviewer hand-off did not converge"
        return f"task {result.blocker.task_id} blocked: {detail}"
    if result.delivery is not None and not result.delivery.passed:
        return "; ".join(result.delivery.reasons) or "delivery/commit gate rejected candidate"
    if result.gate is not None and result.gate.decision != "proceed":
        return "; ".join(result.gate.reasons)
    return ""


def _changed_paths_from_diff(diff: str) -> list[str]:
    paths: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("+++ b/"):
            continue
        path = line.removeprefix("+++ b/")
        if path != "/dev/null" and path not in paths:
            paths.append(path)
    return paths


def _as_partial(success: PipelineResult, terminal: PipelineResult) -> PipelineResult:
    """Return the last verified version without hiding the later loop failure."""
    success.stage = "partially_delivered"
    success.terminal_stage = terminal.stage
    success.terminal_loop = terminal.loop
    success.terminal_report_path = terminal.report_path
    success.terminal_error = terminal.error or (
        "; ".join(terminal.delivery.reasons) if terminal.delivery else ""
    )
    return success


def _as_converged(success: PipelineResult, terminal: PipelineResult) -> PipelineResult:
    """Preserve the last delivered commit when the next Loop intentionally abstains."""

    success.stage = "converged"
    success.terminal_stage = terminal.stage
    success.terminal_loop = terminal.loop
    success.terminal_report_path = terminal.report_path
    success.terminal_error = terminal.error
    return success


def _repair_instruction(delivery: Delivery) -> str:
    """Turn a rejected delivery into a concrete repair brief for the Writer."""
    lines = [
        f"Delivery classified this as {delivery.failure_kind}.",
        "The candidate did not pass delivery. Fix the PRODUCT CODE so it does. "
        "Here are the deterministic runner failures and high-level review findings. "
        "Repair the intended behavior exercised by the failing criterion; do not game "
        "an output assertion by hardcoding its missing marker into unrelated output:",
    ]
    if delivery.acceptance_failures:
        lines.append("\nDeterministic AcceptanceRunner failures:")
        lines.extend(f"- {failure}" for failure in delivery.acceptance_failures)
    # Include exit-0 runs too: an output assertion can fail even when the command
    # itself succeeds.
    for r in delivery.runs:
        tail = r.output[-800:] if r.output else "(no output)"
        lines.append(f"\n$ {r.command} (exit {r.exit_code})\n{tail}")
    for scenario in delivery.scenario_runs:
        tail = scenario.output[-800:] if scenario.output else "(no output)"
        lines.append(
            f"\nScenario {scenario.scenario_id} argv={scenario.command} "
            f"(exit {scenario.exit_code})\n{tail}"
        )
    # the Deliverer's high-level proposal-vs-diff review (why it rejected).
    if not delivery.goal_accepted and delivery.goal_review:
        lines.append(f"\nDeliverer goal review:\n{delivery.goal_review}")
    return "\n".join(lines)


def _delivery_is_writer_repairable(
    delivery: Delivery,
    *,
    repairs: int,
    max_repairs: int,
) -> bool:
    """Only observed product implementation defects may re-enter Writer."""

    return (
        not delivery.passed
        and delivery.integrity_ok
        and not delivery.handoff_failed
        and delivery.failure_kind == "implementation_defect"
        and repairs < max_repairs
    )


def _changed_files_from_diff(diff: str) -> list[str]:
    """Extract authoritative changed paths from a unified Git diff."""

    paths: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("diff --git a/"):
            continue
        _left, separator, right = line.removeprefix("diff --git a/").partition(" b/")
        if separator and right not in paths:
            paths.append(right)
    return paths


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
    if result.delivery and result.delivery.scenario_runs:
        lines += ["", "## End-to-end delivery scenarios"]
        for run in result.delivery.scenario_runs:
            lines += [
                "",
                f"### `{run.scenario_id}` — exit {run.exit_code}",
                f"- **Prompt:** {run.prompt}",
                f"- **Command:** `{run.command}`",
                f"- **Changed fixture files:** {', '.join(run.changed_files) or '(none)'}",
                f"- **Trajectory available:** {run.trajectory_available}",
                "```text",
                run.output[-1500:] if run.output else "(no output)",
                "```",
            ]
    if result.delivery and result.delivery.goal_review:
        lines += ["", "### Goal realization review", result.delivery.goal_review]

    if result.diff:
        diff = result.diff if len(result.diff) < 8000 else result.diff[:8000] + "\n...(truncated)"
        lines += ["", "## Diff", "```diff", diff, "```"]

    return "\n".join(lines) + "\n"
