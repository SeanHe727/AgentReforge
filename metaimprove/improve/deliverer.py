"""High-level proposal realization review over the complete improvement diff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..llm.collect import collect_text
from ..observability import traceable
from ..types import Message
from .models import ImprovementProposal

_DIFF_CAP = 12_000

DELIVERER_PROMPT = """You are the Deliverer: the final HIGH-LEVEL change reviewer.

The per-task Reviewer has already checked local implementation details. A separate
deterministic AcceptanceRunner has already executed the frozen acceptance commands.
Do NOT act as another test agent and do NOT invent runtime claims.

You receive the frozen Orchestrator goal/diagnosis/selected Candidates and the FULL
DIFF for this Improvement Batch. Judge only whether the combined change realizes the
proposal:

1. Candidate realization — does the diff implement EVERY selected Candidate and its
   causal mechanism, rather than merely adding names, prompts, stubs, or unreachable code?
2. Integration — are new interfaces wired into the real execution path, with no
   missing cross-component changes visible in the diff?
3. Batch compatibility — can the selected Candidates coexist without an obvious
   cross-cutting regression, interface conflict, or scope violation?
4. Evidence discipline — distinguish what the diff proves from what would require a
   later target-agent capability evaluation. Do not claim benchmark/output quality.

Do not repeat per-Task review and do not judge whether commands mutated the repository;
the independent Reviewer and deterministic DeliveryCoordinator already own those checks.

Reply with:
GOAL: ACHIEVED or GOAL: NOT ACHIEVED
REASON: <concise causal assessment grounded in the diff>
PROJECT CONCERNS: none, or concrete blocking concerns
VERDICT: ACCEPT or VERDICT: REJECT

Accept only when the full diff coherently realizes the Orchestrator's selected goal
and contains no blocking project-level concern."""


@dataclass
class GoalReview:
    accepted: bool
    text: str


class Deliverer:
    """Pure LLM reviewer: proposal versus whole-loop diff, with no command execution."""

    def __init__(self, *, client: Any):
        self.client = client

    @traceable(name="deliverer.review", run_type="chain")
    async def review(
        self,
        proposal: ImprovementProposal,
        *,
        loop_diff: str,
    ) -> GoalReview:
        if not loop_diff.strip():
            return GoalReview(
                accepted=False,
                text="GOAL: NOT ACHIEVED\nREASON: candidate has no diff\nVERDICT: REJECT",
            )

        diff = loop_diff[:_DIFF_CAP]
        if len(loop_diff) > _DIFF_CAP:
            diff += "\n...(truncated)"
        text = await collect_text(
            self.client,
            [Message(role="user", content=goal_review_message(proposal, diff))],
            system_prompt=DELIVERER_PROMPT,
        )
        upper = text.upper()
        accepted = (
            "GOAL: ACHIEVED" in upper
            and "GOAL: NOT ACHIEVED" not in upper
            and "VERDICT: ACCEPT" in upper
            and "VERDICT: REJECT" not in upper
        )
        return GoalReview(accepted=accepted, text=text)


def goal_review_message(proposal: ImprovementProposal, loop_diff: str) -> str:
    analysis = proposal.analysis
    goals = "\n".join(f"- {goal}" for goal in proposal.goals) or "- (none)"
    selected = "\n".join(f"- {name}" for name in analysis.selected_candidates) or "- (none)"
    tasks = "\n".join(
        f"- [{task.id}] Candidate={task.candidate}: {task.description}"
        for task in proposal.tasks
    )
    requirements = (
        "\n".join(f"- {item}" for item in proposal.delivery_checklist) or "- (none)"
    )
    return (
        f"Proposal summary:\n{proposal.summary}\n\n"
        f"Problem:\n{proposal.problem_statement}\n\n"
        f"Goals:\n{goals}\n\n"
        f"Selected Candidates:\n{selected}\n\n"
        f"Batch packing rationale:\n{analysis.packing_reason}\n\n"
        f"Causal mechanism:\n{analysis.causal_mechanism}\n\n"
        f"Expected capability delta:\n{analysis.expected_capability_delta}\n\n"
        f"Planned tasks:\n{tasks}\n\n"
        f"Declared system-level requirements:\n{requirements}\n\n"
        f"Full loop diff:\n{loop_diff}"
    )
