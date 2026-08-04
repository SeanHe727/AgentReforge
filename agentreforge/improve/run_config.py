"""The two independent dials the orchestrator exposes: detail level + governance.

detail level tunes ONLY the agents' effort (worker tool-turns, Writer<->Reviewer
rounds) and the ceiling on generated-test rigor — nothing about which stages run.
governance is the separate HITL axis (autonomous vs supervised). Keeping them
apart means "how hard the agents try" and "does a human approve" never get
conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DetailLevel(StrEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class GovernanceMode(StrEnum):
    AUTONOMOUS = "autonomous"  # policy decides; no human prompt
    SUPERVISED = "supervised"  # policy first, human approves at checkpoints


@dataclass(frozen=True)
class RecursionPolicy:
    # bounded recursive improvement: cycles across verified versions, plus the
    # verify-driven Writer repair attempts within one cycle. Defaults = single
    # cycle, one repair — no runaway self-improvement loop.
    max_loops: int = 1
    max_repairs: int = 1


@dataclass(frozen=True)
class RunProfile:
    # One Review Cycle is exactly one Writer Attempt followed by one Reviewer Pass.
    # Turn budgets are per invocation and reset on the next Review Cycle.
    max_review_cycles: int
    writer_plan_turns: int
    writer_attempt_turns: int
    reviewer_pass_turns: int
    test_ceiling: str  # highest test_level the analyzer may assign: basic|focused|full

    @property
    def max_rounds(self) -> int:
        """Compatibility name for older reports and callers."""

        return self.max_review_cycles

    @property
    def max_task_turns(self) -> int:
        """Compatibility name; Writer and Reviewer no longer share this budget."""

        return self.writer_attempt_turns


_PROFILES: dict[DetailLevel, RunProfile] = {
    DetailLevel.QUICK: RunProfile(
        max_review_cycles=2,
        writer_plan_turns=3,
        writer_attempt_turns=8,
        reviewer_pass_turns=4,
        test_ceiling="basic",
    ),
    DetailLevel.STANDARD: RunProfile(
        max_review_cycles=3,
        writer_plan_turns=4,
        writer_attempt_turns=12,
        reviewer_pass_turns=6,
        test_ceiling="focused",
    ),
    DetailLevel.DEEP: RunProfile(
        max_review_cycles=4,
        writer_plan_turns=6,
        writer_attempt_turns=18,
        reviewer_pass_turns=9,
        test_ceiling="full",
    ),
}

# rigor order used to clamp a criterion's test_level down to the profile ceiling.
_TEST_ORDER = {"basic": 0, "focused": 1, "full": 2}


def profile_for(level: DetailLevel | str) -> RunProfile:
    """The effort + test-ceiling profile for a detail level."""
    return _PROFILES[DetailLevel(level)]


def clamp_test_level(level: str, ceiling: str) -> str:
    """Cap a criterion's test_level at the profile's ceiling (never above it)."""
    if _TEST_ORDER.get(level, 0) <= _TEST_ORDER.get(ceiling, 2):
        return level
    return ceiling
