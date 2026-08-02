from __future__ import annotations

import asyncio

import pytest
from conftest import make_proposal

from agentreforge.improve.orchestrator import ORCHESTRATOR_PROMPT, Orchestrator
from agentreforge.tools.registry import ToolRegistry


class _UnusedClient:
    pass


def test_orchestrator_distinguishes_backlog_dependencies_and_execution_contract():
    assert "dictionary keyed by a stable" in ORCHESTRATOR_PROMPT
    assert "selected_change_contract" in ORCHESTRATOR_PROMPT
    assert "Contract is one CHANGE UNIT" in ORCHESTRATOR_PROMPT


def test_orchestrator_treats_safety_checks_as_explicit_not_heuristic():
    assert "verified_safety_properties" in ORCHESTRATOR_PROMPT
    assert "Safety probes are system-owned" in ORCHESTRATOR_PROMPT
    assert "execution tool's target adapter executes the actual traversal" in ORCHESTRATOR_PROMPT
    assert "Never encode a condition that rewards the" in ORCHESTRATOR_PROMPT
    assert "safety violation" in ORCHESTRATOR_PROMPT
    assert "literal `..` escape" not in ORCHESTRATOR_PROMPT


def test_orchestrator_has_single_task_achievement_and_anti_reward_hacking_rules():
    assert "Build an Achievement Ledger" in ORCHESTRATOR_PROMPT
    assert "exactly ONE backlog Candidate" in ORCHESTRATOR_PROMPT
    assert "exactly ONE" in ORCHESTRATOR_PROMPT
    assert "`selected_change_contract`" in ORCHESTRATOR_PROMPT
    assert "merely to make AgentReforge's gate or evaluator pass" in ORCHESTRATOR_PROMPT
    assert "The recursion limit is a ceiling, never a target" in ORCHESTRATOR_PROMPT


def test_orchestrator_uses_advisory_scorecards_and_top_two_review():
    assert "PROBLEM TRIAGE before considering solution cost" in ORCHESTRATOR_PROMPT
    assert "failed_verification" in ORCHESTRATOR_PROMPT
    assert "NEVER add them into a mechanical total" in ORCHESTRATOR_PROMPT
    assert "`preliminary_ranking` for every viable Candidate" in ORCHESTRATOR_PROMPT
    assert "`top_two_comparison`" in ORCHESTRATOR_PROMPT
    assert "literal option `DEFER`" in ORCHESTRATOR_PROMPT
    assert "would the baseline likely pass" in ORCHESTRATOR_PROMPT
    assert "does not compute or override your ranking" in ORCHESTRATOR_PROMPT
    assert "TARGET AGENT REPOSITORY from a TASK WORKSPACE" in ORCHESTRATOR_PROMPT
    assert "`target_commit` identifies the agent version" in ORCHESTRATOR_PROMPT
    assert "terminal `failed_verification`" in ORCHESTRATOR_PROMPT
    assert "both likely pass" in ORCHESTRATOR_PROMPT
    assert "stochastic run could easily reverse" in ORCHESTRATOR_PROMPT


def test_proposal_schema_repair_can_recover_on_second_retry():
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), ".")
    valid = make_proposal().model_dump_json()
    repairs = iter(["{}", valid])
    seen: list[tuple[str, str]] = []

    async def repair(bad_output: str, error: str) -> str:
        seen.append((bad_output, error))
        return next(repairs)

    orchestrator._repair = repair  # type: ignore[method-assign]

    proposal = asyncio.run(orchestrator._parse_proposal_with_repair("not json"))

    assert proposal.summary == "Improve a component"
    assert len(seen) == 2
    assert seen[0][0] == "not json"
    assert seen[1][0] == "{}"


def test_proposal_schema_repair_stops_at_budget():
    orchestrator = Orchestrator(_UnusedClient(), ToolRegistry(), ".")
    repair_calls = 0

    async def repair(_bad_output: str, _error: str) -> str:
        nonlocal repair_calls
        repair_calls += 1
        return "{}"

    orchestrator._repair = repair  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="invalid JSON"):
        asyncio.run(
            orchestrator._parse_proposal_with_repair("not json", max_repairs=2)
        )

    assert repair_calls == 2
