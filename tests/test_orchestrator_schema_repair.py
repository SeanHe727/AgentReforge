from __future__ import annotations

import asyncio

import pytest
from conftest import make_proposal

from agentreforge.improve.orchestrator import ORCHESTRATOR_PROMPT, Orchestrator
from agentreforge.tools.registry import ToolRegistry


class _UnusedClient:
    pass


def test_orchestrator_distinguishes_candidate_and_task_dependencies():
    assert "Task `dependencies`" in ORCHESTRATOR_PROMPT
    assert "ONLY exact Task `id` values" in ORCHESTRATOR_PROMPT
    assert "Never put a Candidate" in ORCHESTRATOR_PROMPT


def test_orchestrator_requires_literal_traversal_in_safety_check():
    assert "relative `..` escape" in ORCHESTRATOR_PROMPT
    assert "stable blocked/error marker" in ORCHESTRATOR_PROMPT


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
