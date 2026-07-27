from __future__ import annotations

from metaimprove.improve.models import (
    AcceptanceCriterion,
    ContractClause,
    Evidence,
    ImprovementProposal,
    ImprovementTask,
)


def make_proposal(**overrides) -> ImprovementProposal:
    values = {
        "summary": "Improve a component",
        "problem_statement": "The current behavior is incomplete.",
        "evidence": [
            Evidence(source_type="code", reference="src/agent.py:1", observation="missing case")
        ],
        "goals": ["handle the case"],
        "affected_components": ["src/agent.py"],
        "allowed_write_paths": ["src/agent.py", "tests/"],
        "tasks": [
            ImprovementTask(
                id="implement",
                candidate="focused fix",
                description="Implement the behavior",
                rationale="The missing behavior causes the observed failure.",
                capability_change="The target agent handles the missing case reliably.",
                required_behaviors=[
                    ContractClause(id="RB1", description="Handle the missing case.")
                ],
                invariants=[
                    ContractClause(id="INV1", description="Preserve existing behavior.")
                ],
                affected_components=["src/agent.py"],
                reviewer_focus=["Confirm the missing case is handled without a shortcut."],
                acceptance_criteria_ids=["ac1"],
            )
        ],
        "benefit": 4,
        "risk": 2,
        "effort": 2,
        "confidence": 0.9,
        "decision": "proceed",
        "decision_reason": "Evidence supports a focused fix.",
        "acceptance_criteria": [
            AcceptanceCriterion(
                id="ac1",
                description="The focused test passes",
                mode="red_green",
                check_type="integration",
                command="python3 -m pytest tests/test_agent.py",
            ),
        ],
    }
    values.update(overrides)
    return ImprovementProposal(**values)
