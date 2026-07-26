from __future__ import annotations

from metaimprove.improve.models import (
    AcceptanceCriterion,
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
                description="Implement the behavior",
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
                command="python -m pytest tests/test_agent.py",
            )
        ],
    }
    values.update(overrides)
    return ImprovementProposal(**values)
