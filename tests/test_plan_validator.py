from __future__ import annotations

from agentreforge.improve.plan_validator import validate_plan


def test_plan_validator_accepts_a_dag():
    result = validate_plan(
        [
            {"id": "write", "dependencies": []},
            {"id": "test", "dependencies": ["write"]},
        ]
    )

    assert result.valid


def test_plan_validator_rejects_cycle_and_unknown_dependency():
    cycle = validate_plan(
        [
            {"id": "a", "dependencies": ["b"]},
            {"id": "b", "dependencies": ["a"]},
        ]
    )
    unknown = validate_plan([{"id": "a", "dependencies": ["missing"]}])

    assert not cycle.valid
    assert not unknown.valid
