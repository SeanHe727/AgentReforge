"""Structural validation for an ImprovementProposal's delivery hand-off.

This module validates the shape of the hand-off, not whether the Orchestrator's
test design is semantically wise.  Relevance, sufficiency, and goal realization
belong to the Reviewer/Deliverer agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .models import ImprovementProposal


@dataclass
class AcceptanceValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(self.errors) if self.errors else "valid"


def validate_acceptance(proposal: ImprovementProposal) -> AcceptanceValidation:
    """Validate only the delivery envelope and its references.

    This is deliberately a coarse "fill in the blanks" contract. Task quality,
    safety relevance, executable availability, scenario sufficiency, and test
    design are agent-review concerns rather than schema gates.
    """
    errors: list[str] = []
    criteria = proposal.contract_acceptance_criteria()
    tasks = proposal.execution_tasks()
    delivery_run = proposal.contract_delivery_run()
    delivery_scenarios = proposal.contract_delivery_scenarios()
    criterion_ids = [criterion.id for criterion in criteria]
    known = set(criterion_ids)

    if len(criterion_ids) != len(known):
        errors.append("acceptance criterion ids must be unique")
    if not delivery_run and not delivery_scenarios:
        errors.append(
            "delivery must contain a minimal smoke command or frozen end-to-end scenario"
        )

    for task in tasks:
        unknown = sorted(set(task.acceptance_criteria_ids) - known)
        if unknown:
            errors.append(f"task {task.id!r} references unknown criteria: {', '.join(unknown)}")

    # Commands are frozen suggestions that the Deliverer may actively execute.
    # Availability and runtime meaning are facts to discover during Delivery.
    for index, command in enumerate(delivery_run, start=1):
        if not command.strip():
            errors.append(f"delivery command {index} must not be blank")

    scenario_ids = [scenario.id for scenario in delivery_scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        errors.append("delivery scenario ids must be unique")
    for scenario in delivery_scenarios:
        if not scenario.prompt.strip():
            errors.append(f"delivery scenario {scenario.id!r} requires a prompt")
        if not scenario.command:
            errors.append(f"delivery scenario {scenario.id!r} requires an argv command")
        elif not scenario.command[0].strip():
            errors.append(
                f"delivery scenario {scenario.id!r} requires a non-blank executable"
            )
        for rel_path in scenario.fixture_files:
            path = PurePosixPath(rel_path)
            if path.is_absolute() or ".." in path.parts:
                errors.append(
                    f"delivery scenario {scenario.id!r} has unsafe fixture path "
                    f"{rel_path!r}"
                )
        condition_states: dict[str, str] = {}
        for condition in scenario.executable_conditions:
            previous = condition_states.get(condition.name)
            if previous is None:
                condition_states[condition.name] = condition.state
            elif previous != condition.state:
                errors.append(
                    f"delivery scenario {scenario.id!r} declares conflicting "
                    f"executable conditions for {condition.name!r}"
                )
            else:
                errors.append(
                    f"delivery scenario {scenario.id!r} repeats executable "
                    f"condition {condition.name!r}"
                )

    return AcceptanceValidation(not errors, errors)
