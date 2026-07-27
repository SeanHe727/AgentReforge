"""Deterministic validation for an ImprovementProposal's definition of done."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ImprovementProposal, ImprovementTask


@dataclass
class AcceptanceValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(self.errors) if self.errors else "valid"


def validate_acceptance(proposal: ImprovementProposal) -> AcceptanceValidation:
    """Require executable checks plus complete shared task contracts."""
    errors: list[str] = []
    criteria = proposal.acceptance_criteria
    criterion_ids = [criterion.id for criterion in criteria]
    known = set(criterion_ids)

    if not proposal.allowed_write_paths:
        errors.append("allowed_write_paths must declare the approved write scope")
    if not criteria:
        errors.append("acceptance_criteria must not be empty")
    if len(criterion_ids) != len(known):
        errors.append("acceptance criterion ids must be unique")

    assigned: set[str] = set()
    for task in proposal.tasks:
        _validate_task_contract(task, errors)
        if not task.acceptance_criteria_ids:
            errors.append(f"task {task.id!r} has no acceptance_criteria_ids")
        unknown = sorted(set(task.acceptance_criteria_ids) - known)
        if unknown:
            errors.append(f"task {task.id!r} references unknown criteria: {', '.join(unknown)}")
        assigned.update(task.acceptance_criteria_ids)

    unassigned = [c.id for c in criteria if c.required and c.id not in assigned]
    if unassigned:
        errors.append(f"required criteria are not assigned to a task: {', '.join(unassigned)}")

    for criterion in criteria:
        if criterion.verification == "command" and not criterion.command.strip():
            errors.append(f"command criterion {criterion.id!r} has no command")

    return AcceptanceValidation(not errors, errors)


def _validate_task_contract(task: ImprovementTask, errors: list[str]) -> None:
    """Reject thin hand-offs before a Writer is allowed to modify code."""

    required_text = {
        "description": task.description,
        "rationale": task.rationale,
        "capability_change": task.capability_change,
    }
    for field_name, value in required_text.items():
        if not value.strip():
            errors.append(f"task {task.id!r} has no {field_name}")

    required_lists = {
        "required_behaviors": task.required_behaviors,
        "invariants": task.invariants,
        "reviewer_focus": task.reviewer_focus,
    }
    for field_name, values in required_lists.items():
        if not values:
            errors.append(f"task {task.id!r} has no {field_name}")

    clauses = [
        *task.required_behaviors,
        *task.implementation_constraints,
        *task.invariants,
        *task.prohibited_shortcuts,
    ]
    clause_ids = [clause.id.strip() for clause in clauses]
    if any(not clause_id for clause_id in clause_ids):
        errors.append(f"task {task.id!r} has an empty contract clause id")
    if len(clause_ids) != len(set(clause_ids)):
        errors.append(f"task {task.id!r} contract clause ids must be unique")
    for clause in clauses:
        if not clause.description.strip():
            errors.append(
                f"task {task.id!r} clause {clause.id!r} has no description"
            )
