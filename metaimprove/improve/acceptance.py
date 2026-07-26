"""Deterministic validation for an ImprovementProposal's definition of done."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ImprovementProposal


@dataclass
class AcceptanceValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(self.errors) if self.errors else "valid"


def validate_acceptance(proposal: ImprovementProposal) -> AcceptanceValidation:
    """Require traceable criteria, executable hard checks, and an explicit scope."""
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
