"""Deterministic validation for an ImprovementProposal's definition of done."""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass, field

from .models import AcceptanceCriterion, ImprovementProposal, ImprovementTask


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
    criteria_by_id = {criterion.id: criterion for criterion in criteria}
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
        _validate_task_safety(
            task,
            [
                criteria_by_id[criterion_id]
                for criterion_id in task.acceptance_criteria_ids
                if criterion_id in criteria_by_id
            ],
            errors,
        )
        assigned.update(task.acceptance_criteria_ids)

    unassigned = [c.id for c in criteria if c.required and c.id not in assigned]
    if unassigned:
        errors.append(f"required criteria are not assigned to a task: {', '.join(unassigned)}")

    for criterion in criteria:
        if criterion.verification == "command" and not criterion.command.strip():
            errors.append(f"command criterion {criterion.id!r} has no command")
        elif criterion.verification == "command":
            _validate_command_executable(
                criterion.command,
                f"command criterion {criterion.id!r}",
                errors,
            )
        _validate_safety_criterion(criterion, errors)

    for index, command in enumerate(proposal.delivery_run, start=1):
        _validate_command_executable(command, f"delivery command {index}", errors)

    return AcceptanceValidation(not errors, errors)


def _validate_task_contract(task: ImprovementTask, errors: list[str]) -> None:
    """Reject thin hand-offs before a Writer is allowed to modify code."""

    required_text = {
        "candidate": task.candidate,
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


def _validate_command_executable(command: str, label: str, errors: list[str]) -> None:
    """Reject a frozen Python command when its named interpreter is unavailable."""

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        errors.append(f"{label} cannot be parsed: {exc}")
        return
    executable = ""
    for token in tokens:
        if token == "env":
            continue
        name, separator, _value = token.partition("=")
        if separator and name.replace("_", "").isalnum():
            continue
        executable = token
        break
    if executable in {"python", "python3"} and shutil.which(executable) is None:
        errors.append(
            f"{label} uses unavailable interpreter {executable!r}; "
            "choose an interpreter present in the execution environment"
        )


def _validate_task_safety(
    task: ImprovementTask,
    assigned_criteria: list[AcceptanceCriterion],
    errors: list[str],
) -> None:
    """Require explicit, traceable coverage for lightweight path-tool safety."""

    required = set(task.required_safety_properties)
    if _looks_like_path_tool_task(task) and "path_confinement" not in required:
        errors.append(
            f"task {task.id!r} changes path/navigation tools but does not require "
            "safety property 'path_confinement'"
        )
    covered = {
        safety_property
        for criterion in assigned_criteria
        for safety_property in criterion.verified_safety_properties
    }
    missing = sorted(required - covered)
    if missing:
        errors.append(
            f"task {task.id!r} has no assigned acceptance criterion for safety "
            f"properties: {', '.join(missing)}"
        )


def _looks_like_path_tool_task(task: ImprovementTask) -> bool:
    contract = " ".join(
        [
            task.candidate,
            task.description,
            task.capability_change,
            *(clause.description for clause in task.required_behaviors),
            *(clause.description for clause in task.implementation_constraints),
        ]
    ).lower()
    changes_tool_module = any(
        component.lower().endswith("tools.py") or "/tools/" in component.lower()
        for component in task.affected_components
    )
    path_terms = ("path", "directory", "navigation", "repository search", "file search")
    return changes_tool_module and any(term in contract for term in path_terms)


def _validate_safety_criterion(
    criterion: AcceptanceCriterion,
    errors: list[str],
) -> None:
    if "path_confinement" not in criterion.verified_safety_properties:
        return
    if criterion.verification != "command":
        errors.append(
            f"safety criterion {criterion.id!r} for path_confinement must be executable"
        )
        return
    if ".." not in criterion.command:
        errors.append(
            f"safety criterion {criterion.id!r} for path_confinement must exercise "
            "a '..' traversal attempt"
        )
    if not criterion.required_output_contains:
        errors.append(
            f"safety criterion {criterion.id!r} for path_confinement must require "
            "a stable blocked/error output marker"
        )
