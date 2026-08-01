"""Deterministic validation for an ImprovementProposal's definition of done."""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .models import ImprovementProposal, ImprovementTask


@dataclass
class AcceptanceValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(self.errors) if self.errors else "valid"


def validate_acceptance(proposal: ImprovementProposal) -> AcceptanceValidation:
    """Validate only the executable delivery contract and cross-references.

    Task quality, clause completeness, suggested file scope, and test design are
    agent-review concerns rather than hand-off schema gates.
    """
    errors: list[str] = []
    criteria = proposal.acceptance_criteria
    criterion_ids = [criterion.id for criterion in criteria]
    known = set(criterion_ids)

    if len(criterion_ids) != len(known):
        errors.append("acceptance criterion ids must be unique")
    if not proposal.delivery_run and not proposal.delivery_scenarios:
        errors.append(
            "delivery must contain a minimal smoke command or frozen end-to-end scenario"
        )

    for task in proposal.tasks:
        unknown = sorted(set(task.acceptance_criteria_ids) - known)
        if unknown:
            errors.append(f"task {task.id!r} references unknown criteria: {', '.join(unknown)}")
        referenced = [
            criterion
            for criterion in criteria
            if criterion.id in task.acceptance_criteria_ids
        ]
        for safety in task.required_safety_properties:
            if not any(
                safety in criterion.verified_safety_properties
                for criterion in referenced
            ):
                errors.append(
                    f"task {task.id!r} requires {safety!r} but does not reference "
                    "a matching safety criterion"
                )
            if safety == "path_confinement" and not _task_is_path_relevant(task):
                errors.append(
                    f"task {task.id!r} declares path_confinement for a Candidate "
                    "that does not change a path-taking or filesystem boundary"
                )

    for criterion in criteria:
        safety_check = bool(criterion.verified_safety_properties)
        if safety_check and criterion.mode != "invariant":
            errors.append(
                f"safety criterion {criterion.id!r} must use invariant mode"
            )
        if safety_check and criterion.command.strip():
            errors.append(
                f"safety criterion {criterion.id!r} is system-owned and must not "
                "supply an LLM-generated command"
            )
        for safety in criterion.verified_safety_properties:
            owners = [
                task.id
                for task in proposal.tasks
                if criterion.id in task.acceptance_criteria_ids
                and safety in task.required_safety_properties
            ]
            if not owners:
                errors.append(
                    f"safety criterion {criterion.id!r} verifies {safety!r} but no "
                    "referencing Task requires it"
                )

    for index, command in enumerate(proposal.delivery_run, start=1):
        _reject_scenario_placeholders(command, f"delivery command {index}", errors)
        _validate_command_executable(command, f"delivery command {index}", errors)

    scenario_ids = [scenario.id for scenario in proposal.delivery_scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        errors.append("delivery scenario ids must be unique")
    for scenario in proposal.delivery_scenarios:
        if not scenario.prompt.strip():
            errors.append(f"delivery scenario {scenario.id!r} requires a prompt")
        if not scenario.command:
            errors.append(f"delivery scenario {scenario.id!r} requires an argv command")
        else:
            _validate_argv_executable(
                scenario.command,
                f"delivery scenario {scenario.id!r}",
                errors,
            )
            if not any("{prompt}" in arg for arg in scenario.command):
                errors.append(
                    f"delivery scenario {scenario.id!r} command must pass {{prompt}}"
                )
            if not any("{workspace}" in arg for arg in scenario.command):
                errors.append(
                    f"delivery scenario {scenario.id!r} command must pass {{workspace}}"
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
        if scenario.executable_conditions and not scenario.requires_trajectory:
            errors.append(
                f"delivery scenario {scenario.id!r} with executable conditions "
                "must require trajectory evidence"
            )

    return AcceptanceValidation(not errors, errors)


def _task_is_path_relevant(task: ImprovementTask) -> bool:
    text = " ".join(
        [
            task.candidate,
            task.description,
            task.rationale,
            task.capability_change,
            *(clause.description for clause in task.required_behaviors),
            *(clause.description for clause in task.implementation_constraints),
            *task.reviewer_focus,
        ]
    ).casefold()
    return any(
        marker in text
        for marker in (
            "path confinement",
            "path-taking",
            "file path",
            "filesystem",
            "file access",
            "file-access",
            "traversal",
            "confinement",
            "workspace root",
            "repository discovery",
        )
    )


def _reject_scenario_placeholders(
    command: str,
    label: str,
    errors: list[str],
) -> None:
    placeholders = [
        placeholder
        for placeholder in ("{prompt}", "{workspace}")
        if placeholder in command
    ]
    if placeholders:
        errors.append(
            f"{label} uses scenario-only placeholder(s): {', '.join(placeholders)}"
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


def _validate_argv_executable(
    argv: list[str],
    label: str,
    errors: list[str],
) -> None:
    executable = argv[0].strip() if argv else ""
    if not executable or "{" in executable:
        errors.append(f"{label} requires a concrete executable as argv[0]")
    elif shutil.which(executable) is None:
        errors.append(
            f"{label} uses unavailable executable {executable!r}; "
            "choose one present in the execution environment"
        )
