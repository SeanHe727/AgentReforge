from __future__ import annotations

from conftest import make_proposal

from agentreforge.improve.acceptance import validate_acceptance
from agentreforge.improve.models import DeliveryScenario, ExecutableCondition


def test_minimal_delivery_contract_is_valid():
    result = validate_acceptance(make_proposal())

    assert result.valid
    assert result.errors == []


def test_handoff_rejects_unknown_acceptance_reference():
    proposal = make_proposal()
    proposal.tasks[0].acceptance_criteria_ids = ["missing"]

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "unknown criteria" in result.summary()


def test_task_detail_and_suggested_scope_are_not_schema_gates():
    proposal = make_proposal(allowed_write_paths=[])
    task = proposal.tasks[0]
    task.candidate = ""
    task.rationale = ""
    task.capability_change = ""
    task.required_behaviors = []
    task.invariants = []
    task.reviewer_focus = []
    task.affected_components = ["new or updated test path"]
    proposal.acceptance_criteria[0].verification = "review"
    proposal.acceptance_criteria[0].command = ""

    assert validate_acceptance(proposal).valid


def test_delivery_contract_requires_a_smoke_command_or_scenario():
    proposal = make_proposal(delivery_run=[])

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "smoke command or frozen end-to-end scenario" in result.summary()


def test_frozen_scenario_can_replace_generic_smoke_command():
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="capability",
                prompt="inspect the fixture",
                command=[
                    "python3",
                    "-c",
                    "import sys; print(sys.argv[1], sys.argv[2])",
                    "{prompt}",
                    "{workspace}",
                ],
                fixture_files={"src/example.py": "VALUE = 1\n"},
                expected_behaviors=["inspect src/example.py"],
            )
        ],
    )

    assert validate_acceptance(proposal).valid


def test_scenario_requires_prompt_workspace_placeholders_and_safe_fixture_paths():
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="bad",
                prompt="inspect",
                command=["python3", "-c", "print('nothing')"],
                fixture_files={"../escape.py": "bad"},
            )
        ],
    )

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "must pass {prompt}" in result.summary()
    assert "must pass {workspace}" in result.summary()
    assert "unsafe fixture path" in result.summary()


def test_scenario_environment_contract_requires_trajectory_and_no_conflicts():
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="fallback",
                prompt="verify with python",
                command=[
                    "python3",
                    "-c",
                    "print('done')",
                    "{prompt}",
                    "{workspace}",
                ],
                executable_conditions=[
                    ExecutableCondition(name="python", state="unavailable"),
                    ExecutableCondition(name="python", state="available"),
                ],
            )
        ],
    )

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "conflicting executable conditions" in result.summary()
    assert "must require trajectory evidence" in result.summary()


def test_delivery_commands_require_available_interpreter(monkeypatch):
    proposal = make_proposal(delivery_run=["python -m package --help"])
    criterion = proposal.acceptance_criteria[0]
    criterion.verified_safety_properties = ["path_confinement"]
    criterion.mode = "invariant"
    criterion.command = ""
    monkeypatch.setattr(
        "agentreforge.improve.acceptance.shutil.which",
        lambda executable: None if executable == "python" else f"/bin/{executable}",
    )

    result = validate_acceptance(proposal)

    assert not result.valid
    assert result.summary().count("unavailable interpreter 'python'") == 1


def test_system_owned_safety_property_requires_invariant_mode_and_no_command():
    proposal = make_proposal()
    proposal.tasks[0].candidate = "confine file paths"
    proposal.tasks[0].description = "Keep file access inside the workspace root"
    proposal.tasks[0].required_safety_properties = ["path_confinement"]
    criterion = proposal.acceptance_criteria[0]
    criterion.verified_safety_properties = ["path_confinement"]
    criterion.mode = "red_green"
    criterion.command = "python3 -c \"print('confined')\""

    invalid = validate_acceptance(proposal)

    assert not invalid.valid
    assert "must use invariant mode" in invalid.summary()
    assert "system-owned" in invalid.summary()

    criterion.mode = "invariant"
    criterion.command = ""

    assert validate_acceptance(proposal).valid


def test_unrelated_task_cannot_declare_path_confinement():
    proposal = make_proposal()
    proposal.tasks[0].description = "Improve the runtime execution path"
    proposal.tasks[0].required_safety_properties = ["path_confinement"]
    criterion = proposal.acceptance_criteria[0]
    criterion.mode = "invariant"
    criterion.command = ""
    criterion.verified_safety_properties = ["path_confinement"]

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "does not change a path-taking or filesystem boundary" in result.summary()


def test_safety_check_rejects_llm_generated_probe_and_delivery_placeholder():
    proposal = make_proposal(delivery_run=["python3 -m package --help {workspace}"])
    criterion = proposal.acceptance_criteria[0]
    criterion.verified_safety_properties = ["path_confinement"]
    criterion.mode = "red_green"
    criterion.command = "python3 -m package {prompt} --dir {workspace}"

    result = validate_acceptance(proposal)

    assert not result.valid
    assert "must use invariant mode" in result.summary()
    assert "system-owned" in result.summary()
    assert result.summary().count("scenario-only placeholder") == 1
