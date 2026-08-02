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


def test_scenario_placeholders_are_optional_but_fixture_paths_stay_confined():
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
    assert "unsafe fixture path" in result.summary()


def test_scenario_environment_contract_rejects_only_conflicting_facts():
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


def test_executable_availability_is_runtime_evidence_not_a_schema_gate():
    proposal = make_proposal(delivery_run=["python -m package --help"])
    criterion = proposal.acceptance_criteria[0]
    criterion.verified_safety_properties = ["path_confinement"]
    criterion.mode = "invariant"
    criterion.command = ""

    result = validate_acceptance(proposal)

    assert result.valid


def test_safety_test_design_is_semantic_content_not_a_schema_gate():
    proposal = make_proposal()
    proposal.tasks[0].candidate = "confine file paths"
    proposal.tasks[0].description = "Keep file access inside the workspace root"
    proposal.tasks[0].required_safety_properties = ["path_confinement"]
    criterion = proposal.acceptance_criteria[0]
    criterion.verified_safety_properties = ["path_confinement"]
    criterion.mode = "red_green"
    criterion.command = "python3 -c \"print('confined')\""

    assert validate_acceptance(proposal).valid


def test_schema_does_not_guess_whether_path_confinement_is_relevant():
    proposal = make_proposal()
    proposal.tasks[0].description = "Improve the runtime execution path"
    proposal.tasks[0].required_safety_properties = ["path_confinement"]
    criterion = proposal.acceptance_criteria[0]
    criterion.mode = "invariant"
    criterion.command = ""
    criterion.verified_safety_properties = ["path_confinement"]

    result = validate_acceptance(proposal)

    assert result.valid


def test_schema_allows_placeholders_where_the_target_contract_needs_them():
    proposal = make_proposal(delivery_run=["python3 -m package --help {workspace}"])
    criterion = proposal.acceptance_criteria[0]
    criterion.verified_safety_properties = ["path_confinement"]
    criterion.mode = "red_green"
    criterion.command = "python3 -m package {prompt} --dir {workspace}"

    result = validate_acceptance(proposal)

    assert result.valid
