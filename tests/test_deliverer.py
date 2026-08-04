from __future__ import annotations

import asyncio
import sys

from conftest import make_proposal

from agentreforge.improve.acceptance_runner import (
    AcceptanceRun,
    AcceptanceRunner,
    RunResult,
    ScenarioRunResult,
    acceptance_failures,
    dangerous_command,
)
from agentreforge.improve.deliverer import (
    Deliverer,
    GoalReview,
    _command_hard_failure,
    _delivery_review_output_error,
    _scenario_hard_failure,
    goal_review_message,
)
from agentreforge.improve.delivery_coordinator import DeliveryCoordinator
from agentreforge.improve.models import (
    DeliveryScenario,
    DiagnosticFinding,
    ExecutableCondition,
    InterventionCandidate,
    OrchestratorAnalysis,
)


def test_delivery_gate_checks_exit_and_output():
    proposal = make_proposal()
    criterion = proposal.acceptance_criteria[0]
    criterion.expected_exit_code = 0
    criterion.required_output_contains = ["1 passed"]
    criterion.forbidden_output_contains = ["Traceback"]

    failures = acceptance_failures(
        proposal,
        [RunResult(criterion.command, 0, "1 passed in 0.1s")],
    )

    assert failures == []


def test_ordinary_acceptance_hint_does_not_block_delivery():
    proposal = make_proposal()
    criterion = proposal.acceptance_criteria[0]
    criterion.required_output_contains = ["1 passed"]

    failures = acceptance_failures(
        proposal,
        [RunResult(criterion.command, 1, "failed")],
    )

    assert failures == []


def test_delivery_command_denylist_blocks_destructive_git():
    assert dangerous_command("git reset --hard HEAD") is not None
    assert dangerous_command("python -m pytest tests") is None


def test_decisive_execution_failures_veto_without_proving_success():
    assert _command_hard_failure(RunResult("not-a-command", 127, "not found"))
    assert _command_hard_failure(RunResult("python3 -m pytest", 0, "passed")) == ""
    assert _scenario_hard_failure(
        ScenarioRunResult(
            scenario_id="bounded-run",
            prompt="complete the task",
            command=["agent"],
            exit_code=0,
            output="(stopped: reached max steps)",
            trajectory=[{"type": "done", "outcome": "incomplete"}],
        )
    )


def test_runner_executes_frozen_prompt_in_isolated_fixture_and_collects_evidence(
    tmp_path,
):
    script = (
        "import json, os, sys; "
        "from pathlib import Path; "
        "workspace=Path(sys.argv[2]); "
        "(workspace/'result.txt').write_text(sys.argv[1]); "
        "Path(os.environ['AGENTREFORGE_TRAJECTORY_PATH']).write_text("
        "json.dumps({'type':'tool_result','name':'search_code'})+'\\n'); "
        "print('scenario complete')"
    )
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="search",
                prompt="find the model client",
                command=[
                    "python3",
                    "-c",
                    script,
                    "{prompt}",
                    "{workspace}",
                ],
                fixture_files={"src/model.py": "MODEL = 'demo'\n"},
                expected_behaviors=["locate src/model.py"],
            )
        ],
    )

    result = asyncio.run(
        AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path))
    )

    assert result.passed
    assert len(result.scenario_runs) == 1
    scenario = result.scenario_runs[0]
    assert scenario.output.strip() == "scenario complete"
    assert scenario.changed_files == ["result.txt"]
    assert scenario.artifacts["result.txt"].startswith(
        "Objective:\nfind the model client\n\nPrimary success conditions:"
    )
    assert scenario.trajectory_available
    assert scenario.trajectory[0]["name"] == "search_code"


def test_target_trajectory_is_private_from_the_task_workspace(tmp_path):
    script = (
        "import os, sys; "
        "from pathlib import Path; "
        "workspace=Path(sys.argv[1]).resolve(); "
        "trajectory=Path(os.environ['AGENTREFORGE_TRAJECTORY_PATH']).resolve(); "
        "assert workspace not in trajectory.parents; "
        "assert not (workspace/'.agentreforge_trajectory.jsonl').exists(); "
        "trajectory.write_text('{\"type\":\"done\",\"outcome\":\"completed\"}\\n'); "
        "print('trajectory private')"
    )
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="private-trajectory",
                prompt="inspect the task workspace",
                command=["python3", "-c", script, "{workspace}"],
            )
        ],
    )

    result = asyncio.run(AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path)))

    assert result.passed
    assert result.scenario_runs[0].output.strip() == "trajectory private"
    assert result.scenario_runs[0].changed_files == []


def test_runner_materializes_and_records_executable_conditions(tmp_path, monkeypatch):
    runtime_bin = tmp_path / "runtime-bin"
    runtime_bin.mkdir()
    (runtime_bin / "python").symlink_to(sys.executable)
    (runtime_bin / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(runtime_bin))
    script = (
        "import json, os; from pathlib import Path; "
        "Path(os.environ['AGENTREFORGE_TRAJECTORY_PATH']).write_text("
        "json.dumps({'type':'tool_result','name':'run_bash',"
        "'arguments':{'command':'python -m unittest'},"
        "'content':'(exit 0)'})+'\\n'); print('verified')"
    )
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="python-fallback",
                prompt="verify the fixture",
                command=[
                    "python3",
                    "-c",
                    script,
                    "{prompt}",
                    "{workspace}",
                ],
                executable_conditions=[
                    ExecutableCondition(name="python", state="unavailable"),
                    ExecutableCondition(name="python3", state="available"),
                ],
                requires_trajectory=True,
            )
        ],
    )

    result = asyncio.run(
        AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path))
    )

    assert result.passed
    scenario = result.scenario_runs[0]
    assert scenario.environment_ready
    assert scenario.output.strip() == "verified"
    assert scenario.trajectory_available
    facts = {fact["name"]: fact for fact in scenario.environment_facts}
    assert facts["python"]["observed_state"] == "unavailable"
    assert facts["python"]["resolved_path"] == ""
    assert facts["python"]["satisfied"] is True
    assert facts["python3"]["observed_state"] == "available"
    assert str(facts["python3"]["resolved_path"]).endswith("/python3")
    assert facts["python3"]["satisfied"] is True


def test_unmaterialized_scenario_environment_is_an_environment_failure(
    tmp_path,
):
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="missing-runtime",
                prompt="run the task",
                command=[
                    "python3",
                    "-c",
                    "print('unused')",
                    "{prompt}",
                    "{workspace}",
                ],
                executable_conditions=[
                    ExecutableCondition(
                        name="agentreforge-definitely-missing-runtime",
                        state="available",
                    )
                ],
                requires_trajectory=True,
            )
        ],
    )
    acceptance = asyncio.run(
        AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path))
    )

    assert not acceptance.passed
    assert not acceptance.scenario_runs[0].environment_ready
    assert "environment conditions could not be materialized" in acceptance.failures[0]

    class Client:
        async def chat(self, messages, tools=None, *, system_prompt):
            raise AssertionError("environment failure should not invoke the LLM")
            yield  # pragma: no cover

    review = asyncio.run(
        Deliverer(client=Client()).review(
            proposal,
            loop_diff="diff --git a/tool.py b/tool.py",
            acceptance=acceptance,
        )
    )
    assert not review.accepted
    assert review.failure_kind == "environment_failure"


def test_demo_agent_adapter_collects_real_tool_trajectory(tmp_path):
    package = tmp_path / "demo_agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agent.py").write_text(
        "from pathlib import Path\n"
        "def execute(name, args, cwd):\n"
        "    root = Path(cwd)\n"
        "    if name == 'list_dir': return '\\n'.join(p.name for p in root.iterdir())\n"
        "    if name == 'read_file': return (root / args['path']).read_text()\n"
        "    if name == 'write_file':\n"
        "        (root / args['path']).write_text(args['content']); return 'wrote'\n"
        "    if name == 'run_bash': return '(exit 127) command not found'\n"
        "    return 'error: unknown'\n"
        "def run_task(prompt, cwd):\n"
        "    execute('list_dir', {'path': '.'}, cwd)\n"
        "    execute('read_file', {'path': 'app.py'}, cwd)\n"
        "    execute('write_file', {'path': 'app.py', 'content': 'VALUE = 2\\n'}, cwd)\n"
        "    execute('run_bash', {'command': 'python3 -m unittest'}, cwd)\n"
        "    return 'done'\n",
        encoding="utf-8",
    )
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="demo",
                prompt="inspect, edit, verify",
                command=[
                    "python3",
                    "-m",
                    "demo_agent",
                    "{prompt}",
                    "--dir",
                    "{workspace}",
                ],
                fixture_files={"app.py": "VALUE = 1\n"},
                requires_trajectory=True,
            )
        ],
    )

    result = asyncio.run(
        AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path))
    )

    assert result.passed
    scenario = result.scenario_runs[0]
    assert scenario.changed_files == ["app.py"]
    assert [event["name"] for event in scenario.trajectory if "name" in event] == [
        "list_dir",
        "read_file",
        "write_file",
        "run_bash",
    ]
    assert scenario.trajectory[-2]["is_error"] is True
    assert scenario.trajectory_available


def test_demo_adapter_separates_free_reads_from_bounded_actions(tmp_path):
    package = tmp_path / "demo_agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agent.py").write_text(
        "from pathlib import Path\n"
        "def execute(name, args, cwd):\n"
        "    root = Path(cwd)\n"
        "    if name == 'read_file': return (root / args['path']).read_text()\n"
        "    if name == 'write_file':\n"
        "        (root / args['path']).write_text(args['content']); return 'wrote'\n"
        "def run_task(prompt, cwd, max_steps=8):\n"
        "    for _ in range(10): execute('read_file', {'path': 'seed.txt'}, cwd)\n"
        "    for i in range(9):\n"
        "        execute('write_file', {'path': f'out-{i}.txt', 'content': 'x'}, cwd)\n"
        "    return 'done'\n",
        encoding="utf-8",
    )
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="budgeted",
                prompt="inspect and edit",
                command=["python3", "-m", "demo_agent", "{prompt}"],
                fixture_files={"seed.txt": "seed"},
            )
        ],
    )

    result = asyncio.run(AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path)))
    scenario = result.scenario_runs[0]

    assert len([item for item in scenario.trajectory if item.get("name") == "read_file"]) == 10
    assert len([item for item in scenario.trajectory if item.get("action_step")]) == 8
    assert any(item.get("budget_blocked") for item in scenario.trajectory)
    assert scenario.changed_files == [f"out-{index}.txt" for index in range(8)]


def test_demo_adapter_records_model_component_inputs_and_outputs(tmp_path):
    package = tmp_path / "demo_agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agent.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def chat(messages, tools=None):\n"
        "    if not any(item.get('role') == 'tool' for item in messages):\n"
        "        return {'role': 'assistant', 'content': '', 'tool_calls': [{\n"
        "            'id': 'call-1', 'function': {'name': 'write_file',\n"
        "            'arguments': json.dumps({'path': 'result.txt', 'content': 'ok'})}}]}\n"
        "    return {'role': 'assistant', 'content': 'verified', 'tool_calls': []}\n"
        "def execute(name, args, cwd):\n"
        "    (Path(cwd) / args['path']).write_text(args['content']); return 'wrote'\n"
        "def run_task(prompt, cwd, max_steps=8):\n"
        "    messages = [{'role': 'system', 'content': 'system'},\n"
        "                {'role': 'user', 'content': prompt}]\n"
        "    for _ in range(max_steps):\n"
        "        message = chat(messages, tools=[]); messages.append(message)\n"
        "        calls = message.get('tool_calls') or []\n"
        "        if not calls: return message['content']\n"
        "        for call in calls:\n"
        "            fn = call['function']; args = json.loads(fn['arguments'])\n"
        "            result = execute(fn['name'], args, cwd)\n"
        "            messages.append({'role': 'tool', 'tool_call_id': call['id'],\n"
        "                             'content': result})\n"
        "    return '(stopped: reached max steps)'\n",
        encoding="utf-8",
    )
    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="component-io",
                prompt="write the result",
                command=["python3", "-m", "demo_agent", "{prompt}"],
                requires_trajectory=True,
            )
        ],
    )

    result = asyncio.run(AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path)))
    turns = [
        item
        for item in result.scenario_runs[0].trajectory
        if item.get("type") == "agent_turn"
    ]

    assert [item["turn"] for item in turns] == [1, 2]
    assert [message["role"] for message in turns[0]["input_messages"]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in turns[1]["input_messages"]] == [
        "assistant",
        "tool",
    ]
    assert turns[1]["content"] == "verified"


def test_demo_agent_adapter_runs_deterministic_path_confinement_probe(tmp_path):
    package = tmp_path / "demo_agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agent.py").write_text(
        "from pathlib import Path\n"
        "def execute(name, args, cwd):\n"
        "    root = Path(cwd).resolve()\n"
        "    path = (root / args['path']).resolve()\n"
        "    try: path.relative_to(root)\n"
        "    except ValueError: return 'error: path escapes workspace root'\n"
        "    return path.read_text()\n"
        "def run_task(prompt, cwd): return 'done'\n",
        encoding="utf-8",
    )
    proposal = make_proposal()
    proposal.tasks[0].required_safety_properties = ["path_confinement"]

    result = asyncio.run(
        AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path))
    )

    assert result.passed
    safety = next(
        run for run in result.runs if run.command == "adapter:safety:path_confinement"
    )
    assert safety.exit_code == 0
    assert "safe: traversal blocked" in safety.output


def test_demo_agent_adapter_rejects_unsafe_path_confinement(tmp_path):
    package = tmp_path / "demo_agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agent.py").write_text(
        "from pathlib import Path\n"
        "def execute(name, args, cwd):\n"
        "    return (Path(cwd) / args['path']).resolve().read_text()\n"
        "def run_task(prompt, cwd): return 'done'\n",
        encoding="utf-8",
    )
    proposal = make_proposal()
    proposal.tasks[0].required_safety_properties = ["path_confinement"]

    result = asyncio.run(
        AcceptanceRunner(timeout_s=10).run(proposal, cwd=str(tmp_path))
    )

    assert not result.passed
    safety = next(
        run for run in result.runs if run.command == "adapter:safety:path_confinement"
    )
    assert safety.exit_code == 1
    assert "exposed the outside sentinel" in safety.output
    assert any("system safety probe" in failure for failure in result.failures)


def test_goal_review_message_includes_diff_and_authoritative_runner_output():
    proposal = make_proposal(
        goals=["make repository navigation reachable"],
        analysis=OrchestratorAnalysis(
            findings=[
                DiagnosticFinding(
                    symptom="no navigation",
                    root_cause="missing tools",
                    capability_gap="repository awareness",
                    evidence_refs=["demo_agent/tools.py"],
                )
            ],
            candidates=[
                InterventionCandidate(
                    name="navigation tools",
                    level="tool",
                    mechanism="add and wire list/search tools",
                    expected_capability_delta="repository awareness",
                )
            ],
            selected_candidates=["navigation tools"],
            packing_reason="one bounded Candidate fits the batch budget",
            causal_mechanism="wire tools into the active tool surface",
            expected_capability_delta="repository awareness",
        ),
        delivery_checklist=["new tools are wired into the active agent loop"],
        delivery_scenarios=[
            DeliveryScenario(
                id="navigation",
                prompt="locate the model client",
                command=[
                    "python3",
                    "-m",
                    "demo_agent",
                    "{prompt}",
                    "--dir",
                    "{workspace}",
                ],
                expected_behaviors=["use repository navigation"],
            )
        ],
    )

    acceptance = AcceptanceRun(
        passed=True,
        runs=[
            RunResult(
                "python3 -m demo_agent --smoke",
                0,
                "agent started\nobjective reached",
            )
        ],
        scenario_runs=[
            ScenarioRunResult(
                scenario_id="navigation",
                prompt="locate the model client",
                command=["python3", "-m", "demo_agent", "locate the model client"],
                exit_code=0,
                output="found demo_agent/llm.py",
                trajectory=[{"type": "tool_result", "name": "search_code"}],
                trajectory_available=True,
            )
        ],
    )
    message = goal_review_message(
        proposal,
        "diff --git a/demo_agent/tools.py",
        acceptance,
    )

    assert "navigation tools" in message
    assert "wire tools into the active tool surface" in message
    assert "new tools are wired into the active agent loop" in message
    assert "diff --git a/demo_agent/tools.py" in message
    assert "python3 -m demo_agent --smoke" in message
    assert '"exit_code": 0' in message
    assert "objective reached" in message
    assert "locate the model client" in message
    assert "found demo_agent/llm.py" in message
    assert "search_code" in message


def test_deliverer_retries_an_empty_review():
    class Client:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tools=None, *, system_prompt):
            self.calls += 1
            text = (
                ""
                if self.calls == 1
                else (
                    '{"ready": true, "missing_objectives": [], '
                    '"integration_concerns": [], "summary": "wired"}'
                )
            )
            yield {"type": "text_delta", "text": text}

    client = Client()
    result = asyncio.run(
        Deliverer(client=client).review(
            make_proposal(),
            loop_diff="diff --git a/src/agent.py b/src/agent.py",
            acceptance=AcceptanceRun(
                passed=True,
                runs=[RunResult("python3 -m src.agent --help", 0, "usage")],
            ),
        )
    )

    assert result.accepted
    assert client.calls == 2


def test_deliverer_rejects_missing_required_trajectory_before_llm():
    class Client:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tools=None, *, system_prompt):
            self.calls += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"ready": true, "failure_kind": "none", '
                    '"missing_objectives": [], "integration_concerns": [], '
                    '"blocking_evidence": [], "summary": "self-reported success"}'
                ),
            }

    proposal = make_proposal(
        delivery_scenarios=[
            DeliveryScenario(
                id="ordered-tools",
                prompt="inspect, edit, then verify",
                command=[
                    "python3",
                    "-m",
                    "demo_agent",
                    "{prompt}",
                    "--dir",
                    "{workspace}",
                ],
                expected_behaviors=["inspect before editing"],
                requires_trajectory=True,
            )
        ]
    )
    client = Client()

    result = asyncio.run(
        Deliverer(client=client).review(
            proposal,
            loop_diff="diff --git a/src/agent.py b/src/agent.py",
            acceptance=AcceptanceRun(
                passed=True,
                scenario_runs=[
                    ScenarioRunResult(
                        scenario_id="ordered-tools",
                        prompt="inspect, edit, then verify",
                        command=["python3", "-m", "demo_agent"],
                        exit_code=0,
                        output="I inspected and verified the change.",
                        changed_files=["app.py"],
                    )
                ],
            ),
        )
    )

    assert not result.accepted
    assert result.failure_kind == "verification_gap"
    assert "ordered-tools" in result.text
    assert client.calls == 0


def test_delivery_review_requires_consistent_failure_classification():
    assert (
        _delivery_review_output_error(
            '{"ready": false, "failure_kind": "verification_gap", '
            '"missing_objectives": [], "integration_concerns": [], '
            '"blocking_evidence": ["scenario only ran --help"], '
            '"summary": "capability was not exercised"}'
        )
        == ""
    )
    assert "classified failure_kind" in _delivery_review_output_error(
        '{"ready": false, "failure_kind": "none", '
        '"missing_objectives": ["not demonstrated"], '
        '"integration_concerns": [], "blocking_evidence": [], "summary": ""}'
    )
    assert "ready=true" in _delivery_review_output_error(
        '{"ready": true, "failure_kind": "implementation_defect", '
        '"missing_objectives": [], "integration_concerns": [], '
        '"blocking_evidence": [], "summary": ""}'
    )


def test_delivery_coordinator_requires_both_runner_and_deliverer():
    class FakeRunner:
        async def run(self, proposal, *, cwd):
            return AcceptanceRun(
                passed=False,
                runs=[RunResult("test", 1, "failed")],
                failures=["test failed"],
            )

    class FakeDeliverer:
        def __init__(self):
            self.acceptance = None

        async def review(self, proposal, *, loop_diff, acceptance):
            self.acceptance = acceptance
            return GoalReview(
                accepted=True,
                text="GOAL: ACHIEVED\nVERDICT: ACCEPT",
            )

    fake_deliverer = FakeDeliverer()
    coordinator = DeliveryCoordinator(
        runner=FakeRunner(),
        deliverer=fake_deliverer,
    )

    result = asyncio.run(
        coordinator.deliver(
            make_proposal(),
            cwd="/candidate",
            loop_diff="diff --git a/a.py b/a.py",
        )
    )

    assert not result.passed
    assert not result.delivery_gate_ok
    assert result.acceptance_failures == ["test failed"]
    assert result.goal_accepted
    assert fake_deliverer.acceptance is not None
    assert fake_deliverer.acceptance.failures == ["test failed"]
    assert "test failed" in result.reasons
    assert result.goal_review.startswith("GOAL: ACHIEVED")


def test_agentic_deliverer_chooses_and_runs_a_frozen_scenario(tmp_path):
    class Client:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tools=None, *, system_prompt):
            self.calls += 1
            if self.calls == 1:
                yield {
                    "type": "text_delta",
                    "text": (
                        '{"ready": true, "missing_requirements": [], '
                        '"execution_focus": ["observe the target"], '
                        '"summary": "scenario is ready"}'
                    ),
                }
                return
            if self.calls == 2:
                yield {
                    "type": "text_delta",
                    "text": (
                        '{"scenario_id": "actual-run", '
                        '"observed_facts": ["process printed observed target"], '
                        '"trajectory_findings": [], "artifact_findings": [], '
                        '"baseline_consistent": false, "candidate_consistent": true, '
                        '"discriminating_evidence": ["observed target"], '
                        '"outcome_assessments": [{"condition_id": "legacy-primary", '
                        '"category": "primary_success", "status": "supported", '
                        '"evidence": ["observed target"], '
                        '"explanation": "the target path ran"}], '
                        '"confounders": [], "sufficient": true, '
                        '"summary": "scenario produced usable evidence"}'
                    ),
                }
                return
            yield {
                "type": "text_delta",
                "text": (
                    '{"ready": true, "failure_kind": "none", '
                    '"missing_objectives": [], "integration_concerns": [], '
                    '"proposal_violations": [], "blocking_evidence": [], '
                    '"summary": "observed the frozen target path"}'
                )
            }

    proposal = make_proposal(
        delivery_run=[],
        delivery_scenarios=[
            DeliveryScenario(
                id="actual-run",
                prompt="Observe the target entry point.",
                command=["python3", "-c", "print('observed target')"],
            )
        ],
    )
    client = Client()

    result = asyncio.run(
        DeliveryCoordinator(
            runner=AcceptanceRunner(timeout_s=10),
            deliverer=Deliverer(client=client),
        ).deliver(
            proposal,
            cwd=str(tmp_path),
            loop_diff="diff --git a/agent.py b/agent.py",
        )
    )

    assert result.passed
    assert result.goal_accepted
    assert client.calls == 3
    assert result.scenario_runs[0].scenario_id == "actual-run"
    assert result.scenario_runs[0].output.strip() == "observed target"


def test_agentic_deliverer_cannot_accept_without_running_the_target(tmp_path):
    class Client:
        async def chat(self, messages, tools=None, *, system_prompt):
            yield {
                "type": "text_delta",
                "text": (
                    '{"ready": true, "failure_kind": "none", '
                    '"missing_objectives": [], "integration_concerns": [], '
                    '"proposal_violations": [], "blocking_evidence": [], '
                    '"summary": "looks fine"}'
                ),
            }

    result = asyncio.run(
        DeliveryCoordinator(
            runner=AcceptanceRunner(timeout_s=10),
            deliverer=Deliverer(client=Client()),
        ).deliver(
            make_proposal(),
            cwd=str(tmp_path),
            loop_diff="diff --git a/agent.py b/agent.py",
        )
    )

    assert not result.passed
    assert not result.goal_accepted
    assert result.failure_kind == "verification_gap"
    assert "without executing any frozen runtime action" in result.goal_review


def test_delivery_coordinator_records_judged_root_cause():
    class FakeRunner:
        async def run(self, proposal, *, cwd):
            return AcceptanceRun(passed=True)

    class FakeDeliverer:
        async def review(self, proposal, *, loop_diff, acceptance):
            return GoalReview(
                accepted=False,
                text='{"ready": false}',
                failure_kind="verification_gap",
            )

    result = asyncio.run(
        DeliveryCoordinator(
            runner=FakeRunner(),
            deliverer=FakeDeliverer(),
        ).deliver(
            make_proposal(),
            cwd="/candidate",
            loop_diff="diff --git a/a.py b/a.py",
        )
    )

    assert result.failure_kind == "verification_gap"
    assert any("root cause: verification_gap" in reason for reason in result.reasons)


def test_delivery_coordinator_classifies_unrelated_failed_safety_as_plan_gap():
    class FakeRunner:
        async def run(self, proposal, *, cwd):
            return AcceptanceRun(
                passed=False,
                failures=[
                    "system safety probe 'adapter:safety:path_confinement': "
                    "exit 1, expected 0"
                ],
            )

    class FakeDeliverer:
        async def review(self, proposal, *, loop_diff, acceptance):
            return GoalReview(
                accepted=True,
                text='{"ready": true, "failure_kind": "none"}',
            )

    result = asyncio.run(
        DeliveryCoordinator(
            runner=FakeRunner(),
            deliverer=FakeDeliverer(),
        ).deliver(
            make_proposal(),
            cwd="/candidate",
            loop_diff="diff --git a/a.py b/a.py",
        )
    )

    assert not result.passed
    assert result.goal_accepted
    assert result.failure_kind == "plan_gap"
