"""Trusted execution substrate used by the agentic Deliverer.

The substrate starts frozen commands/scenarios, confines fixtures, and records
facts.  It does not decide whether those facts prove the improvement goal.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import DeliveryScenario, ImprovementProposal

_OUTPUT_CAP = 4000
_ARTIFACT_CAP = 1000
_MAX_ARTIFACTS = 20
_MAX_TRAJECTORY_EVENTS = 100

# A delivery command should verify the target, never mutate external state.
_DANGEROUS = [
    r"\brm\s+-[a-z]*r",
    r"\bsudo\b",
    r"\bgit\s+(push|reset|clean|checkout|merge|rebase)\b",
    r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh)\b",
    r"\bdd\b|\bmkfs\b",
    r"\bchmod\s+(-R|777)\b|\bchown\b",
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r":\(\)\s*\{",
    r">\s*/(dev|etc|bin|usr|sys|boot)\b",
    r"\b(kill|killall|pkill)\b",
]


@dataclass
class RunResult:
    command: str
    exit_code: int | None
    output: str


@dataclass
class ScenarioRunResult:
    scenario_id: str
    prompt: str
    command: list[str]
    exit_code: int | None
    output: str
    changed_files: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    trajectory: list[dict] = field(default_factory=list)
    trajectory_available: bool = False
    environment_ready: bool = True
    environment_facts: list[dict[str, object]] = field(default_factory=list)


@dataclass
class AcceptanceRun:
    """Execution facts plus universal hard failures.

    ``passed`` means only that no hard execution/safety failure was observed; it
    is never evidence that the selected capability was realized.
    """

    passed: bool
    runs: list[RunResult] = field(default_factory=list)
    scenario_runs: list[ScenarioRunResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class AcceptanceRunner:
    def __init__(
        self,
        *,
        governance: str = "autonomous",
        approve_command: Callable[[str], Awaitable[bool]] | None = None,
        timeout_s: float = 300.0,
    ):
        self.governance = str(governance)
        self.approve_command = approve_command
        self.timeout_s = timeout_s

    async def run_command(self, command: str, *, cwd: str) -> RunResult:
        """Execute one frozen command selected by the Deliverer agent."""

        return await self._safe_run(command, cwd)

    async def run_scenario(
        self,
        scenario: DeliveryScenario,
        *,
        cwd: str,
    ) -> ScenarioRunResult:
        """Execute one frozen scenario selected by the Deliverer agent."""

        return await self._safe_run_scenario(scenario, cwd)

    async def run_safety_probe(self, safety: str, *, cwd: str) -> RunResult:
        """Execute a system-owned universal safety probe."""

        if safety == "path_confinement":
            return await self._run_path_confinement_probe(cwd)
        return RunResult(
            f"adapter:safety:{safety}",
            None,
            f"unsupported system safety probe: {safety}",
        )

    async def run(self, proposal: ImprovementProposal, *, cwd: str) -> AcceptanceRun:
        scenarios = proposal.contract_delivery_scenarios()
        commands = list(proposal.contract_delivery_run())
        commands = list(dict.fromkeys(commands))
        safety_properties = _required_safety_properties(proposal)
        if not commands and not scenarios and not safety_properties:
            return AcceptanceRun(
                passed=False,
                failures=["no executable acceptance command defined"],
            )

        runs = [await self._safe_run(command, cwd) for command in commands]
        if "path_confinement" in safety_properties:
            runs.append(await self._run_path_confinement_probe(cwd))
        scenario_runs = [
            await self._safe_run_scenario(scenario, cwd)
            for scenario in scenarios
        ]
        failures = acceptance_failures(proposal, runs)
        failures.extend(
            f"delivery scenario {run.scenario_id!r}: environment conditions "
            f"could not be materialized ({json.dumps(run.environment_facts)})"
            for run in scenario_runs
            if not run.environment_ready
        )
        failures.extend(
            f"delivery scenario {run.scenario_id!r}: exit {run.exit_code}, expected 0"
            for run in scenario_runs
            if run.environment_ready and run.exit_code != 0
        )
        failures.extend(
            f"system safety probe {run.command!r}: exit {run.exit_code}, expected 0"
            for run in runs
            if run.command.startswith("adapter:safety:")
            and run.exit_code != 0
        )
        return AcceptanceRun(
            passed=not failures,
            runs=runs,
            scenario_runs=scenario_runs,
            failures=failures,
        )

    async def _safe_run(self, command: str, cwd: str) -> RunResult:
        danger = dangerous_command(command)
        if danger is None:
            return await self._run(command, cwd)
        if self.governance == "supervised" and self.approve_command is not None:
            if await self.approve_command(command):
                return await self._run(command, cwd)
            return RunResult(command, None, f"blocked (dangerous, human declined): {danger}")
        return RunResult(command, None, f"blocked (dangerous command denied): {danger}")

    async def _run(self, command: str, cwd: str) -> RunResult:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                output, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_s,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return RunResult(command, None, f"timed out after {self.timeout_s}s")
        except Exception as exc:  # noqa: BLE001 - report spawn failure as a fact
            return RunResult(command, None, f"failed to run: {exc}")

        text = output.decode("utf-8", errors="replace")
        if len(text) > _OUTPUT_CAP:
            text = text[:_OUTPUT_CAP] + "\n...(truncated)"
        return RunResult(command, process.returncode, text)

    async def _safe_run_scenario(
        self,
        scenario: DeliveryScenario,
        cwd: str,
    ) -> ScenarioRunResult:
        rendered_for_policy = shlex.join(scenario.command)
        danger = dangerous_command(rendered_for_policy)
        if danger is not None:
            if self.governance == "supervised" and self.approve_command is not None:
                if not await self.approve_command(rendered_for_policy):
                    return _blocked_scenario(scenario, f"human declined: {danger}")
            else:
                return _blocked_scenario(scenario, f"dangerous command denied: {danger}")
        return await self._run_scenario(scenario, cwd)

    async def _run_scenario(
        self,
        scenario: DeliveryScenario,
        cwd: str,
    ) -> ScenarioRunResult:
        with tempfile.TemporaryDirectory(prefix="agentreforge-delivery-") as temp:
            scenario_root = Path(temp).resolve()
            workspace = scenario_root / "workspace"
            workspace.mkdir()
            initial = _write_fixture(workspace, scenario.fixture_files)
            # Monitoring evidence belongs to the Deliverer, not to the disposable
            # Task Workspace visible to the target agent. Keeping it outside the
            # workspace prevents self-observation, recursive reads, and artifact
            # contamination while preserving the adapter's private write path.
            trajectory_path = scenario_root / "target_trajectory.jsonl"
            prompt = scenario.render_prompt()
            argv = [
                arg.replace("{prompt}", prompt).replace(
                    "{workspace}", str(workspace)
                )
                for arg in scenario.command
            ]
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["AGENTREFORGE_TRAJECTORY_PATH"] = str(trajectory_path)
            environment_ready, environment_facts = _materialize_scenario_environment(
                scenario,
                env,
                scenario_root / "path",
            )
            if not environment_ready:
                return ScenarioRunResult(
                    scenario_id=scenario.id,
                    prompt=prompt,
                    command=argv,
                    exit_code=None,
                    output="scenario environment conditions could not be materialized",
                    environment_ready=False,
                    environment_facts=environment_facts,
                )
            process_argv = argv
            if _is_demo_agent_target(Path(cwd), argv):
                process_argv = [
                    sys.executable,
                    "-m",
                    "agentreforge.improve.demo_agent_adapter",
                    "--target",
                    str(Path(cwd).resolve()),
                    "--workspace",
                    str(workspace),
                    "--prompt",
                    prompt,
                    "--trajectory",
                    str(trajectory_path),
                    "--max-turns",
                    str(scenario.budgets.max_agent_turns),
                    "--max-actions",
                    str(scenario.budgets.max_action_steps),
                ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *process_argv,
                    cwd=cwd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    output, _ = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout_s,
                    )
                    exit_code = process.returncode
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    output = f"timed out after {self.timeout_s}s".encode()
                    exit_code = None
            except Exception as exc:  # noqa: BLE001 - preserve execution evidence
                output = f"failed to run: {exc}".encode()
                exit_code = None

            text = output.decode("utf-8", errors="replace")
            if len(text) > _OUTPUT_CAP:
                text = text[:_OUTPUT_CAP] + "\n...(truncated)"
            changed_files, artifacts = _collect_artifacts(
                workspace,
                initial,
                ignored={trajectory_path},
            )
            trajectory = _load_scenario_trajectory(trajectory_path)
            return ScenarioRunResult(
                scenario_id=scenario.id,
                prompt=prompt,
                command=argv,
                exit_code=exit_code,
                output=text,
                changed_files=changed_files,
                artifacts=artifacts,
                trajectory=trajectory,
                trajectory_available=bool(trajectory),
                environment_ready=True,
                environment_facts=environment_facts,
            )

    async def _run_path_confinement_probe(self, cwd: str) -> RunResult:
        target = Path(cwd).resolve()
        label = "adapter:safety:path_confinement"
        if not _has_demo_agent_adapter(target):
            return RunResult(
                label,
                None,
                "no target adapter can verify path_confinement",
            )
        with tempfile.TemporaryDirectory(prefix="agentreforge-safety-") as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            sentinel = "AGENTREFORGE_OUTSIDE_SENTINEL"
            outside = root / "outside.txt"
            outside.write_text(sentinel, encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            argv = [
                sys.executable,
                "-m",
                "agentreforge.improve.demo_agent_adapter",
                "--target",
                str(target),
                "--workspace",
                str(workspace),
                "--probe-path-confinement",
                "--outside-path",
                "../outside.txt",
                "--sentinel",
                sentinel,
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=cwd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                output, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_s,
                )
                return RunResult(
                    label,
                    process.returncode,
                    output.decode("utf-8", errors="replace")[:_OUTPUT_CAP],
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return RunResult(label, None, f"timed out after {self.timeout_s}s")
            except Exception as exc:  # noqa: BLE001
                return RunResult(label, None, f"failed to run: {exc}")


def dangerous_command(command: str) -> str | None:
    for pattern in _DANGEROUS:
        if re.search(pattern, command, re.IGNORECASE):
            return pattern
    return None


def _has_demo_agent_adapter(target: Path) -> bool:
    return (target / "demo_agent" / "agent.py").is_file()


def _is_demo_agent_target(target: Path, argv: list[str]) -> bool:
    return _has_demo_agent_adapter(target) and any(
        left == "-m" and right == "demo_agent"
        for left, right in zip(argv, argv[1:], strict=False)
    )


def _required_safety_properties(proposal: ImprovementProposal) -> set[str]:
    return {
        str(safety)
        for task in proposal.execution_tasks()
        for safety in task.required_safety_properties
    } | {
        str(safety)
        for criterion in proposal.contract_acceptance_criteria()
        for safety in criterion.verified_safety_properties
    }


def _blocked_scenario(
    scenario: DeliveryScenario,
    reason: str,
) -> ScenarioRunResult:
    return ScenarioRunResult(
        scenario_id=scenario.id,
        prompt=scenario.render_prompt(),
        command=scenario.command,
        exit_code=None,
        output=f"blocked ({reason})",
    )


def _materialize_scenario_environment(
    scenario: DeliveryScenario,
    env: dict[str, str],
    mirror_root: Path,
) -> tuple[bool, list[dict[str, object]]]:
    """Build a PATH that satisfies the Orchestrator's typed executable contract."""

    desired = {item.name: item.state for item in scenario.executable_conditions}
    unavailable = {
        name for name, state in desired.items() if state == "unavailable"
    }
    original_path = env.get("PATH", os.defpath)
    if unavailable:
        env["PATH"] = _path_without_executables(
            original_path,
            unavailable,
            mirror_root,
        )

    facts: list[dict[str, object]] = []
    ready = True
    for name, state in desired.items():
        resolved = shutil.which(name, path=env["PATH"])
        observed = "available" if resolved else "unavailable"
        satisfied = observed == state
        ready = ready and satisfied
        facts.append(
            {
                "name": name,
                "required_state": state,
                "observed_state": observed,
                "resolved_path": resolved or "",
                "satisfied": satisfied,
            }
        )
    return ready, facts


def _path_without_executables(
    original_path: str,
    unavailable: set[str],
    mirror_root: Path,
) -> str:
    """Mirror only PATH directories containing a blocked executable."""

    result: list[str] = []
    mirror_root.mkdir(parents=True, exist_ok=True)
    for index, raw_directory in enumerate(original_path.split(os.pathsep)):
        directory = Path(raw_directory or ".").resolve()
        if not directory.is_dir():
            continue
        contains_blocked = any(
            (directory / name).exists() for name in unavailable
        )
        if not contains_blocked:
            result.append(str(directory))
            continue
        mirror = mirror_root / str(index)
        mirror.mkdir()
        for entry in directory.iterdir():
            if entry.name in unavailable:
                continue
            try:
                if entry.is_file() and os.access(entry, os.X_OK):
                    (mirror / entry.name).symlink_to(entry)
            except OSError:
                continue
        result.append(str(mirror))
    return os.pathsep.join(result)


def _write_fixture(workspace: Path, fixture_files: dict[str, str]) -> dict[str, str]:
    initial: dict[str, str] = {}
    for rel_path, content in fixture_files.items():
        target = (workspace / rel_path).resolve()
        if target != workspace and workspace not in target.parents:
            raise ValueError(f"fixture path escapes workspace: {rel_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        initial[rel_path] = content
    return initial


def _collect_artifacts(
    workspace: Path,
    initial: dict[str, str],
    *,
    ignored: set[Path],
) -> tuple[list[str], dict[str, str]]:
    current: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.resolve() in ignored:
            continue
        rel = str(path.relative_to(workspace))
        try:
            current[rel] = path.read_text(encoding="utf-8")[:_ARTIFACT_CAP]
        except (OSError, UnicodeDecodeError):
            current[rel] = "(binary or unreadable)"
    changed = sorted(
        path
        for path in set(initial) | set(current)
        if initial.get(path) != current.get(path)
    )
    artifacts = {
        path: current.get(path, "(deleted)")
        for path in changed[:_MAX_ARTIFACTS]
    }
    return changed, artifacts


def _load_scenario_trajectory(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(events) >= _MAX_TRAJECTORY_EVENTS:
            break
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def acceptance_failures(
    proposal: ImprovementProposal,
    runs: list[RunResult],
) -> list[str]:
    delivery_run = proposal.contract_delivery_run()
    failures = [
        f"delivery command {run.command!r}: exit {run.exit_code}, expected 0"
        for run in runs
        if run.command in delivery_run and run.exit_code != 0
    ]
    return failures
