"""Deterministic execution of the frozen target-system acceptance contract."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .models import ImprovementProposal

_OUTPUT_CAP = 4000

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
class AcceptanceRun:
    """Facts from executing the frozen command contract; no LLM judgement."""

    passed: bool
    runs: list[RunResult] = field(default_factory=list)
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

    async def run(self, proposal: ImprovementProposal, *, cwd: str) -> AcceptanceRun:
        commands = list(proposal.delivery_run)
        commands.extend(
            criterion.command
            for criterion in proposal.acceptance_criteria
            if criterion.verification == "command" and criterion.command.strip()
        )
        commands = list(dict.fromkeys(commands))
        if not commands:
            return AcceptanceRun(
                passed=False,
                failures=["no executable acceptance command defined"],
            )

        runs = [await self._safe_run(command, cwd) for command in commands]
        failures = acceptance_failures(proposal, runs)
        return AcceptanceRun(passed=not failures, runs=runs, failures=failures)

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


def dangerous_command(command: str) -> str | None:
    for pattern in _DANGEROUS:
        if re.search(pattern, command, re.IGNORECASE):
            return pattern
    return None


def acceptance_failures(
    proposal: ImprovementProposal,
    runs: list[RunResult],
) -> list[str]:
    by_command = {run.command: run for run in runs}
    failures = [
        f"delivery command {run.command!r}: exit {run.exit_code}, expected 0"
        for run in runs
        if run.command in proposal.delivery_run and run.exit_code != 0
    ]
    for criterion in proposal.acceptance_criteria:
        if not criterion.required or criterion.verification != "command":
            continue
        run = by_command.get(criterion.command)
        if run is None:
            failures.append(f"{criterion.id}: command was not run")
            continue
        if run.exit_code != criterion.expected_exit_code:
            failures.append(
                f"{criterion.id}: exit {run.exit_code}, expected {criterion.expected_exit_code}"
            )
        for expected in criterion.required_output_contains:
            if expected not in run.output:
                failures.append(f"{criterion.id}: output missing {expected!r}")
        for forbidden in criterion.forbidden_output_contains:
            if forbidden in run.output:
                failures.append(f"{criterion.id}: output contains forbidden {forbidden!r}")
    return failures
