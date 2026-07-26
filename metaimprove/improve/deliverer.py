"""The Deliverer — execute the frozen acceptance contract, then review the candidate.

Required command criteria are deterministic hard gates.  The LLM then reviews
their evidence and the whole-loop diff for integration concerns; it is never the
only acceptance signal.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..llm.collect import collect_text
from ..observability import traceable
from ..types import Message
from .models import ImprovementProposal

_OUTPUT_CAP = 4000

# clearly destructive / exfiltrating / state-mutating command patterns. A delivery
# run should only exercise the target, never do these — so they are denied.
_DANGEROUS = [
    r"\brm\s+-[a-z]*r",  # rm -r / -rf
    r"\bsudo\b",
    r"\bgit\s+(push|reset|clean|checkout|merge|rebase)\b",  # destructive/state-changing git
    r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh)\b",  # pipe a download into a shell
    r"\bdd\b|\bmkfs\b",
    r"\bchmod\s+(-R|777)\b|\bchown\b",
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r":\(\)\s*\{",  # fork bomb
    r">\s*/(dev|etc|bin|usr|sys|boot)\b",  # redirect over system paths
    r"\b(kill|killall|pkill)\b",
]


def _dangerous(command: str) -> str | None:
    """Return the matched dangerous pattern, or None if the command looks safe."""
    for pat in _DANGEROUS:
        if re.search(pat, command, re.IGNORECASE):
            return pat
    return None

DELIVERER_PROMPT = """You are the Deliverer: the final, WHOLE-CANDIDATE acceptance reviewer.
You get (1) the OUTPUT of running the candidate once, (2) a checklist, and (3) the
full diff of this improvement loop. Judge the candidate as a whole:

A. Checklist — for each item, judge ONLY from the run output: write
   `PASS <item>` or `FAIL <item> - <reason>`.
B. Whole-project review — inspect the loop DIFF for problems the per-task reviewer
   can't see: broken or inconsistent interfaces, cross-cutting regressions, style
   or structure that doesn't fit the codebase, out-of-scope or risky changes. List
   concrete concerns, or write "project review: none".

Final line: exactly `VERDICT: ACCEPT` (only if every checklist item passed AND there
are no blocking project concerns) or `VERDICT: REJECT`."""


@dataclass
class RunResult:
    command: str
    exit_code: int | None
    output: str


@dataclass
class Delivery:
    passed: bool
    runs: list[RunResult] = field(default_factory=list)
    hard_gate_ok: bool = True
    integrity_ok: bool = True
    checklist_review: str = ""
    reasons: list[str] = field(default_factory=list)
    mutation_diff: str = ""
    verified_tree: str = ""


class Deliverer:
    def __init__(
        self,
        *,
        client: Any,
        governance: str = "autonomous",
        approve_command: Callable[[str], Awaitable[bool]] | None = None,
        timeout_s: float = 300.0,
    ):
        self.client = client
        self.governance = str(governance)
        self.approve_command = approve_command
        self.timeout_s = timeout_s

    @traceable(name="deliverer.deliver", run_type="chain")
    async def deliver(
        self, proposal: ImprovementProposal, *, cwd: str, loop_diff: str = ""
    ) -> Delivery:
        """Run each unique acceptance command once, then review the full candidate."""
        commands = list(proposal.delivery_run)
        commands.extend(
            criterion.command
            for criterion in proposal.acceptance_criteria
            if criterion.verification == "command" and criterion.command.strip()
        )
        commands = list(dict.fromkeys(commands))
        # nothing to run -> can't deliver.
        if not commands:
            return Delivery(
                passed=False,
                hard_gate_ok=False,
                reasons=["no executable acceptance command defined"],
            )

        # Execute each unique command once (denying dangerous ones), collect output.
        runs = [await self._safe_run(cmd, cwd) for cmd in commands]

        # Deterministic hard gate: command exit codes and declared output assertions.
        failures = _acceptance_failures(proposal, runs)
        hard_ok = not failures

        # The LLM judges integration semantics and the whole-loop diff.
        checklist = list(proposal.delivery_checklist)
        checklist.extend(
            criterion.description
            for criterion in proposal.acceptance_criteria
            if criterion.required
        )
        checklist = list(dict.fromkeys(checklist))
        review, review_ok = await self._review(checklist, runs, loop_diff)

        passed = hard_ok and review_ok
        reasons = list(failures)
        if not hard_ok:
            reasons.insert(0, "hard gate failed")
        if not review_ok:
            reasons.append("delivery review rejected (checklist or project concern)")
        return Delivery(
            passed=passed,
            runs=runs,
            hard_gate_ok=hard_ok,
            checklist_review=review,
            reasons=reasons or ["hard gate + checklist + project review passed"],
        )

    async def _safe_run(self, command: str, cwd: str) -> RunResult:
        """Deny dangerous commands (autonomous rejects; supervised asks a human)."""
        danger = _dangerous(command)
        if danger is None:
            return await self._run(command, cwd)
        if self.governance == "supervised" and self.approve_command is not None:
            if await self.approve_command(command):
                return await self._run(command, cwd)
            return RunResult(command, None, f"blocked (dangerous, human declined): {danger}")
        # autonomous, or supervised with no approver -> refuse. A None exit fails the
        # hard gate, so a dangerous command blocks delivery.
        return RunResult(command, None, f"blocked (dangerous command denied): {danger}")

    async def _run(self, command: str, cwd: str) -> RunResult:
        """Run one command, capturing combined output (a spawn failure is exit None)."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return RunResult(command, None, f"timed out after {self.timeout_s}s")
        except Exception as exc:  # noqa: BLE001 - surface a spawn failure, don't crash
            return RunResult(command, None, f"failed to run: {exc}")

        text = out.decode("utf-8", errors="replace")
        if len(text) > _OUTPUT_CAP:
            text = text[:_OUTPUT_CAP] + "\n...(truncated)"
        return RunResult(command, proc.returncode, text)

    async def _review(
        self, checklist: list[str], runs: list[RunResult], loop_diff: str
    ) -> tuple[str, bool]:
        """One LLM pass: checklist vs run output + whole-project review of the diff."""
        # nothing to judge -> the hard gate stands alone.
        if not checklist and not loop_diff.strip():
            return "(no checklist or diff to review)", True

        # assemble the single run transcript + the loop diff for the LLM.
        transcript = "\n\n".join(
            f"$ {r.command}\n(exit {r.exit_code})\n{r.output}" for r in runs
        )
        items = "\n".join(f"- {c}" for c in checklist) or "- (none)"
        diff = loop_diff[:_OUTPUT_CAP]
        if len(loop_diff) > _OUTPUT_CAP:
            diff += "\n...(truncated)"
        message = (
            f"Run output:\n{transcript}\n\nChecklist:\n{items}\n\nLoop diff:\n{diff}"
        )
        text = await collect_text(
            self.client, [Message(role="user", content=message)], system_prompt=DELIVERER_PROMPT
        )
        # accept only on an explicit ACCEPT verdict.
        accept = "VERDICT: ACCEPT" in text.upper() and "VERDICT: REJECT" not in text.upper()
        return text, accept


def _acceptance_failures(
    proposal: ImprovementProposal, runs: list[RunResult]
) -> list[str]:
    """Evaluate the executable acceptance criteria without LLM judgement."""
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
