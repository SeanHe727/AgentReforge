"""Output-only repair for AgentReforge hand-off contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..llm.collect import collect_text
from ..types import Message

HANDOFF_REPAIR_PROMPT = """You repair one agent component's OUTPUT CONTRACT.
Do not edit product code, add requirements, or perform the downstream component's
job. Preserve the available facts and rewrite only the producer's output so it
satisfies the supplied interface. Return only the corrected output."""


@dataclass
class HandoffRepair:
    text: str
    error: str = ""
    repairs: int = 0


async def repair_handoff_output(
    client,
    *,
    producer: str,
    invalid_output: str,
    validation_error: str,
    contract: str,
    context: str,
    validate: Callable[[str], str],
    max_repairs: int = 2,
) -> HandoffRepair:
    """Ask the producing output module to rewrite its interface, not its work."""

    candidate = invalid_output
    error = validation_error
    for repair_i in range(1, max_repairs + 1):
        candidate = await collect_text(
            client,
            [
                Message(
                    role="user",
                    content=(
                        f"Producer: {producer}\n\n"
                        f"Output contract:\n{contract}\n\n"
                        f"Validation error:\n{error}\n\n"
                        f"Immutable context:\n{context[:12_000]}\n\n"
                        f"Previous invalid output:\n{candidate[:8_000]}"
                    ),
                )
            ],
            system_prompt=HANDOFF_REPAIR_PROMPT,
        )
        error = validate(candidate)
        if not error:
            return HandoffRepair(candidate, repairs=repair_i)
    return HandoffRepair(candidate, error=error, repairs=max_repairs)
