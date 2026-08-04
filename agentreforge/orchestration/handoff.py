"""Output-only repair for AgentReforge hand-off contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from ..llm.collect import collect_text
from ..types import Message

HANDOFF_REPAIR_PROMPT = """You repair one agent component's OUTPUT CONTRACT.
Do not edit product code, add requirements, or perform the downstream component's
job. Preserve the available facts and rewrite only the producer's output so it
satisfies the supplied interface. Return only the corrected output."""

HANDOFF_FINALIZER_PROMPT = """You are the output stage of an agent component.
The work/action budget has ended. Produce the required structured handoff from the
immutable task context and authoritative execution artifacts. Do not call tools,
continue implementation, invent edits, or treat a self-reported summary as stronger
than the supplied artifact. Return only one output matching the contract."""


@dataclass
class HandoffRepair:
    text: str
    error: str = ""
    repairs: int = 0


async def finalize_handoff_output(
    client,
    *,
    producer: str,
    contract: str,
    context: dict,
) -> str:
    """Run one budget-independent, tool-free producer output turn."""

    return await collect_text(
        client,
        [
            Message(
                role="user",
                content=json.dumps(
                    {
                        "request_kind": "finalize_handoff",
                        "producer": producer,
                        "output_contract": contract,
                        "immutable_execution_context": context,
                    },
                    ensure_ascii=False,
                ),
            )
        ],
        system_prompt=HANDOFF_FINALIZER_PROMPT,
    )


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
                    content=json.dumps(
                        {
                            "request_kind": "repair_handoff",
                            "producer": producer,
                            "output_contract": contract,
                            "validation_error": error,
                            "immutable_context": context[:12_000],
                            "previous_invalid_output": candidate[:8_000],
                        },
                        ensure_ascii=False,
                    ),
                )
            ],
            system_prompt=HANDOFF_REPAIR_PROMPT,
        )
        error = validate(candidate)
        if not error:
            return HandoffRepair(candidate, repairs=repair_i)
    return HandoffRepair(candidate, error=error, repairs=max_repairs)
