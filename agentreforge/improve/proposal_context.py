"""Read-only, section-scoped access to frozen Proposal context."""

from __future__ import annotations

import json
from typing import Any

from ..tools.base import Tool, ToolContext, ToolResult, object_schema
from .models import ImprovementProposal

PROPOSAL_SECTIONS = (
    "goals",
    "non_goals",
    "guardrails",
    "evidence",
    "dependencies",
    "alternatives",
)


def proposal_section(
    proposal: ImprovementProposal,
    section: str,
) -> Any:
    """Return one global context section without exposing mutable Proposal state."""

    if section == "goals":
        return list(proposal.goals)
    if section == "non_goals":
        return list(proposal.non_goals)
    if section == "guardrails":
        return [
            item.model_dump(mode="json") for item in proposal.proposal_guardrails
        ]
    if section == "evidence":
        return [item.model_dump(mode="json") for item in proposal.evidence]
    if section == "dependencies":
        return list(proposal.dependencies)
    if section == "alternatives":
        return list(proposal.alternatives_considered)
    raise KeyError(section)


def proposal_lookup_tool(proposal: ImprovementProposal) -> Tool:
    """Build a read-only lookup tool scoped to one frozen Proposal object."""

    async def read_proposal(
        args: dict[str, Any],
        _context: ToolContext,
    ) -> ToolResult:
        section = str(args.get("section") or "").strip()
        if section not in PROPOSAL_SECTIONS:
            return ToolResult(
                content=f"Error: unknown Proposal section {section!r}.",
                is_error=True,
            )
        return ToolResult(
            content=json.dumps(
                {
                    "section": section,
                    "content": proposal_section(proposal, section),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return Tool(
        name="read_proposal",
        description=(
            "Read one frozen whole-picture Proposal section for context or consistency "
            "verification. The Change Contract remains the direct implementation task."
        ),
        parameters=object_schema(
            {
                "section": {
                    "type": "string",
                    "enum": list(PROPOSAL_SECTIONS),
                    "description": "Frozen Proposal section to inspect",
                }
            },
            required=["section"],
        ),
        handler=read_proposal,
        is_read_only=True,
    )
