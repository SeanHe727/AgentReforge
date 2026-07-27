"""Improvement Orchestrator: the analyst that produces an ImprovementProposal.

Reuses the ReAct loop (query) with a READ-ONLY tool subset so the model can
investigate the actual source, grounded by the execution trajectory and the
user's intent. It does NOT write code (that's the Writer). Its final message is
the ImprovementProposal as JSON, validated via Pydantic — with bounded
self-correcting repair retries if the JSON doesn't satisfy the schema.

Kept separate from Planner on purpose: Planner decomposes a plain goal for the
common /plan and /team execution modes; the Orchestrator is the richer,
evidence-grounded planner for the self-improvement pipeline.
"""

from __future__ import annotations

import json
from typing import Any

from ..agent.query import query
from ..llm.base import LlmClient
from ..llm.collect import collect_text
from ..llm.parse import parse_json_model
from ..observability import traceable
from ..tools.base import Tool, ToolContext, ToolResult, object_schema
from ..tools.registry import ToolRegistry
from ..types import Message
from .context import OrchestratorContext, OrchestratorContextBuilder
from .history_index import ImprovementHistoryIndex
from .models import ImprovementProposal
from .plan_validator import validate_plan_tool
from .records import ReforgeLoopRecord

ORCHESTRATOR_PROMPT = """You are the improvement Orchestrator: an analyst, not an implementer.

You receive two strictly separated histories:
1. `target_agent_runs`: what the TARGET AGENT was asked to do and how it behaved.
   Use these runs to diagnose what capability should improve.
2. `previous_reforge_loops`: what AgentReforge itself planned, wrote, reviewed,
   delivered, and committed earlier in THIS recursive run. Use these to avoid
   repetition and plan the next improvement. Never treat Reforge workflow events as
   target-agent behavior.

You also receive a deterministic repository map and READ-ONLY code tools. Produce an
evidence-grounded ImprovementProposal only after completing this workflow:

PHASE 1 — ORIENT
- State internally what the target agent was asked to do, its observable outcome,
  the current agent architecture, and what previous Reforge loops already changed.
- If the target task or outcome is missing, reduce confidence; do not invent it.
- The context contains bounded summaries. Use `get_target_evidence` or
  `get_reforge_loop` only when a cited detail needs inspection.

PHASE 2 — DIAGNOSE
- Identify 1-3 capability gaps. For each, distinguish observed symptom, likely root
  cause, missing capability, evidence references, and uncertainty.
- A missing action in one trajectory is not automatically a missing system feature.
  Check the source to learn whether the failure is instruction, workflow, tool, state,
  or architecture related.
- Current-run facts are authoritative and already supplied directly. Use
  `search_improvement_history` only after an initial diagnosis, to find analogous
  OLD runs; retrieval results are supporting experience, not proof of current behavior.

PHASE 3 — GENERATE CANDIDATES
- Compare viable interventions at different leverage levels: prompt/instruction,
  workflow/control flow, and tool/module/architecture where supported by evidence.
- Do not default to a prompt tweak merely because it is cheap. Do not force a new
  module when the evidence supports a smaller fix.
- Reject changes that only patch the observed example without a reusable mechanism.

PHASE 4 — PACK THE IMPROVEMENT BATCH
- Rank candidates by evidence strength, cross-task capability benefit, causal clarity,
  risk, effort, and testability.
- A Loop is one bounded Improvement Batch and may select 1-3 Candidates.
- If the highest-value Candidate is large (effort > 2), normally select it alone and
  split it into multiple independently reviewable Tasks.
- If the highest-value Candidates are small (effort <= 2), pack up to three into the
  same Loop even when they address different objectives, provided they do not conflict,
  fit the total budget, and each has an independently verifiable Task contract.
- Candidate coherence is NOT required. Compatibility, bounded scope, explicit ownership,
  and independent acceptance are required.
- Never pack Candidates merely because they rank highly: check file/interface conflicts,
  dependencies, aggregate risk, aggregate effort, and whether one Candidate invalidates
  another's acceptance evidence.
- Explain the packing decision and causal chain for every selected Candidate.

PHASE 5 — PLAN
- Only now create the Task DAG. Every Task MUST name its owning selected Candidate.
  Multiple Tasks may implement one large Candidate; separate small Candidates normally
  receive separate Tasks.
- Keep the two dependency namespaces distinct:
  Candidate `dependencies` contain exact Candidate names, while Task `dependencies`
  contain ONLY exact Task `id` values declared in this proposal. Never put a Candidate
  name, description, or capability label in a Task dependency. A runnable root Task has
  an empty dependency list.
- Then define the write boundary, per-Task acceptance contract, batch-level Delivery
  checks, rollback plan, and final decision.

Rules:
- Use the read-only tools (read_file, search_code, grep, list_dir, glob) to inspect the
  REAL source before proposing. Do not guess.
- Ground every claim in inspectable evidence: a code location, a trajectory record, or a test.
- evidence[] MUST be non-empty: include at least one entry with source_type "code" whose
  reference is a concrete file path (ideally file:line) you actually inspected. A proposal
  with no evidence will be rejected by the policy gate — never leave evidence empty.
- You must NOT write or modify any code.
- Score benefit/risk/effort 1-5 and confidence 0.0-1.0. Propose a decision
  (proceed / abstain / needs_human); a deterministic policy gate makes the final call.
- Prefer abstain when evidence is weak or expected value is low.
- BEFORE emitting the proposal, call `validate_plan` with your `tasks` (a list of
  {id, dependencies}). If it reports problems (duplicate ids, unknown dependencies,
  cycles, no runnable root), FIX the task graph and validate again. For an unknown Task
  dependency, replace it with an exact declared Task id or remove it when the relationship
  belongs only at Candidate level. Only output the final proposal once validate_plan passes.
- Define the approved write boundary in `allowed_write_paths`: repo-relative files,
  directory prefixes ending in "/", or glob patterns. Keep authorization narrow around
  the selected Improvement Batch; do not use this safety rule to bias solution selection
  toward superficial local edits.
- Every Task's `affected_components` MUST contain concrete repo-relative file paths
  covered by `allowed_write_paths`. If a Task will create a file, choose its exact path
  now and include that path in both fields; placeholders such as "new test path" are
  invalid.
- Define a traceable acceptance contract. Every task references one or more criterion
  ids; every required criterion is assigned to a task. Required criteria should use
  `verification: "command"` with a safe, concrete command. The pipeline validates this
  contract and runs the policy gate BEFORE any Writer starts.
- `acceptance_criteria` are hard checks of the changed TARGET AGENT SYSTEM itself:
  imports, component behavior, tool interfaces, CLI smoke, and integration consistency.
- `delivery_run` is SYSTEM delivery input for a deterministic command hard gate: a small
  read-only integration/smoke run proving the improved agent package starts. The Deliverer
  LLM does not interpret its output; it separately reviews the full diff against your goal.
  Do not grade generated-code quality here. Capability evaluation is a separate post-run
  harness after AgentReforge has produced a candidate commit.
- Verification commands must not leave runtime artifacts in the candidate. Prefer
  `PYTHONDONTWRITEBYTECODE=1 python3` for Python commands (do not assume a `python`
  alias exists) and never include `__pycache__`,
  `*.pyc`, `.pytest_cache`, coverage output, or runtime state files in allowed_write_paths;
  the post-write policy gate hard-denies those artifacts.
- Do NOT create a Writer Task whose job is to prove that Delivery commands leave the
  worktree unchanged. DeliveryCoordinator owns this invariant deterministically by
  comparing pre/post candidate tree snapshots. Tasks may add real product tests or smoke
  scripts, but repository immutability is not a product-code requirement.
- A Task that adds or changes file, directory, search, or other path-taking tools MUST
  declare `required_safety_properties: ["path_confinement"]`. Assign it an executable
  acceptance criterion with `verified_safety_properties: ["path_confinement"]` that
  attempts a relative `..` escape and emits a stable blocked/error marker listed in
  `required_output_contains`. The exact id of that safety criterion MUST also appear
  in that same Task's `acceptance_criteria_ids`; declaring the property on only one
  side is invalid. For now this relative traversal check is the minimum; do not expand
  it into a comprehensive security suite.

When done, output ONLY the proposal as ONE JSON object in a ```json code block. Field types:
- analysis: {
  findings: [{symptom, root_cause, capability_gap, evidence_refs: [str], uncertainty}],
  candidates: [{name, level: prompt|workflow|tool|module|architecture, mechanism,
    expected_capability_delta, evidence_strength: 1-5, benefit: 1-5, risk: 1-5,
    effort: 1-5, dependencies: [candidate_name], conflicts_with: [candidate_name],
    rejected_reason}],
  selected_candidates: [candidate_name],
  batch_budget: {max_candidates, max_tasks, max_total_effort, selected_total_effort},
  packing_reason, compatibility_notes: [str], selection_reason, causal_mechanism,
  expected_capability_delta}
- evidence[]: {source_type: trajectory|test|benchmark|code|log, reference: str, observation: str}
- tasks[] is the immutable contract shared by Writer and Reviewer:
  {id: str, candidate: str, description: str, rationale: str, capability_change: str,
  required_behaviors: [{id, description}],
  implementation_constraints: [{id, description}],
  invariants: [{id, description}],
  prohibited_shortcuts: [{id, description}],
  affected_components: [str], reviewer_focus: [str],
  required_safety_properties: [path_confinement],
  dependencies: [task_id], acceptance_criteria_ids: [str]}.
  Use unique clause ids within each task (for example RB1, CON1, INV1, PS1).
  `description` is the concrete objective; `capability_change` states what reusable
  target-agent ability changes; required behaviors must be observable; constraints
  define implementation boundaries; invariants preserve existing behavior; prohibited
  shortcuts name plausible surface-level implementations that must not pass. The Writer
  implements this contract and the Reviewer audits the exact same clauses.
- acceptance_criteria[]: {id: str, description: str,
  mode: red_green|invariant|metric_improvement|non_regression|manual,
  check_type: unit|integration|smoke,
  verification: command|review|manual, command: str, expected_exit_code: int,
  required_output_contains: [str], forbidden_output_contains: [str],
  verified_safety_properties: [path_confinement],
  test_level: full|focused|basic, required: bool}
- allowed_write_paths[]: the exact approved write scope.
- benefit/risk/effort: int 1-5; confidence: float 0-1
- decision: proceed|abstain|needs_human
- delivery_run[]: system-level integration/smoke commands for the improved agent package.
- delivery_checklist[]: high-level system requirements the Deliverer can judge from the
  full diff (every selected Candidate is implemented and wired, the combined Batch is
  compatible, and there is no obvious cross-component regression). Do not require it to
  infer runtime facts or repeat per-Task review.
- also: summary, problem_statement, goals[], non_goals[], affected_components[],
  dependencies[], decision_reason, rollback_plan, alternatives_considered[]"""


class Orchestrator:
    def __init__(
        self,
        client: LlmClient,
        registry: ToolRegistry,
        cwd: str,
        *,
        code_index: Any = None,
        history_index: ImprovementHistoryIndex | None = None,
    ):
        self.client = client
        self.cwd = cwd
        self.code_index = code_index
        self.history_index = history_index
        # analyst gets read-only tools ONLY — it must never modify code.
        self.registry = _read_only_registry(registry)
        # plus a deterministic DAG check it must run before finalizing its plan.
        self.registry.register(validate_plan_tool)

    @traceable(name="orchestrator.analyze", run_type="chain")
    async def analyze(
        self,
        intent: str,
        target_trajectory: list[dict[str, Any]],
        loop_history: list[ReforgeLoopRecord] | None = None,
        max_turns: int = 12,
        context: OrchestratorContext | None = None,
    ) -> ImprovementProposal:
        context = context or OrchestratorContextBuilder(self.cwd).build(
            intent=intent,
            target_trajectory=target_trajectory,
            previous_reforge_loops=loop_history or [],
            run_manifest={},
        )
        # The initial prompt contains a bounded evidence catalog. This read-only
        # drill-down tool exposes the full selected TARGET event by stable id.
        self.registry.register(_evidence_tool(context))
        self.registry.register(_reforge_loop_tool(context))
        if self.history_index is not None:
            self.registry.register(_history_tool(context, self.history_index))
        # 1. run the ReAct loop; collect its final text (the proposal JSON).
        text = await self._investigate(context, max_turns)
        # 2. Schema failures happen before the Pipeline can request a semantic
        # proposal revision, so repair them here with a small bounded retry budget.
        return await self._parse_proposal_with_repair(text)

    async def revise(self, proposal: ImprovementProposal, problem: str) -> ImprovementProposal:
        """Re-generate the proposal to fix a specific problem (e.g. an invalid DAG)."""
        # feed the problem + the previous proposal back; ask for a corrected one.
        msg = (
            f"Your previous proposal had a problem that must be fixed:\n{problem}\n\n"
            f"Previous proposal:\n{proposal.model_dump_json(indent=2)}\n\n"
            "Output ONLY a corrected proposal as a single ```json block — fix the "
            "problem and keep everything else valid. For a missing safety-property "
            "mapping, update BOTH sides: tag an executable acceptance criterion with "
            "`verified_safety_properties`, and add that exact criterion id to the owning "
            "Task's `acceptance_criteria_ids`. For an unknown Task dependency, use ONLY "
            "an exact `id` from the proposal's `tasks` list; never use a Candidate name "
            "or description as a Task dependency. When a path_confinement criterion says "
            "it must exercise traversal, its executable `command` MUST literally pass a "
            "`..` path to the changed path-taking tool and assert the declared stable "
            "blocked/error output marker."
        )
        text = await collect_text(
            self.client, [Message(role="user", content=msg)], system_prompt=ORCHESTRATOR_PROMPT
        )
        return await self._parse_proposal_with_repair(text)

    async def _parse_proposal_with_repair(
        self,
        text: str,
        *,
        max_repairs: int = 2,
    ) -> ImprovementProposal:
        """Parse a proposal, feeding exact schema errors back with a bounded budget."""

        candidate = text
        for repair_i in range(max_repairs + 1):
            try:
                return parse_json_model(candidate, ImprovementProposal)
            except ValueError as exc:
                if repair_i >= max_repairs:
                    raise
                candidate = await self._repair(candidate, str(exc))
        raise AssertionError("unreachable")

    async def _investigate(
        self,
        context: OrchestratorContext,
        max_turns: int,
    ) -> str:
        text = ""
        async for event in query(
            client=self.client,
            registry=self.registry,
            system_prompt=ORCHESTRATOR_PROMPT,
            user_message=_build_request(context),
            cwd=self.cwd,
            code_index=self.code_index,
            max_turns=max_turns,
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text

    async def _repair(self, bad_output: str, error: str) -> str:
        # feed the model its own output + the validation error; ask for corrected JSON.
        msg = (
            f"Your JSON did not satisfy the schema:\n{error}\n\n"
            f"Your previous output was:\n{bad_output}\n\n"
            "Output ONLY the corrected proposal as a single ```json block that matches "
            "the schema exactly (fix the field types shown in the error)."
        )
        return await collect_text(
            self.client, [Message(role="user", content=msg)], system_prompt=ORCHESTRATOR_PROMPT
        )


def _read_only_registry(registry: ToolRegistry) -> ToolRegistry:
    read_only = ToolRegistry()
    read_only.register_all(
        [t for name in registry.list_names() if (t := registry.get(name)) and t.is_read_only]
    )
    return read_only


def _build_request(context: OrchestratorContext) -> str:
    """Serialize one mandatory, typed management view instead of raw mixed logs."""

    return (
        "OrchestratorContext (deterministically prepared; trajectory kinds are separate):\n"
        f"```json\n{json.dumps(context.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n"
        "```\n\nFollow PHASES 1-5 in order. Use evidence references from the context and "
        "inspect the real source with read-only tools. Then output the final "
        "ImprovementProposal JSON."
    )


def _evidence_tool(context: OrchestratorContext) -> Tool:
    async def get_evidence(args: dict, _tool_context: ToolContext) -> ToolResult:
        event_id = str(args.get("event_id") or "").strip()
        if not event_id:
            return ToolResult(content="Error: 'event_id' is required.", is_error=True)
        record = context.raw_target_evidence.get(event_id)
        if record is None:
            return ToolResult(
                content=f"Error: target-agent evidence not found: {event_id}",
                is_error=True,
            )
        return ToolResult(content=json.dumps(record, ensure_ascii=False, indent=2))

    return Tool(
        name="get_target_evidence",
        description=(
            "Read one full TARGET AGENT trajectory event by an event_id listed in "
            "OrchestratorContext.evidence_catalog. This never returns AgentReforge "
            "workflow events."
        ),
        parameters=object_schema(
            {
                "event_id": {
                    "type": "string",
                    "description": "Stable target-agent evidence event id",
                }
            },
            required=["event_id"],
        ),
        handler=get_evidence,
    )


def _history_tool(
    context: OrchestratorContext,
    history_index: ImprovementHistoryIndex,
) -> Tool:
    async def search_history(args: dict, _tool_context: ToolContext) -> ToolResult:
        query_text = str(args.get("query") or "").strip()
        if not query_text:
            return ToolResult(content="Error: 'query' is required.", is_error=True)
        matches = history_index.search(
            query_text,
            target_repo=context.run_manifest.get("target") or context.repository.root,
            exclude_run_id=str(context.run_manifest.get("run_id") or ""),
            limit=int(args.get("limit") or 5),
        )
        return ToolResult(
            content=(
                json.dumps(matches, ensure_ascii=False, indent=2)
                if matches
                else "No relevant historical AgentReforge loops found."
            )
        )

    return Tool(
        name="search_improvement_history",
        description=(
            "Search OLDER AgentReforge loop records for similar capability gaps, "
            "interventions, reviewer findings, or delivery outcomes. Never use this "
            "instead of the directly supplied current target run/current-loop facts."
        ),
        parameters=object_schema(
            {
                "query": {
                    "type": "string",
                    "description": "Capability gap, failure pattern, or intervention to find",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches (default 5, maximum 20)",
                },
            },
            required=["query"],
        ),
        handler=search_history,
    )


def _reforge_loop_tool(context: OrchestratorContext) -> Tool:
    async def get_loop(args: dict, _tool_context: ToolContext) -> ToolResult:
        loop_id = str(args.get("loop_id") or "").strip()
        if not loop_id:
            return ToolResult(content="Error: 'loop_id' is required.", is_error=True)
        record = context.raw_reforge_loops.get(loop_id)
        if record is None:
            return ToolResult(
                content=f"Error: AgentReforge loop not found: {loop_id}",
                is_error=True,
            )
        return ToolResult(content=record.model_dump_json(indent=2))

    return Tool(
        name="get_reforge_loop",
        description=(
            "Read the full component-divided record for a previous AgentReforge loop "
            "in THIS recursive run. It is workflow audit data, not target-agent evidence."
        ),
        parameters=object_schema(
            {
                "loop_id": {
                    "type": "string",
                    "description": "Loop id listed in previous_reforge_loops",
                }
            },
            required=["loop_id"],
        ),
        handler=get_loop,
    )
