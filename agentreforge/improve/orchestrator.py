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
   Use these runs to diagnose what capability should improve. Every run is tagged
   with `target_commit`, `evidence_source`, and `is_current`.
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
- Treat `is_current=true` runs as evidence about the current candidate commit.
  Baseline or older-commit runs remain useful historical evidence, but they cannot
  prove that a capability is still broken after a delivered Loop. Delivery Scenario
  trajectories from a successful Loop are fed back as `delivered_scenario` runs for
  the new commit and are the freshest behavioral evidence in the next Loop.
- The context contains bounded summaries. Use `get_target_evidence` or
  `get_reforge_loop` only when a cited detail needs inspection.
- Build an Achievement Ledger from `previous_reforge_loops` before diagnosing:
  delivered achievements, rolled-back attempts, remaining gaps, and current source
  evidence. A delivered record is a claim to verify against the current tree, not an
  excuse to repeat the same intervention. If the capability is present, move on. If it
  is present but ineffective, diagnose the next direct cause instead of rewording or
  reimplementing the recorded solution.
- Treat every non-completed Loop's `failure_kind` and `attempt_fingerprint` as a
  Negative-Attempt Ledger. After `verification_gap`, do not repeat the same Candidate
  with the same unavailable evidence requirement. Change the verification mechanism,
  choose a different evidenced Candidate, or abstain. Rewording the same scenario is
  not a new attempt.

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
- Exclude every Candidate whose capability delta is already in the verified Achievement
  Ledger unless new trajectory, test, or source evidence proves that capability remains
  ineffective. In that case target the newly diagnosed direct cause; do not select a
  cosmetic refinement of the completed intervention.
- Capability-level de-duplication is your responsibility, not string equality. Treat
  renamed variants such as "fallback guidance" and "strengthen fallback guidance" as
  the same delivered capability when their mechanism and expected delta are equivalent.
  Reopen a delivered capability only with CURRENT-commit evidence that it remains
  ineffective; stale baseline evidence is insufficient.
- A Delivery command, smoke marker, expected string, report, or documentation change is
  verification infrastructure—not a target-agent capability. Never select work whose
  purpose is merely to make AgentReforge's gate or evaluator pass. It is eligible only
  when independent evidence shows the target agent itself has the corresponding runtime
  or usability defect.

PHASE 4 — SELECT ONE IMPROVEMENT
- Rank candidates by evidence strength, cross-task capability benefit, causal clarity,
  risk, effort, and testability.
- One Loop selects exactly ONE Candidate and produces exactly ONE implementation Task.
  The Task may touch multiple files and complete a coherent cross-component capability,
  but it has one objective, one Reviewer decision, and one Loop commit.
- If the best Candidate is too large for one bounded Task, select the smallest coherent
  capability slice that is independently useful and verifiable. Record deferred slices
  as remaining gaps for later Loops; do not create a multi-Task batch.
- Explain why this Candidate is the highest-value unsolved capability and why it is not
  a duplicate of a delivered achievement.

PHASE 5 — PLAN
- Create exactly one Task and name its one selected Candidate.
- Keep the two dependency namespaces distinct:
  Candidate `dependencies` contain exact Candidate names, while Task `dependencies`
  contain ONLY exact Task `id` values declared in this proposal. Never put a Candidate
  name, description, or capability label in a Task dependency. The single Task is a
  runnable root and therefore has an empty dependency list.
- Then define the write boundary, per-Task acceptance contract, batch-level Delivery
  checks, rollback plan, and final decision.

Rules:
- Use the read-only tools (read_file, search_code, grep, list_dir, glob) to inspect the
  REAL source before proposing. Do not guess.
- Ground every claim in inspectable evidence: a code location, a trajectory record, or a test.
- Evidence and Candidate comparison improve the plan, but they are reasoning aids rather
  than schema gates. Use as much structure as the change genuinely needs.
- You must NOT write or modify any code.
- Score benefit/risk/effort 1-5 and confidence 0.0-1.0 as advisory metadata. Propose
  proceed, abstain, or needs_human explicitly.
- The recursion limit is a ceiling, never a target. Choose `abstain` when there is no
  new evidence-backed unsolved capability, when only completed achievements remain,
  when the only available work would optimize a Delivery/evaluation artifact, or when
  evidence is too weak to identify a direct cause. Do not invent work to consume Loops.
- BEFORE emitting the proposal, call `validate_plan` with your `tasks` (a list of
  {id, dependencies}). If it reports problems (duplicate ids, unknown dependencies,
  cycles, no runnable root), FIX the task graph and validate again. For an unknown Task
  dependency, replace it with an exact declared Task id or remove it when the relationship
  belongs only at Candidate level. Only output the final proposal once validate_plan passes.
- `allowed_write_paths`, `affected_components`, clauses, and acceptance criteria are
  implementation suggestions for Writer/Reviewer, not hard authorization boundaries.
  Keep them useful and concise; do not over-specify a small change.
- Acceptance criteria are review/test hints. Only explicit safety-property checks,
  `delivery_run`, and frozen `delivery_scenarios` participate in Delivery.
- For a target agent with a runnable CLI, prefer one or two `delivery_scenarios` that
  exercise the selected capability end to end. Each scenario contains a frozen prompt,
  a safe argv command using `{prompt}` and `{workspace}`, a small isolated fixture, and
  observable expected/forbidden behaviors. Design scenarios before Writer runs; do not
  adapt acceptance difficulty after seeing the implementation.
- Scenario commands are argv arrays, never shell strings. Inspect the real target
  entrypoint and use `{workspace}` for the disposable task directory. Keep scenarios
  bounded and directly causal: they decide Delivery for this Candidate, not general
  benchmark quality. Optional later diagnostic probes may explain failure but cannot
  change the frozen pass/fail scenarios.
- Set `requires_trajectory=true` whenever a scenario must prove internal tool use,
  inspect-before-edit, verify-after-edit, call ordering, or any other process fact.
  A target's own final response is not evidence that it performed those actions. If
  the target cannot emit trajectory evidence, design artifact/output-observable
  requirements instead; do not claim an unobservable process property.
- When a scenario depends on executable availability, declare only the necessary
  typed `executable_conditions` (`name` plus `available|unavailable`). The trusted
  Runner materializes and records these conditions. Never encode environment setup
  in the prompt, fixture, arbitrary environment variables, or a shell command.
  A scenario with executable conditions must set `requires_trajectory=true`.
- `delivery_run` is SYSTEM input for the deterministic delivery/commit gate: a small
  fallback package-start command when a target-agent scenario cannot be executed. Runner
  commands, exit codes, bounded output, scenario artifacts, and available trajectory are
  supplied to the Delivery Judge as runtime evidence, but a deterministic Runner failure
  cannot be overridden by the Judge. Do not grade general generated-code quality here.
- Verification commands must not leave runtime artifacts in the candidate. Prefer
  `PYTHONDONTWRITEBYTECODE=1 python3` for Python commands (do not assume a `python`
  alias exists) and never include `__pycache__`,
  `*.pyc`, `.pytest_cache`, coverage output, or runtime state files in allowed_write_paths;
  the post-write policy gate hard-denies those artifacts.
- Do NOT create a Writer Task whose job is to prove that Delivery commands leave the
  worktree unchanged. DeliveryCoordinator owns this invariant deterministically by
  comparing pre/post candidate tree snapshots. Tasks may add real product tests or smoke
  scripts, but repository immutability is not a product-code requirement.
- For a genuine safety property, mark its criterion with
  `verified_safety_properties` and use `mode=invariant`. Safety probes are system-owned:
  leave the criterion `command` empty. Never attach a safety property to an ordinary
  capability scenario or generate a prompt asking the target agent to claim that it
  was blocked. The Runner's target adapter executes the actual tool-level traversal
  probe and decides from its exit code. Never encode a condition that rewards the
  safety violation. `{prompt}` and `{workspace}` remain
  scenario-only placeholders and must not appear in `delivery_run`.
- `required_safety_properties` is EMPTY BY DEFAULT. Declare `path_confinement` only
  when the selected Candidate itself implements, modifies, or promises to preserve a
  path-taking/filesystem boundary. Never attach it to an unrelated prompt, runtime
  fallback, model, or workflow change. A Task that declares it must reference one
  matching invariant criterion; the criterion must list `path_confinement` and leave
  `command` empty. Do not require a safety property that the baseline does not have
  unless implementing that property is the Candidate's explicit objective.

When done, output ONLY the proposal as ONE JSON object in a ```json code block. Field types:
- analysis: {
  findings: [{symptom, root_cause, capability_gap, evidence_refs: [str], uncertainty}],
  candidates: [{name, level: prompt|workflow|tool|module|architecture, mechanism,
    expected_capability_delta, evidence_strength: 1-5, benefit: 1-5, risk: 1-5,
    effort: 1-5, dependencies: [candidate_name], conflicts_with: [candidate_name],
    rejected_reason}],
  selected_candidates: [exactly_one_candidate_name],
  batch_budget: {max_candidates, max_tasks, max_total_effort, selected_total_effort},
  packing_reason, compatibility_notes: [str], selection_reason, causal_mechanism,
  expected_capability_delta}
- evidence[]: {source_type: trajectory|test|benchmark|code|log, reference: str, observation: str}
- tasks[] is the shared Writer/Reviewer brief:
  {id: str, candidate: str, description: str, rationale: str, capability_change: str,
  required_behaviors: [{id, description}],
  implementation_constraints: [{id, description}],
  invariants: [{id, description}],
  prohibited_shortcuts: [{id, description}],
  affected_components: [str], reviewer_focus: [str],
  required_safety_properties: [] by default, or [path_confinement] only for a
  path/filesystem-boundary Candidate,
  dependencies: [task_id], acceptance_criteria_ids: [str]}.
  Exactly one Task is allowed per proceeding Loop. Only `id`, `description`, and valid
  Task dependencies are essential. Other fields should clarify the work when useful
  and may remain empty for a simple Task.
- acceptance_criteria[]: {id: str, description: str,
  mode: red_green|invariant|metric_improvement|non_regression|manual,
  check_type: unit|integration|smoke,
  verification: command|review|manual, command: str, expected_exit_code: int,
  required_output_contains: [str], forbidden_output_contains: [str],
  verified_safety_properties: [path_confinement],
  test_level: full|focused|basic, required: bool}
- allowed_write_paths[]: suggested repository areas, not a hard write boundary.
- benefit/risk/effort: int 1-5; confidence: float 0-1
- decision: proceed|abstain|needs_human
- delivery_run[]: system-level integration/smoke commands for the improved agent package.
- delivery_scenarios[]: {id: str, prompt: str,
  command: [argv strings containing {prompt} and {workspace}],
  fixture_files: {repo_relative_path: content},
  expected_behaviors: [str], forbidden_behaviors: [str],
  executable_conditions: [{name: str, state: available|unavailable}],
  requires_trajectory: bool}.
- delivery_checklist[]: high-level system requirements the Deliverer can judge from the
  full diff and scenario evidence (the selected Candidate is implemented, reachable, and
  observed in the intended path). Do not require it to repeat function-level review.
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
            "mapping, update BOTH sides: tag an invariant acceptance criterion with "
            "`verified_safety_properties`, leave its command empty because the probe is "
            "system-owned, and add that exact criterion id to the owning Task's "
            "`acceptance_criteria_ids`. For an unknown Task dependency, use ONLY "
            "an exact `id` from the proposal's `tasks` list; never use a Candidate name "
            "or description as a Task dependency."
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
