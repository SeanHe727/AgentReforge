from __future__ import annotations

import asyncio
import json
from pathlib import Path

from conftest import make_proposal

from agentreforge.agent.reviewer import Review
from agentreforge.improve.proposal_context import proposal_lookup_tool
from agentreforge.improve.reviewer import (
    AgenticReviewer,
    _review_output_error,
    review_registry,
)
from agentreforge.improve.run_config import DetailLevel, profile_for
from agentreforge.improve.writer_reviewer import (
    WriterPlan,
    _compact_contract_from_spec,
    _task_brief,
    _task_contract_text,
    _writer_plan_error,
)
from agentreforge.orchestration.task_executor import (
    TaskExecutor,
    _declared_affected_components,
    _outside_declared_scope,
)
from agentreforge.tools.base import Tool, ToolContext, ToolResult, object_schema
from agentreforge.tools.registry import ToolRegistry


def test_writer_and_reviewer_share_one_traceable_task_contract():
    proposal = make_proposal()
    task = proposal.tasks[0]
    criteria = {criterion.id: criterion for criterion in proposal.acceptance_criteria}

    contract = _task_contract_text(task)
    reviewer_brief = _task_brief(task, criteria)

    assert "Shared Task Contract [implement]" in contract
    assert "Required review clause ids: RB1, INV1" in contract
    assert "[RB1] Handle the missing case." in contract
    assert "[INV1] Preserve existing behavior." in contract
    reviewer_payload = json.loads(reviewer_brief)
    compact = reviewer_payload["writer_task_contract"]
    assert compact["contract_id"] == task.id
    assert compact["objective"] == task.description
    assert compact["requirements"][0]["id"] == "RB1"
    assert compact["constraints"][0]["id"] == "INV1"
    assert compact["acceptance_checks"][0]["id"] == "ac1"
    assert "reviewer_focus" not in compact


def test_compact_writer_contract_excludes_audit_only_fields():
    proposal = make_proposal()
    task = proposal.tasks[0]
    criteria = {criterion.id: criterion for criterion in proposal.acceptance_criteria}

    contract = _compact_contract_from_spec(task, criteria).model_dump(mode="json")

    assert set(contract) == {
        "contract_id",
        "objective",
        "capability_delta",
        "implementation_direction",
        "requirements",
        "constraints",
        "non_goals",
        "suggested_components",
        "required_safety_properties",
        "acceptance_checks",
    }
    assert "rationale" not in contract
    assert "reviewer_focus" not in contract


def test_standard_profile_separates_cycle_and_stage_turn_budgets():
    profile = profile_for(DetailLevel.STANDARD)

    assert profile.max_review_cycles == 3
    assert profile.writer_plan_turns == 4
    assert profile.writer_attempt_turns == 12
    assert profile.reviewer_pass_turns == 6


def test_writer_plan_is_short_and_typed():
    plan = WriterPlan.model_validate(
        {
            "approach": "Inspect state flow, implement it, then run focused tests.",
            "steps": [
                {
                    "id": "P1",
                    "action": "Inspect the completion state flow.",
                    "target_components": ["agent.py"],
                    "verification": "Identify the current stop transition.",
                },
                {
                    "id": "P2",
                    "action": "Implement and verify the bounded state transition.",
                    "target_components": ["agent.py", "test_agent.py"],
                    "verification": "Run the focused unit test.",
                },
            ],
        }
    )

    assert _writer_plan_error(plan.model_dump_json()) == ""


def test_writer_prompt_treats_affected_components_as_suggestions():
    from agentreforge.improve.writer_reviewer import WRITER_PROMPT_SUFFIX

    assert "Affected components are suggestions" in WRITER_PROMPT_SUFFIX


def test_proposal_lookup_is_read_only_and_section_scoped(tmp_path):
    proposal = make_proposal(
        goals=["Improve reusable completion behavior."],
        non_goals=["Do not implement a benchmark task feature."],
    )
    tool = proposal_lookup_tool(proposal)

    result = asyncio.run(
        tool.handler(
            {"section": "non_goals"},
            ToolContext(cwd=str(tmp_path)),
        )
    )

    payload = json.loads(result.content)
    assert tool.is_read_only
    assert payload == {
        "section": "non_goals",
        "content": ["Do not implement a benchmark task feature."],
    }


def test_writer_and_reviewer_reject_generated_task_interfaces_in_agent_repo():
    from agentreforge.improve.reviewer import REVIEWER_PROMPT
    from agentreforge.improve.writer_reviewer import WRITER_PROMPT_SUFFIX

    assert "repository that owned the failing artifact" in WRITER_PROMPT_SUFFIX
    assert "generated task's concrete CLI flag" in REVIEWER_PROMPT
    assert "substring/keyword" in WRITER_PROMPT_SUFFIX
    assert "Audit every changed path and behavioral hunk" in REVIEWER_PROMPT


def test_task_diff_scope_uses_actual_changed_paths_not_writer_claims():
    brief = "Affected components: demo_agent/agent.py, tests/"

    scope = _declared_affected_components(brief)

    assert scope == ["demo_agent/agent.py", "tests/"]
    assert _outside_declared_scope(
        ["demo_agent/agent.py", "tests/test_agent.py", "README.md"],
        scope,
    ) == ["README.md"]


def test_suggested_scope_does_not_create_a_structural_finding(tmp_path):
    class Worktree:
        path = tmp_path

        async def changed_since(self, _ref):
            return ["README.md"]

        async def changed_paths(self):
            return ["README.md"]

    executor = TaskExecutor(client=object(), registry=object(), reviewer=object())
    findings = asyncio.run(
        executor._structural_findings(
            Worktree(),
            since="task-start",
            declared_scope=["agent.py"],
        )
    )

    assert findings == []


def test_task_executor_keeps_blocker_builder_after_scope_helpers():
    assert callable(TaskExecutor._build_blocker)


def test_reviewer_findings_remain_separate_and_keep_ids():
    executor = TaskExecutor(client=object(), registry=object(), reviewer=object())
    review = Review(
        approved=False,
        feedback="Two concrete issues remain.",
        structured_findings=[
            {
                "id": "F1",
                "severity": "major",
                "location": "agent.py:10",
                "description": "State is inferred from prose.",
                "evidence": "The branch uses substring matching.",
                "required_fix": "Use a typed status.",
            },
            {
                "id": "F2",
                "severity": "major",
                "location": "agent.py:30",
                "description": "Failure does not trigger repair.",
                "evidence": "The loop continues without a new instruction.",
                "required_fix": "Feed the failure into the next action.",
            },
        ],
    )

    result = executor._to_review(review, location="task")

    assert [finding.id for finding in result.findings] == ["F1", "F2"]
    assert result.findings[0].evidence == "The branch uses substring matching."


def test_reviewer_handoff_is_typed_and_internally_consistent():
    valid = """{
      "verdict": "approve",
      "blocking_findings": [],
      "non_blocking_findings": [],
      "summary": "works"
    }"""
    contradictory = """{
      "verdict": "approve",
      "blocking_findings": [{
        "severity": "major", "location": "agent.py",
        "description": "broken", "required_fix": "fix it"
      }],
      "non_blocking_findings": [],
      "summary": "broken"
    }"""

    assert _review_output_error(valid) == ""
    assert "contradicts" in _review_output_error(contradictory)


def test_reviewer_bash_runs_in_copy_and_invalidates_source_mutation(tmp_path):
    source = tmp_path / "candidate"
    source.mkdir()
    product = source / "agent.py"
    product.write_text("VALUE = 1\n", encoding="utf-8")

    async def mutating_bash(args, context):
        Path(context.cwd, "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
        return ToolResult("changed copy")

    full = ToolRegistry()
    full.register(
        Tool(
            name="bash",
            description="fake shell",
            parameters=object_schema({"command": {"type": "string"}}, ["command"]),
            handler=mutating_bash,
            is_read_only=False,
        )
    )
    scoped = review_registry(full, source_root=str(source))
    bash = scoped.get("bash")
    assert bash is not None

    result = asyncio.run(
        bash.handler(
            {"command": "modify candidate"},
            ToolContext(cwd=str(source)),
        )
    )

    assert result.is_error
    assert "runtime result is invalid" in result.content
    assert product.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_writer_note_has_no_schema_and_reviewer_uses_authoritative_diff():
    valid_review = """{
      "verdict": "approve",
      "blocking_findings": [],
      "non_blocking_findings": [],
      "summary": "implementation is sound"
    }"""

    class Client:
        def __init__(self):
            self.calls = 0
            self.last_message = ""

        async def chat(self, messages, tools=None, *, system_prompt):
            self.calls += 1
            self.last_message = messages[-1].content
            yield {"type": "text_delta", "text": valid_review}

    client = Client()
    reviewer = AgenticReviewer(
        client=client,
        registry=ToolRegistry(),
        cwd=".",
    )
    task = "Shared Task Contract [implement]\nRequired review clause ids: "

    result = asyncio.run(
        reviewer.review(
            task,
            "Optional Writer note (not evidence):\nnot JSON at all\n\n"
            "Authoritative Git diff (this task only):\n"
            "diff --git a/agent.py b/agent.py",
        )
    )

    assert result.approved
    assert client.calls == 1
    assert "not JSON at all" in client.last_message
    assert "diff --git a/agent.py b/agent.py" in client.last_message
