from __future__ import annotations

import asyncio

from conftest import make_proposal

from agentreforge.improve.reviewer import (
    AgenticReviewer,
    _review_output_error,
)
from agentreforge.improve.writer_reviewer import (
    _task_brief,
    _task_contract_text,
)
from agentreforge.orchestration.task_executor import (
    TaskExecutor,
    _declared_affected_components,
    _outside_declared_scope,
)
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
    assert contract in reviewer_brief
    assert "Executable acceptance checks:" in reviewer_brief


def test_writer_prompt_treats_affected_components_as_suggestions():
    from agentreforge.improve.writer_reviewer import WRITER_PROMPT_SUFFIX

    assert "Affected components are suggestions" in WRITER_PROMPT_SUFFIX


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
