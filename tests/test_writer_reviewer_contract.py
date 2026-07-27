from __future__ import annotations

from conftest import make_proposal

from agentreforge.improve.reviewer import _parse, _validate_writer_report
from agentreforge.improve.writer_reviewer import (
    _parse_writer_report,
    _task_brief,
    _task_contract_text,
)
from agentreforge.orchestration.task_executor import (
    TaskExecutor,
    _declared_affected_components,
    _outside_declared_scope,
)


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


def test_writer_prompt_forbids_inventing_paths_outside_task_scope():
    from agentreforge.improve.writer_reviewer import WRITER_PROMPT_SUFFIX

    assert "Modify only the exact files listed under Affected components" in (
        WRITER_PROMPT_SUFFIX
    )


def test_task_diff_scope_uses_actual_changed_paths_not_writer_claims():
    brief = "Affected components: demo_agent/agent.py, tests/"

    scope = _declared_affected_components(brief)

    assert scope == ["demo_agent/agent.py", "tests/"]
    assert _outside_declared_scope(
        ["demo_agent/agent.py", "tests/test_agent.py", "README.md"],
        scope,
    ) == ["README.md"]


def test_task_executor_keeps_blocker_builder_after_scope_helpers():
    assert callable(TaskExecutor._build_blocker)


def test_reviewer_cannot_approve_without_covering_every_contract_clause():
    task = (
        "Shared Task Contract [implement]\n"
        "Required review clause ids: RB1, INV1\n"
    )

    incomplete = _parse("APPROVED\n[RB1] PASS — implemented", task)
    complete = _parse(
        "APPROVED\n[RB1] PASS — implemented\n[INV1] PASS — preserved",
        task,
    )

    assert not incomplete.approved
    assert "INV1" in incomplete.feedback
    assert complete.approved


def test_reviewer_all_clause_pass_checklist_is_an_unambiguous_approval():
    task = (
        "Shared Task Contract [implement]\n"
        "Required review clause ids: RB1, INV1\n"
    )

    complete_without_verdict = _parse(
        "[RB1] PASS — runtime evidence\n[INV1] PASS — code evidence",
        task,
    )
    checklist_with_failure = _parse(
        "[RB1] PASS — runtime evidence\n[INV1] FAIL — regression",
        task,
    )

    assert complete_without_verdict.approved
    assert not checklist_with_failure.approved


def test_writer_report_is_typed_and_must_cover_contract_clauses():
    task = (
        "Shared Task Contract [implement]\n"
        "Required review clause ids: RB1, INV1\n"
    )
    complete = """```json
{
  "task_id": "implement",
  "summary": "implemented the behavior",
  "changed_files": ["src/agent.py"],
  "clause_evidence": [
    {"clause_id": "RB1", "implementation": "added branch", "evidence": "src/agent.py:4"},
    {"clause_id": "INV1", "implementation": "kept fallback", "evidence": "src/agent.py:8"}
  ],
  "commands_run": [
    {"command": "python -m pytest", "exit_code": 0, "output_summary": "passed"}
  ],
  "known_limitations": [],
  "deviations": []
}
```"""
    missing = complete.replace(
        ',\n    {"clause_id": "INV1", "implementation": "kept fallback", '
        '"evidence": "src/agent.py:8"}',
        "",
    )

    report = _parse_writer_report(complete)

    assert report is not None
    assert report.task_id == "implement"
    assert _validate_writer_report(task, complete) == ""
    assert "INV1" in _validate_writer_report(task, missing)


def test_writer_report_rejects_wrong_task_and_unknown_clause():
    task = (
        "Shared Task Contract [implement]\n"
        "Required review clause ids: RB1\n"
    )
    wrong = """```json
{
  "task_id": "other",
  "summary": "changed it",
  "changed_files": [],
  "clause_evidence": [
    {"clause_id": "OTHER", "implementation": "x", "evidence": "y"}
  ],
  "commands_run": [],
  "known_limitations": [],
  "deviations": []
}
```"""

    assert "does not match" in _validate_writer_report(task, wrong)
