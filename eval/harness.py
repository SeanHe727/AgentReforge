"""Run a demo_agent version on a problem, then LLM-grade its solution.

A "version" is just a path to the demo repo (baseline, or an improved worktree), so
the same code compares any version. The demo agent runs as a subprocess in that repo
with a fixed model, writing its solution into an isolated work dir; the grader is a
separate strong-model LLM call scoring the produced files against the rubric.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def solve_problem(agent_repo: str, problem: dict, work_dir: str, *, max_steps: int = 15) -> str:
    """Run demo_agent at agent_repo on the problem, writing code into work_dir."""
    # absolute: the agent subprocess runs with cwd=agent_repo, so a relative --dir
    # would land inside the agent repo, not here.
    work = str(Path(work_dir).resolve())
    Path(work).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "demo_agent", problem["prompt"], "--dir", work,
         "--max-steps", str(max_steps)],
        cwd=agent_repo, capture_output=True, text=True, timeout=600,
    )
    # the agent's summary (stdout); stderr surfaced only if it produced nothing.
    return proc.stdout.strip() or f"(agent error: {proc.stderr.strip()[:500]})"


def collect_solution(work_dir: str, *, cap: int = 12000) -> str:
    """Concatenate the produced source files into one labelled blob for the grader."""
    root = Path(work_dir)
    parts = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        parts.append(f"# ===== {rel} =====\n{p.read_text(errors='replace')}")
    blob = "\n\n".join(parts) or "(no .py files produced)"
    return blob[:cap] + ("\n...(truncated)" if len(blob) > cap else "")


GRADER_SYSTEM = (
    "You are a strict senior engineer grading a solution to a software-engineering task. "
    "Judge how well the code satisfies the task AND the rubric: correctness, completeness, "
    "edge cases, error handling, and design. Missing files, stubs, or shallow shortcuts "
    "score low. Reply with `SCORE: <0-100>` on the first line, then 2-4 lines citing which "
    "rubric points are met or missed."
)


def grade_solution(problem: dict, solution: str) -> tuple[int, str]:
    """LLM-grade one solution against its rubric; return (score 0-100, justification)."""
    message = (
        f"Task:\n{problem['prompt']}\n\n"
        f"Rubric (what a strong solution does):\n{problem['rubric']}\n\n"
        f"Candidate solution:\n{solution}"
    )
    text = _judge_chat(message)
    m = re.search(r"SCORE:\s*(\d{1,3})", text)
    score = max(0, min(100, int(m.group(1)))) if m else 0
    return score, text


def _judge_chat(message: str) -> str:
    """One call to the grader model (JUDGE_MODEL, default gpt-5.4-mini)."""
    payload = {
        "model": os.environ.get("JUDGE_MODEL", "gpt-5.4-mini"),
        "messages": [
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user", "content": message},
        ],
        "max_completion_tokens": 700,
    }
    req = urllib.request.Request(
        _BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]
