"""Run a demo_agent version on every problem and leave its solutions on disk.

No grading here — the produced code is meant to be handed to a judge (Codex). A
"version" is just a path to the demo repo, so this runs the baseline or any improved
worktree. Each solution lands in <out>/<problem_id>/ next to a TASK.md copy so the
judge sees the task + rubric beside the code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .harness import solve_problem
from .problems import problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="path to the demo-agent repo to run")
    ap.add_argument("--out", required=True, help="output dir; solutions go to <out>/<id>/")
    ap.add_argument("--max-steps", type=int, default=15)
    args = ap.parse_args()

    for prob in problems():
        d = Path(args.out) / prob["id"]
        print(f">>> solving {prob['id']} ...", flush=True)
        summary = solve_problem(args.agent, prob, str(d), max_steps=args.max_steps)
        # drop the task + rubric beside the code so the judge has full context.
        (d / "TASK.md").write_text(f"{prob['prompt']}\n\n---\nRubric:\n{prob['rubric']}\n")
        print(f"    {summary[:100]}")

    print(f"\nsolutions in: {args.out}")


if __name__ == "__main__":
    main()
