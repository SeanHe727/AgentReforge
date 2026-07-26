"""Calibrate problem difficulty: run the BASELINE demo coder on every problem and
grade it. High baseline scores => problems too easy (no headroom); make them harder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .harness import collect_solution, grade_solution, solve_problem
from .problems import problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="path to the baseline coder repo")
    ap.add_argument("--work", required=True, help="dir to write solutions into")
    ap.add_argument("--max-steps", type=int, default=15)
    args = ap.parse_args()

    rows = []
    for prob in problems():
        work = str(Path(args.work) / prob["id"])
        print(f"\n>>> solving {prob['id']} ...", flush=True)
        summary = solve_problem(args.agent, prob, work, max_steps=args.max_steps)
        solution = collect_solution(work)
        score, notes = grade_solution(prob, solution)
        rows.append((prob["id"], score))
        print(f"    score={score}  | agent: {summary[:80]}")
        print(f"    grader: {notes.splitlines()[0] if notes else ''}")

    print("\n===== BASELINE calibration =====")
    for pid, score in rows:
        print(f"  {pid:<18} {score}/100")
    avg = sum(s for _, s in rows) / len(rows) if rows else 0
    print(f"  {'AVERAGE':<18} {avg:.0f}/100")
    print("\nHigh average (>~70) => too easy, make harder. Low/mid => good headroom.")


if __name__ == "__main__":
    main()
