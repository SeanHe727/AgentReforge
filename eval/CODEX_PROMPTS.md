# Codex prompts for the demo-agent evaluation

Flow: the demo agents solve each problem **from scratch (greenfield)**; **Codex writes
the hidden tests from scratch**; then the agent's produced code is fed to Codex to run
those tests and judge. Three ready-to-paste prompts. **Prompt A** = Codex authors a
hidden pytest suite per problem. **Prompt B** = Codex grades one from-scratch solution
(absolute, for difficulty calibration). **Prompt C** = Codex compares two solutions
(pairwise, the real our-agent-vs-other comparison). The problem specs + rubrics live in
`eval/problems.py`.

---

## Prompt A — author the hidden test suites (from scratch)

> You are building the graders for a benchmark of hard software-engineering tasks. Coding
> agents will implement each task **from scratch** as a fresh package; your job is to write
> the **hidden pytest suite** that decides whether a from-scratch solution is correct. I
> will give you problem specs (each with an `id`, a `prompt` describing the package to
> build + its file/API names, and a `rubric`). For EACH problem, write to disk under
> `tests/<id>/` a thorough `test_*.py` suite that:
>
> 1. Imports the package/modules exactly as named in the spec (e.g. `webapp`, `taskq`) —
>    assume the agent's solution package is importable on `sys.path`.
> 2. Covers the HAPPY path AND every edge case / hard requirement in the rubric (typed
>    routing 404s, idempotency/no-double-apply, deep merge, circuit half-open, rollback,
>    `${VAR}` cycles, concurrency safety, etc.).
> 3. Is deterministic: inject clocks/transports where the spec allows; no real network,
>    no real sleeps for timing (drive time via the injected clock).
> 4. Fails clearly (good assert messages) so a partial solution's gaps are obvious.
>
> Aim for a suite a strong solution passes ~fully and a naive single-pass solution only
> partially passes. Output only test files. Here are the specs:
>
> ```
> <paste PROBLEMS from eval/problems.py here>
> ```

---

## Prompt B — grade ONE solution (absolute, for calibration)

> You are a strict senior engineer grading a coding agent's solution to a brownfield
> engineering task. Inputs below: the TASK, the RUBRIC, the agent's final CODEBASE, and
> the hidden TEST SUITE for this problem.
>
> 1. **Run the hidden tests** (`pytest`) against the solution and note the pass/fail count.
> 2. Score 0–100 weighting: hidden-test pass rate (heaviest) + completeness + edge-case
>    handling + design + NOT breaking existing behavior.
> 3. Penalize hard: failing tests, stubs, shortcuts that only appear to work, regressions.
>
> Reply with `SCORE: <0-100>` on the first line, then 3–5 lines citing tests passed/failed
> and rubric points met/missed. Be harsh — a typical single-pass solution should land 40–70.
>
> TASK:\n<...>\nRUBRIC:\n<...>\nHIDDEN TESTS:\n<...>\nSOLUTION:\n<...>

---

## Prompt C — compare TWO solutions (pairwise, our-agent vs other)

> You are judging which of two coding agents improved a demo coding agent better. Both
> started from the SAME baseline agent and were given the SAME goal, model, and
> time/cost budget (one used a structured workflow, one did not — you don't know which is
> which). Below are the TASK + RUBRIC and, from each improved agent, its solution to that
> task (and hidden-test results if provided).
>
> Decide which solution is better on: correctness (hidden tests), completeness, edge cases,
> design, and robustness. Ignore length and style unless they affect quality. Guard against
> position bias — judge on merit only.
>
> Reply with `WINNER: A` or `WINNER: B` or `WINNER: TIE` on the first line, then 3–5 lines
> of concrete justification citing specific rubric points and test outcomes.
>
> TASK:\n<...>\nRUBRIC:\n<...>\nSOLUTION A:\n<...>\nSOLUTION B:\n<...>

**Note:** for Prompt C, run each pair BOTH orderings (A/B then B/A swapped) and only count a
win if it's consistent — that cancels position bias.
