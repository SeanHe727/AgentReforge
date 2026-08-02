# AgentReforge — demo-agent improvement run (results for Codex)

Context for a judge. We are measuring whether **AgentReforge** (our self-improvement
pipeline) can improve a demo coding agent's ability. This file summarizes one run and
points to the code to judge.

## The setup

- **Demo coding agent** (`AgentReforge-demo/demo_agent/`): a real ReAct coding agent (gpt-5.4-mini +
  write/read/run_bash tools). Deliberately weak baseline: bare prompt, no planning, no
  self-verification, short step budget. This is the TARGET; AgentReforge edits its code.
- **Test dataset** (`eval/problems.py`): 6 hard multi-file engineering problems
  (web_framework, task_queue, config_system, event_bus, resilient_client, plugin_system),
  each solved **from scratch (greenfield)**.
- **Model held constant** at gpt-5.4-mini for both the coder and the improver, so score
  deltas reflect the SCAFFOLD, not the model. Same step budget (max_steps=15) both sides.

## Baseline (before improvement)

Codex static review, avg **~56/100**. Key weaknesses:

- **Very weak self-verification**: agent finishes without running/importing its code.
  `task_queue` was emitted as a single non-importable line → SyntaxError (8/100).
- **No fix-on-failure**: `plugin_system` mis-reported which plugin failed (never checked
  results).
- **Surface-level semantics**: e.g. `event_bus` had CONCURRENT/ORDERED enums but not two
  real delivery semantics.

Baseline solutions: `eval/solutions/baseline/<problem_id>/`.

## What AgentReforge did

`agent-reforge improve --cwd /path/to/AgentReforge-demo -m gpt-5.4-mini --mode autonomous --level deep`.

Across several runs the pipeline's Reviewer repeatedly CAUGHT real, subtle bugs in the
Writer's attempts — a `verified` flag flipped by any read/bash call; brittle keyword
heuristics for picking a verification command; a deadlock in a verification phase machine.
These are exactly the semantic flaws a one-shot improver would ship. (During this we also
improved AgentReforge itself: the Writer now pushes each increment to the Reviewer via a
`request_review` tool; more Writer<->Reviewer rounds; each round edits the prior attempt;
and the Reviewer now judges by IMPACT — rejecting real defects, approving when only minor
nits remain.)

The delivered improvement is **prompt-only**: `demo_agent/agent.py`'s system prompt now requires
a brief plan then explicit self-verification before finishing; the `run_task` loop itself is
unchanged (no hard code enforcement — a soft, prompt-level guarantee). Separately, a genuine
tool bug was fixed in baseline: `run_bash` now always reports `(exit N)` so a failing check
can't be mistaken for success. Improved demo agent: branch `improve/20260726_150948_996321`.

## After improvement

Ran the improved coder on the same 6 problems, same max_steps=15. Solutions:
`eval/solutions/ours/<problem_id>/`.

- **Targeted win: `task_queue`** — baseline = non-importable SyntaxError blob; ours = parses
  and runs.
- **5 of 6 completed** (only config_system hit max_steps).
- **Regression: `config_system`** — hit max_steps and left 2 half-written files with
  SyntaxErrors (baseline had this one working). This is the step-budget issue: the self-verify
  nudge costs a few extra turns, so the biggest task runs out at 15 steps.
- Open question: re-run BOTH baseline+ours at a higher equal budget (e.g. max_steps=25) for a
  fair completeness comparison before judging.

## What to judge

For each `<problem_id>`, compare **`eval/solutions/baseline/<id>/`** vs
**`eval/solutions/ours/<id>/`** (each has a `TASK.md` with the task + rubric). Decide which
solution is better on correctness, completeness, edge cases, design, robustness — and note
whether ours' self-verification produced more runnable/correct code.

## Not yet done

The fair **"other" improver** (an equal-budget, non-workflow baseline improver, i.e. NOT
one-shot) is not built yet — needed for the real our-vs-other comparison. This run only
compares baseline vs AgentReforge-improved.
