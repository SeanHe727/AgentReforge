# v0.1 mini benchmark result

## Frozen comparison

- Baseline coder: `b5e99c7281415c6e00401cd2df6bf76a1472f385`
- AgentReforge candidate: `c52a0480177297ff1c2b945e2e04fd8257ec2078`
- Candidate branch: `improve/20260727_150942_312321`
- Model: `gpt-5.4-mini`
- Budget: 12 agent steps
- Dataset: four deterministic brownfield cases
- Repeats: two independent paired runs

| Arm | Hidden-test passes | Pass rate | Mean wall time |
|---|---:|---:|---:|
| Baseline | 8 / 8 | 100% | 6.04 s |
| AgentReforge candidate | 8 / 8 | 100% | 8.83 s |

The candidate is a clean non-regression on this mini suite, but the suite is too
easy for this model and therefore does **not** establish a capability improvement.
The extra time is consistent with the candidate's explicit inspect/search/verify
instruction and should be treated as an efficiency cost, not hidden.

The generated candidate itself passed deterministic delivery, independent Reviewer
checks, and manual confinement checks for relative traversal, absolute paths, and
the read/write/list/search tool surface.

## Calibration note

A tools-only candidate (`362d95f33cfa39ddd81c7bab5502d55c447acac1`)
scored 2/4 in one calibration while its paired baseline scored 4/4. Inspection
showed two real missed edge cases, not harness failures. That negative result is
why the frozen v0.1 candidate combines safe navigation tools with explicit
inspect-before-edit and verify-before-finish guidance.

The raw local evidence is stored under:

- `.agentreforge/benchmarks/v01-combined-calibration/`
- `.agentreforge/benchmarks/v01-combined-repeat2/`
- `.agentreforge/benchmarks/v01-calibration/` (tools-only negative calibration)

These generated solution directories are intentionally not committed.
