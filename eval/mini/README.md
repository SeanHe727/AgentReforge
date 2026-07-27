# AgentReforge mini benchmark

This is the fast, deterministic evaluation for the early AgentReforge workflow.
It complements the larger, LLM-judged greenfield suite in `eval/problems.py`.

The benchmark copies a small brownfield starter repository into an isolated
solution directory, asks each coder version to implement the same task with the
same model and step budget, and then executes a hidden `unittest` file that was
not present in the coder's working directory.

Current cases:

- `nested_config`: repository inspection and a compatible nested lookup change;
- `event_unsubscribe`: targeted bug repair with identity and ordering semantics;
- `retry_policy`: boundary conditions and backoff behavior;
- `unique_slugs`: unrelated feature work as a non-regression guardrail.

Example:

```bash
OPENAI_API_KEY=... CODER_MODEL=gpt-5.4-mini \
python3 -m eval.mini.run \
  --arm baseline=/path/to/baseline \
  --arm candidate=/path/to/candidate \
  --repeats 2 \
  --max-steps 12 \
  --out /tmp/agentreforge-mini
```

The primary metric is case-level hidden-test pass rate. `results.json` contains
per-run evidence and `summary.md` contains the paired comparison. Generated
solutions and result directories should not be committed.
