"""Hard real-engineering problems demo_agent must solve — the test dataset.

Each is a demanding multi-component system with concurrency, persistence, or state
machines and many correctness traps. A single bare pass typically ships something
incomplete or subtly wrong; a well-improved agent (plans, self-tests, integrates)
does better — that gap is what the judge reads. `rubric` states the HARD bar so
partial solutions score low. These specs also feed Codex when it authors brownfield
starter repos + hidden tests (see CODEX_PROMPTS.md).
"""

from __future__ import annotations

PROBLEMS: list[dict] = [
    {
        "id": "web_framework",
        "prompt": (
            "Build a WSGI web framework in `webapp/`. Requirements:\n"
            "- Routing with TYPED path converters: `/users/<int:id>`, `/f/<path:rest>`; "
            "reject wrong types with 404.\n"
            "- Blueprints/route groups with a URL prefix, mountable on the app.\n"
            "- An ONION middleware chain where a middleware may SHORT-CIRCUIT (e.g. auth "
            "returns 401 before the handler runs).\n"
            "- Request (query/headers/body) and Response (status/headers/body) objects; "
            "a real `app(environ, start_response)` WSGI callable.\n"
            "- 404 + 405 (method not allowed) + 500 handling.\n"
            "- A `TestClient` for in-process requests, and a `demo.py` exercising typed "
            "routes, a blueprint, auth short-circuit, and errors."
        ),
        "rubric": "Typed converters w/ 404 on mismatch; blueprints w/ prefix; onion "
                  "middleware with real short-circuit; WSGI callable + TestClient; 404/405/500 "
                  "distinguished; demo covers all. Missing WSGI, converters, or short-circuit "
                  "=> low.",
    },
    {
        "id": "task_queue",
        "prompt": (
            "Build a concurrent task queue in `taskq/`. Requirements:\n"
            "- MULTIPLE worker threads consuming jobs concurrently, thread-safe.\n"
            "- PRIORITY and DELAYED (run-at) scheduling.\n"
            "- Retry with exponential backoff + max attempts, then DEAD-LETTER.\n"
            "- At-least-once with IDEMPOTENCY keys so a redelivered job isn't double-applied.\n"
            "- ATOMIC JSON persistence; on restart, RECOVER in-flight/pending jobs without "
            "loss or double-run beyond the retry policy.\n"
            "- Graceful shutdown that drains/persists.\n"
            "- A `demo.py` showing concurrency, priority, retry->dead-letter, and a "
            "simulated crash + recovery."
        ),
        "rubric": "Thread-safe multi-worker; priority + delay; backoff+cap+dead-letter; "
                  "idempotency prevents double-apply; atomic persistence + correct crash "
                  "recovery; graceful drain. Missing concurrency safety, idempotency, or "
                  "recovery correctness => low.",
    },
    {
        "id": "config_system",
        "prompt": (
            "Build a config system in `config/`. Requirements:\n"
            "- Precedence merge: defaults < JSON file < env vars, with DEEP merge of nested "
            "maps and defined LIST merge semantics.\n"
            "- `${VAR}` interpolation referencing other keys/env, with cycle detection.\n"
            "- A typed SCHEMA (required, types, nested, defaults) with validation errors "
            "that name the exact dotted path.\n"
            "- Dotted access `cfg.get('db.host')` and typed getters.\n"
            "- HOT RELOAD: re-read on change and fire registered change callbacks with a diff.\n"
            "- A `demo.py` showing overrides, interpolation, a validation error, and reload."
        ),
        "rubric": "Correct precedence + DEEP merge; ${VAR} interpolation w/ cycle detection; "
                  "typed schema w/ path-accurate errors; dotted access; hot reload + change "
                  "callbacks. Shallow merge, no interpolation, or no reload => low.",
    },
    {
        "id": "event_bus",
        "prompt": (
            "Build an async event bus in `eventbus/`. Requirements:\n"
            "- `subscribe(topic, handler)` supporting SYNC and ASYNC (asyncio) handlers; "
            "`publish` awaits async ones.\n"
            "- WILDCARD topics (`order.*`, `order.#` multi-level) with defined precedence.\n"
            "- Delivery modes: CONCURRENT vs ORDERED; a BOUNDED queue with backpressure.\n"
            "- Error ISOLATION; a handler that keeps failing is routed to a DEAD-LETTER sink.\n"
            "- `once` subscriptions and `unsubscribe`; thread/async safety.\n"
            "- A `demo.py` covering async handlers, multi-level wildcards, a failing handler, "
            "and backpressure."
        ),
        "rubric": "Sync+async handlers awaited; single & multi-level wildcards w/ precedence; "
                  "concurrent/ordered modes + bounded backpressure; error isolation + dead-letter; "
                  "once/unsubscribe. Missing async, multi-level wildcard, or backpressure => low.",
    },
    {
        "id": "resilient_client",
        "prompt": (
            "Build a resilient HTTP client in `httpclient/` (mock transport, NO network). "
            "Requirements:\n"
            "- Retry on 5xx/timeouts with exponential backoff + JITTER + a retry BUDGET.\n"
            "- A full CIRCUIT BREAKER: closed -> open (after N failures) -> half-open (after "
            "cooldown) -> closed (after a success threshold).\n"
            "- Per-host BULKHEAD concurrency limit.\n"
            "- Response CACHE with TTL for idempotent GETs.\n"
            "- Per-request TIMEOUT; INJECTABLE clock + transport for deterministic tests.\n"
            "- A `demo.py` driving retry, circuit open/half-open/recovery, bulkhead, and cache."
        ),
        "rubric": "Backoff+jitter+budget; full breaker state machine w/ half-open threshold; "
                  "per-host bulkhead; TTL cache; timeout; injectable clock+transport. A partial "
                  "breaker (no half-open) or real time (untestable) => low.",
    },
    {
        "id": "plugin_system",
        "prompt": (
            "Build a plugin system in `plugins/`. Requirements:\n"
            "- A `Plugin` interface (name, version, requires: list of name+version constraint, "
            "setup(ctx), teardown()).\n"
            "- DISCOVERY of registered plugins + interface validation.\n"
            "- Resolve dependencies with SEMVER constraints (e.g. `db>=1.2`), TOPO-ordered "
            "load, error on cycle or unsatisfiable constraint; support OPTIONAL deps.\n"
            "- Lifecycle: setup in order, teardown in reverse; if one setup fails, ROLL BACK "
            "(teardown) the already-setup plugins and report which failed.\n"
            "- Shared context injection into setup().\n"
            "- A `demo.py` with a dependency chain, an optional dep, a version conflict, and "
            "a plugin whose setup fails (triggering rollback)."
        ),
        "rubric": "Interface validation; semver constraint resolution + topo load + cycle/"
                  "unsatisfiable errors; optional deps; setup/teardown order; rollback on mid-"
                  "chain failure; context injection. Missing semver, rollback, or topo => low.",
    },
]


def problems() -> list[dict]:
    """All coding problems in the dataset."""
    return list(PROBLEMS)
