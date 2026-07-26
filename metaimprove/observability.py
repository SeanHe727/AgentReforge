"""Optional LangSmith tracing.

Re-exports `traceable` from langsmith so the pipeline stages can be decorated for
observability. If langsmith isn't installed, `traceable` degrades to a no-op
decorator, so the code never hard-depends on it. Even when installed, langsmith's
`@traceable` only records anything when `LANGSMITH_TRACING=true` (with an API key)
is set — otherwise it just calls the function normally, zero overhead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    from langsmith import traceable as _traceable

    traceable = _traceable
except Exception:  # noqa: BLE001 - langsmith is optional
    def traceable(*args: Any, **kwargs: Any) -> Any:
        # support both @traceable and @traceable(name=...) forms.
        if args and callable(args[0]):
            return args[0]

        def deco(fn: Callable) -> Callable:
            return fn

        return deco
