"""Parse an LLM's text output into a validated Pydantic model.

Shared by the Planner (pure-JSON output) and the Orchestrator (which may emit
prose then a ```json block after investigating). Extracts the JSON, then lets
Pydantic validate it — turning unreliable model text into a trusted object.
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> str:
    # prefer the LAST fenced ```json ... ``` block: a verbose model may emit
    # intermediate blocks while investigating, and the final answer comes last.
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        return fenced[-1].strip()
    # else take the outermost { ... } object.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def parse_json_model(text: str, model_cls: type[T]) -> T:
    try:
        return model_cls.model_validate_json(_extract_json(text or ""))
    except ValidationError as exc:
        raise ValueError(f"LLM produced invalid JSON for {model_cls.__name__}: {exc}") from exc
