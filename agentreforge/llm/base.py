"""
LLM Client: An interface for the client
A client must provide:
    model_name,
    provider_name,
    def chat(messages: list[Message], tools: list[dict[str, Any]] | None=None, system_prompt: str,)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from ..types import Message


class LlmClient(Protocol):
    model_name: str
    provider_name: str

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        system_prompt: str,
    ) -> AsyncIterator[dict[str, Any]]: ...
