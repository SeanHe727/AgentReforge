"""
1. Build request(url + header + payload)
2. Read the raw response
3. Convert the stream response of model into structured event:
        raw stream response -_iter_sse()-> JSON -json.loads()-> dict(Chunk) -accumulate-> event
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ..types import Message


@dataclass
class OpenAICompatibleClient:
    provider_name: str
    model: str
    api_key: str
    base_url: str
    # Reasoning models count hidden reasoning and visible JSON against the same
    # completion budget. 8k can truncate otherwise compact typed handoffs before
    # their closing delimiter; this is a ceiling, not a pre-allocated charge.
    max_tokens: int = 16384
    temperature: float = 0.7
    timeout: float = 120.0

    @property
    def model_name(self) -> str:
        return self.model

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        system_prompt: str,
    ) -> AsyncIterator[dict[str, Any]]:
        # 1. check if the api_key is empty -> surface as an error event and stop.
        if not self.api_key:
            yield {"type": "error", "error": RuntimeError("API key is not configured.")}
            return

        # 2. build the request: URL + headers + payload.
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages, system_prompt),
            "stream": True,
            # newer models (gpt-5.x) reject `max_tokens`; `max_completion_tokens`
            # is the current name and is accepted by gpt-4o too, so use it for all.
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"  # let model decide to call tools or not

        # 3. sent request, then parse the response
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout) as client,
                client.stream("POST", url, headers=headers, json=payload) as response,
            ):
                response.raise_for_status()
                async for data in _iter_sse(response):
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # 4. parse chunks into events, then yield events.
                    async for event in self._parse_chunk(chunk):
                        yield event
        except httpx.HTTPError as exc:
            yield {"type": "error", "error": exc}

    async def _parse_chunk(self, chunk: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """
        Convert the chunks into events
        JSON -> Dict -> event
        """
        # convert the chunk (OpenAI format) into our events. event: {type, ...payload}
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}

            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                yield {"type": "thinking_delta", "thinking": reasoning}

            content = delta.get("content")
            if isinstance(content, str) and content:
                yield {"type": "text_delta", "text": content}

            # Tool calls also stream in fragments; emit each raw delta. The ReAct
            # loop merges them by index into complete calls (that's its job).
            for tool_call in delta.get("tool_calls") or []:
                yield {"type": "tool_call_delta", "tool_call": tool_call}

            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                yield {
                    "type": "message_end",
                    "stop_reason": _map_finish_reason(str(finish_reason)),
                }

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            yield {
                "type": "usage",
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                },
            }

    def _format_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        # transform the system prompt + messages into the OpenAI wire format:
        # system goes first, then each message. Tool calls / tool results need
        # extra fields beyond {role, content}.
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            if message.role == "tool":
                # a tool result must say which call it answers, via tool_call_id.
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": message.content,
                    }
                )
            elif message.role == "assistant" and message.tool_calls:
                # an assistant turn that proposed tool calls carries them along.
                formatted.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": message.tool_calls,
                    }
                )
            else:
                formatted.append({"role": message.role, "content": message.content})
        return formatted


def _map_finish_reason(reason: str) -> str:
    if reason in {"tool_calls", "tool_use"}:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return "end_turn"


async def _iter_sse(response: httpx.Response) -> AsyncIterator[str]:
    """
    convert the stream response into a structured JSON
    """
    buffer = ""
    async for text in response.aiter_text():
        buffer += text
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            data = _extract_data(event)
            if data:
                yield data
    # Flush any trailing event that had no closing blank line.
    data = _extract_data(buffer)
    if data:
        yield data


def _extract_data(event: str) -> str:
    """Extract the payload: join the content of all `data:` lines in one event."""
    data_lines = []
    for line in event.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    return "\n".join(data_lines)
