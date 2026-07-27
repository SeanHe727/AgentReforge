from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from ..tools.base import Tool, ToolContext, ToolResult, object_schema
from .config import McpServerSpec, load_mcp_specs


class McpClientManager:
    def __init__(self, project_root: str | Path):
        self.project_root = str(Path(project_root).resolve())
        self.specs = load_mcp_specs(self.project_root)
        self.last_errors: dict[str, str] = {}

    async def load_tools(self) -> list[Tool]:
        # discover + wrap tools from every enabled server; isolate broken ones.
        tools: list[Tool] = []
        self.last_errors.clear()
        for spec in self.specs.values():
            if not spec.enabled:
                continue
            try:
                tools.extend(await self._wrap_server_tools(spec))
            except Exception as exc:  # noqa: BLE001 - a broken server shouldn't kill startup
                self.last_errors[spec.name] = str(exc)
        return tools

    async def _wrap_server_tools(self, spec: McpServerSpec) -> list[Tool]:
        # 1. connect + DISCOVER the server's tools.
        async with self._session(spec) as session:
            result = await session.list_tools()
            remote_tools = list(result.tools)

        # 2. WRAP each remote tool as a local Tool whose handler forwards the call.
        wrapped: list[Tool] = []
        for remote in remote_tools:
            local_name = f"mcp__{spec.name}__{remote.name}"
            schema = remote.inputSchema or object_schema({})

            # default args capture THIS spec/name (avoid the late-binding closure bug).
            async def handler(
                args: dict[str, Any],
                context: ToolContext,
                *,
                server: McpServerSpec = spec,
                remote_name: str = remote.name,
            ) -> ToolResult:
                return await self.call_server_tool(server, remote_name, args)

            wrapped.append(
                Tool(
                    name=local_name,
                    description=remote.description or f"MCP tool {remote.name}",
                    parameters=schema,
                    handler=handler,
                    is_read_only=False,  # remote side effects unknown -> treat as unsafe
                )
            )
        return wrapped

    async def call_server_tool(
        self, spec: McpServerSpec, tool_name: str, args: dict[str, Any]
    ) -> ToolResult:
        # the MCP call_tool: run a tool on the remote server, get its result.
        async with self._session(spec) as session:
            result = await session.call_tool(tool_name, args)
        return ToolResult(content=_content_to_text(result.content), is_error=bool(result.isError))

    @asynccontextmanager
    async def _session(self, spec: McpServerSpec):
        # open an MCP session over the right transport, run the initialize
        # handshake, yield it, and tear everything down on exit.
        if spec.type in {"stdio", "local"}:
            if not spec.command:
                raise ValueError(f"MCP server {spec.name} is missing 'command'")
            params = StdioServerParameters(command=spec.command, args=spec.args, env={**os.environ})
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
        elif spec.type in {"http", "streamable_http", "streamable-http"}:
            if not spec.url:
                raise ValueError(f"MCP server {spec.name} is missing 'url'")
            async with (
                streamablehttp_client(spec.url) as (read, write, _session_id),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
        else:
            raise ValueError(f"unsupported MCP transport: {spec.type}")


def _content_to_text(content: Any) -> str:
    # MCP results carry structured content blocks; flatten them to text.
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_content_to_text(item) for item in content)))
    if hasattr(content, "text"):
        return str(content.text)
    return str(content)
