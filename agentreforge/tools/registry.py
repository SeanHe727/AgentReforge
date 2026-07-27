from __future__ import annotations

from ..tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        # name -> Tool, so we can both list all tools and look one up by name.
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        # Used on the "return trip" to find the tool the model asked to call.
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict]:
        # Used on the "outbound trip": the schemas we hand to the LLM.
        return [self._tools[name].definition() for name in self.list_names()]
