from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class McpServerSpec:
    name: str
    type: str = "stdio"  # "stdio" (spawn a subprocess) or "http"
    command: str | None = None  # stdio: the executable, e.g. "uvx" or "npx"
    args: list[str] = field(default_factory=list)  # stdio: its args
    url: str | None = None  # http: the server URL
    enabled: bool = True


def load_mcp_specs(project_root: str | Path) -> dict[str, McpServerSpec]:  # {name: McpServerSpec}
    # Read .agentreforge/mcp.json if present. Standard shape:
    # {"mcpServers": {"time": {"command": "uvx", "args": ["mcp-server-time"]}}}
    config_path = Path(project_root) / ".agentreforge" / "mcp.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    specs: dict[str, McpServerSpec] = {}
    for name, cfg in (data.get("mcpServers") or {}).items():
        specs[name] = McpServerSpec(
            name=name,
            type=str(cfg.get("type") or "stdio"),
            command=cfg.get("command"),
            args=list(cfg.get("args") or []),
            url=cfg.get("url"),
            enabled=bool(cfg.get("enabled", True)),
        )
    return specs
