from __future__ import annotations

from datetime import datetime
from pathlib import Path


class PromptAssembler:
    def __init__(
        self,
        *,
        cwd: str,
        model: str,
        provider: str,
        tool_names: list[str] | None = None,
        memories: list[str] | None = None,
    ):
        # params initialize. Resolve cwd to an absolute path so the model sees an
        # unambiguous working directory.
        self.cwd = str(Path(cwd).resolve())
        self.model = model
        self.provider = provider
        self.tool_names = tool_names or []
        self.memories = memories or []

    def _project_memory(self) -> str:
        # read project instruction files so the agent gets project-specific
        # context. Always injected (unlike skills, which are lazy-loaded on demand).
        chunks = []
        for name in ("AGENTREFORGE.md", ".agentreforge/AGENTREFORGE.md"):
            path = Path(self.cwd) / name
            if path.exists():
                try:
                    chunks.append(path.read_text(encoding="utf-8")[:4000])
                except OSError:
                    continue
        return "\n\n".join(chunks)[:8000]

    def build(self) -> str:
        # basic informations for agent, joined into one system prompt string.
        parts = [
            # 1. identity & purpose
            "You are agent-reforge, a powerful AI coding assistant running in a terminal.",
            # 2. environment awareness
            f"Current time: {datetime.now().isoformat(timespec='seconds')}",
            f"Working directory: {self.cwd}",
            f"Model: {self.model} ({self.provider})",
            # 3. capabilities
            f"Available tools: {', '.join(self.tool_names) if self.tool_names else 'none yet'}",
            "",
            # 4. guidelines and requirements
            "Guidelines:",
            "- Be concise, direct, and implementation-oriented.",
            "- Prefer deterministic local tools before guessing.",
            "- Ask a clarifying question only when proceeding would be risky.",
        ]
        # 5. long-term memory: inject recalled facts so the agent "remembers".
        if self.memories:
            parts.append("")
            parts.append("Known facts from long-term memory:")
            parts.extend(f"- {fact}" for fact in self.memories)
        # 6. project instructions: always-injected project context.
        project_memory = self._project_memory()
        if project_memory:
            parts.append("")
            parts.append("Project instructions (AGENTREFORGE.md):")
            parts.append(project_memory)
        return "\n".join(parts)
