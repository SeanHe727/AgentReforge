from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.openai_compatible import OpenAICompatibleClient
from ..memory.manager import MemoryManager
from ..rag.code_index import CodeIndex
from ..tools.builtins import get_builtin_tools
from ..tools.registry import ToolRegistry

_MEMORY_DB = Path.home() / ".agentreforge" / "memory.db"
_CODE_INDEX_DB = Path.home() / ".agentreforge" / "code_index.db"


@dataclass
class Thread:
    id: str
    events: list[dict[str, Any]] = field(default_factory=list)


class RuntimeState:
    def __init__(self, config, cwd: str, api_key: str):
        self.config = config
        self.cwd = cwd
        self.api_key = api_key
        # build agent dependencies once, reused for every turn.
        self.client = OpenAICompatibleClient(
            provider_name=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
        # set up and register all the tools
        self.registry = ToolRegistry()
        self.registry.register_all(get_builtin_tools())
        # memory + code index (rag)
        self.memory = MemoryManager(_MEMORY_DB, scope=cwd)
        self.code_index = CodeIndex(root=cwd, db_path=_CODE_INDEX_DB)
        self.code_index.rebuild()  # refresh the index at startup
        # in-memory thread store: thread_id -> Thread
        self.threads: dict[str, Thread] = {}

    def create_thread(self) -> Thread:
        # make a new thread with a unique id, store it, return it.
        thread = Thread(id=f"thread_{uuid.uuid4().hex[:12]}")
        self.threads[thread.id] = thread
        return thread

    def get_thread(self, thread_id: str) -> Thread | None:
        return self.threads.get(thread_id)
