from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class MemoryEntry:
    id: int
    scope: str
    content: str
    created_at: str


class MemoryManager:
    def __init__(self, db_path: str | Path, scope: str):
        self.db_path = Path(db_path).expanduser()
        self.scope = scope
        # first-run setup: ensure the folder exists, then create the table.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # open (or create) the SQLite file and return a connection.
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        # make sure the table (and index) exist; no-op after the first run.
        with self._connect() as conn:
            conn.execute(
                "create table if not exists memories ("
                "id integer primary key autoincrement,"
                "scope text not null,"
                "content text not null,"
                "created_at text not null)"
            )
            conn.execute("create index if not exists idx_memories_scope on memories(scope, id)")

    def save(self, content: str) -> int:
        # save one memory into the database, tagged with the current scope.
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "insert into memories(scope, content, created_at) values (?, ?, ?)",
                (self.scope, content.strip(), created_at),
            )
            return int(cursor.lastrowid)

    def list_memory(self, limit: int = 20) -> list[MemoryEntry]:
        # retrieve the most recent memories for the current scope.
        with self._connect() as conn:
            rows = conn.execute(
                "select id, scope, content, created_at from memories "
                "where scope = ? order by id desc limit ?",
                (self.scope, limit),
            ).fetchall()
        return [MemoryEntry(*row) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        # keyword match: keep memories whose content contains every query term.
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return self.list_memory(limit)
        matches = []
        for entry in self.list_memory(limit=200):
            if all(term in entry.content.lower() for term in terms):
                matches.append(entry)
            if len(matches) >= limit:
                break
        return matches

    def clear(self) -> int:
        # delete all memories for the current scope; returns how many were removed.
        with self._connect() as conn:
            cursor = conn.execute("delete from memories where scope = ?", (self.scope,))
            return int(cursor.rowcount)
