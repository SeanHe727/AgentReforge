"""Lightweight retrieval over historical AgentReforge loop records.

Current-run facts are injected directly by ``OrchestratorContextBuilder``.
This index is intentionally only for finding potentially relevant experience
from older recursive runs. SQLite FTS keeps the first version local,
deterministic, inspectable, and dependency-free; semantic embeddings can be
added later without changing the record source of truth.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .records import ReforgeLoopRecord


class ImprovementHistoryIndex:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "create virtual table if not exists improvement_history using fts5("
                "text, target_repo UNINDEXED, run_id UNINDEXED, loop_id UNINDEXED, "
                "component UNINDEXED, record_path UNINDEXED)"
            )

    def rebuild(self, repo_root: str | Path) -> int:
        """Re-index durable loop records. The JSON files remain authoritative."""

        root = Path(repo_root).resolve()
        records_root = root / ".meta-improve" / "records"
        with self._connect() as conn:
            conn.execute(
                "delete from improvement_history where target_repo = ?",
                (str(root),),
            )
        count = 0
        for path in sorted(records_root.glob("*/loops/loop_*/record.json")):
            try:
                record = ReforgeLoopRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            count += self.index_loop(record, root, record_path=str(path))
        return count

    def index_loop(
        self,
        record: ReforgeLoopRecord,
        repo_root: str | Path,
        *,
        record_path: str = "",
    ) -> int:
        root = str(Path(repo_root).resolve())
        documents = [
            (
                "loop",
                " ".join(
                    [
                        record.proposal_summary,
                        json.dumps(record.diagnosis, ensure_ascii=False),
                        " ".join(record.changed_paths),
                        " ".join(record.remaining_gaps),
                        record.stage,
                        record.error,
                    ]
                ),
            )
        ]
        documents.extend(
            (
                component.component,
                " ".join(
                    [
                        component.summary,
                        component.status,
                        json.dumps(component.details, ensure_ascii=False),
                    ]
                ),
            )
            for component in record.components
        )
        with self._connect() as conn:
            conn.execute(
                "delete from improvement_history where target_repo = ? and loop_id = ?",
                (root, record.loop_id),
            )
            conn.executemany(
                "insert into improvement_history("
                "text, target_repo, run_id, loop_id, component, record_path"
                ") values (?, ?, ?, ?, ?, ?)",
                [
                    (
                        text,
                        root,
                        record.run_id,
                        record.loop_id,
                        component,
                        record_path,
                    )
                    for component, text in documents
                    if text.strip()
                ],
            )
        return len(documents)

    def search(
        self,
        query: str,
        *,
        target_repo: str | Path,
        exclude_run_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        terms = [term for term in query.replace('"', " ").split() if term]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        sql = (
            "select run_id, loop_id, component, record_path, "
            "snippet(improvement_history, 0, '[', ']', ' … ', 24), bm25(improvement_history) "
            "from improvement_history where improvement_history match ? "
            "and target_repo = ?"
        )
        params: list[Any] = [expression, str(Path(target_repo).resolve())]
        if exclude_run_id:
            sql += " and run_id != ?"
            params.append(exclude_run_id)
        sql += " order by bm25(improvement_history) limit ?"
        params.append(max(1, min(limit, 20)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "run_id": run_id,
                "loop_id": loop_id,
                "component": component,
                "record_path": record_path,
                "snippet": snippet,
                "score": score,
            }
            for run_id, loop_id, component, record_path, snippet, score in rows
        ]
