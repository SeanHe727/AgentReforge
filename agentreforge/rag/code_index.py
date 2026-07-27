from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Only index real text/code files, and skip noisy generated/vendor dirs.
TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".js",
    ".java",
    ".go",
    ".rs",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".html",
    ".css",
    ".sql",
    ".sh",
}
SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", "__pycache__", ".agentreforge"}


@dataclass
class CodeSearchResult:
    path: str
    line: int
    snippet: str


class CodeIndex:
    def __init__(self, root: str | Path, db_path: str | Path):
        self.root = Path(root).resolve()
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        # ensure table exist, if not create one
        with self._connect() as conn:
            conn.execute(
                "create table if not exists code_chunks ("
                "id integer primary key autoincrement,"
                "root text not null,"
                "path text not null,"
                "line integer not null,"
                "content text not null)"
            )
            conn.execute("create index if not exists idx_code_root_path on code_chunks(root, path)")

    def _iter_files(self) -> Iterator[Path]:
        # skip unimportant files: noisy dirs, and non-code file types.
        for path in self.root.rglob("*"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path

    def rebuild(self) -> int:
        with self._connect() as conn:
            # clear old index for this project
            conn.execute("delete from code_chunks where root = ?", (str(self.root),))
            count = 0
            for file_path in self._iter_files():
                rel = str(file_path.relative_to(self.root))
                try:
                    # split the file into lines
                    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                # scan the lines one by one, and index each non-blank line
                for line_number, line in enumerate(lines, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    conn.execute(
                        "insert into code_chunks(root, path, line, content) values (?, ?, ?, ?)",
                        (str(self.root), rel, line_number, stripped),
                    )
                    count += 1
            # return the count of indexed lines
            return count

    def search(self, query: str, limit: int = 20) -> list[CodeSearchResult]:
        # parse the query into keywords
        terms = [t.lower() for t in query.split() if t.strip()]
        if not terms:
            return []
        # search code by root (isolation) and coarse-match the first term in SQL
        with self._connect() as conn:
            rows = conn.execute(
                "select path, line, content from code_chunks "
                "where root = ? and lower(content) like ? order by path, line limit 500",
                (str(self.root), f"%{terms[0]}%"),
            ).fetchall()
        # refine in Python: keep rows containing all terms
        results = []
        for path, line, content in rows:
            if all(term in content.lower() for term in terms):
                results.append(CodeSearchResult(path, int(line), content))
            if len(results) >= limit:
                break
        return results
