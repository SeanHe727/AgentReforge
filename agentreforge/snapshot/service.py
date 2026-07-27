"""SnapshotService: content-addressed point-in-time checkpoints of the project.

Like git's object model: each unique file content is stored once as a blob named
by its SHA-256 hash; each snapshot is a small manifest mapping path -> hash.
Unchanged files across snapshots share a blob, so N snapshots cost only the
deltas + tiny manifests (not N full copies). Stored outside the project so it
never touches the user's .git.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

_SKIP = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".agentreforge",
    "dist",
    "build",
    ".pytest_cache",
}


class SnapshotService:
    def __init__(self, cwd: str | Path, store_root: str | Path | None = None):
        self.cwd = Path(cwd).resolve()
        root = Path(store_root or Path.home() / ".agentreforge" / "snapshots")
        self.store = root / _project_key(self.cwd)
        self.blobs = self.store / "blobs"
        self.snaps = self.store / "snaps"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.snaps.mkdir(parents=True, exist_ok=True)

    def create(self, label: str = "manual") -> str:
        # store each file's content by hash (dedup), record path->hash in a manifest.
        files: dict[str, str] = {}
        for path in self._iter_files():
            rel = str(path.relative_to(self.cwd))
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            blob = self.blobs / digest
            if not blob.exists():  # only write content we haven't seen before
                blob.write_bytes(data)
            files[rel] = digest

        snap_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f_") + label
        manifest = {
            "id": snap_id,
            "label": label,
            "created_at": datetime.now().isoformat(),
            "cwd": str(self.cwd),
            "files": files,
        }
        (self.snaps / f"{snap_id}.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return snap_id

    def list(self) -> list[dict]:
        # newest first; strip the (large) files map for a compact listing.
        out = []
        for f in sorted(self.snaps.glob("*.json"), reverse=True):
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append(
                {k: v for k, v in m.items() if k != "files"} | {"file_count": len(m["files"])}
            )
        return out

    def restore(self, ref: str) -> str:
        manifest = self._resolve(ref)
        for rel, digest in manifest["files"].items():
            target = self.cwd / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.blobs / digest).read_bytes())
        return manifest["id"]

    def _iter_files(self):
        for path in self.cwd.rglob("*"):
            if any(part in _SKIP for part in path.relative_to(self.cwd).parts):
                continue
            if path.is_file():
                yield path

    def _resolve(self, ref: str) -> dict:
        manifests = sorted(self.snaps.glob("*.json"), reverse=True)
        if ref.isdigit():
            idx = int(ref) - 1
            if 0 <= idx < len(manifests):
                return json.loads(manifests[idx].read_text(encoding="utf-8"))
        else:
            f = self.snaps / f"{ref}.json"
            if f.exists():
                return json.loads(f.read_text(encoding="utf-8"))
        raise ValueError(f"snapshot not found: {ref}")


def _project_key(cwd: Path) -> str:
    digest = hashlib.md5(str(cwd).encode("utf-8")).hexdigest()[:12]
    return f"{cwd.name}_{digest}"
