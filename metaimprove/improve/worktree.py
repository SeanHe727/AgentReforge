"""Git worktree isolation for the improvement pipeline.

The Writer edits code in a *separate* working directory (a git worktree) on its
own branch, so the user's main working tree is never touched. If evaluation
passes we can merge the branch back; if it fails we remove the worktree and
nothing remains — a clean, built-in rollback. This is the "isolation" half of
the reliability story; content-addressed snapshots are the other half.

Everything runs through `git worktree`, so the isolated copy shares the one
`.git` object store (cheap) rather than being a full clone.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class GitError(RuntimeError):
    """A git command exited non-zero (message carries stderr)."""


@dataclass
class GitResult:
    code: int
    stdout: str
    stderr: str


async def _run_git(*args: str, cwd: Path) -> GitResult:
    """Run `git <args>` in cwd, capturing output. Never raises on non-zero."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return GitResult(
        proc.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


async def _git(*args: str, cwd: Path) -> str:
    """Run git and return trimmed stdout, raising GitError on failure."""
    res = await _run_git(*args, cwd=cwd)
    if res.code != 0:
        raise GitError(f"git {' '.join(args)} failed ({res.code}): {res.stderr.strip()}")
    return res.stdout.strip()


class WorktreeSession:
    """Async context manager around `git worktree add/remove`.

    Usage:
        async with WorktreeSession(repo_root, base="HEAD") as wt:
            # Writer edits files under wt.path and runs tests there...
            await wt.commit("apply improvement task T1")
            changes = await wt.diff()
        # on exit the worktree dir and its branch are removed automatically
        # (pass keep=True to leave them in place for inspection).
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        base: str = "HEAD",
        branch: str | None = None,
        worktrees_root: str | Path | None = None,
        keep: bool = False,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.base = base
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.branch = branch or f"improve/{stamp}"
        root = Path(worktrees_root or self.repo_root / ".meta-improve" / "worktrees")
        self.path = (root / stamp).resolve()
        self.keep = keep
        self._base_commit: str | None = None
        self._entered = False

    @property
    def base_commit(self) -> str | None:
        """The concrete commit the worktree branched from (resolved on enter)."""
        return self._base_commit

    async def __aenter__(self) -> WorktreeSession:
        # Must be inside a real git working tree.
        inside = await _run_git("rev-parse", "--is-inside-work-tree", cwd=self.repo_root)
        if inside.code != 0 or inside.stdout.strip() != "true":
            raise GitError(f"{self.repo_root} is not a git working tree")
        # Pin base to a concrete commit so the record is stable even if the main
        # branch moves while the session runs.
        self._base_commit = await _git("rev-parse", self.base, cwd=self.repo_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        await _git(
            "worktree",
            "add",
            "-b",
            self.branch,
            str(self.path),
            self._base_commit,
            cwd=self.repo_root,
        )
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self.keep:
            await self.remove()

    async def commit(self, message: str, *, expected_tree: str | None = None) -> str | None:
        """Stage everything and commit. Returns the new commit sha, or None if
        there was nothing to commit.  When `expected_tree` is given, refuse to
        commit a tree other than the exact candidate that passed delivery."""
        actual_tree = await self.snapshot()
        if expected_tree is not None and actual_tree != expected_tree:
            raise GitError(
                f"candidate tree changed after delivery: {actual_tree} != {expected_tree}"
            )
        status = await _git("status", "--porcelain", cwd=self.path)
        if not status.strip():
            return None
        await _git("commit", "-m", message, cwd=self.path)
        commit = await _git("rev-parse", "HEAD", cwd=self.path)
        committed_tree = await _git("rev-parse", f"{commit}^{{tree}}", cwd=self.path)
        if expected_tree is not None and committed_tree != expected_tree:
            raise GitError(
                f"committed tree differs from delivered tree: {committed_tree} != {expected_tree}"
            )
        return commit

    async def diff(self) -> str:
        """Full diff of the worktree against its base commit — every change the
        Writer made, whether committed or still in the working tree, including
        brand-new files. `add -N` (intent-to-add) registers untracked paths so
        they show up as additions; plain `git diff <commit>` would omit them."""
        assert self._base_commit is not None
        await _git("add", "-N", ".", cwd=self.path)
        return await _git("diff", self._base_commit, cwd=self.path)

    async def changed_paths(self) -> list[str]:
        """Repo-relative paths changed vs the base commit (tracked + untracked).
        `add -N` first so brand-new files are included."""
        assert self._base_commit is not None
        await _git("add", "-N", ".", cwd=self.path)
        out = await _git("diff", "--name-only", self._base_commit, cwd=self.path)
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def head(self) -> str:
        """The worktree's current HEAD commit."""
        return await _git("rev-parse", "HEAD", cwd=self.path)

    async def snapshot(self) -> str:
        """A tree-hash of the current working tree: a diff reference point that
        makes NO commit and doesn't touch the branch (git write-tree)."""
        await _git("add", "-A", cwd=self.path)
        return await _git("write-tree", cwd=self.path)

    async def diff_since(self, ref: str) -> str:
        """Full diff of the working tree since `ref` (a commit or a snapshot tree).
        `add -N` registers untracked files so brand-new files show as additions."""
        await _git("add", "-N", ".", cwd=self.path)
        return await _git("diff", ref, cwd=self.path)

    async def reset_hard(self, ref: str) -> None:
        """Discard all changes back to `ref` (roll back a failed loop). Best-effort."""
        await _run_git("reset", "--hard", ref, cwd=self.path)
        await _run_git("clean", "-fd", cwd=self.path)

    async def restore_snapshot(self, tree: str) -> None:
        """Restore an exact uncommitted tree snapshot, including staged new files."""
        await _git("read-tree", "--reset", "-u", tree, cwd=self.path)
        await _run_git("clean", "-fd", cwd=self.path)

    async def changed_since(self, ref: str) -> list[str]:
        """Repo-relative paths changed between `ref` and the current worktree
        state (tracked + untracked). Used to check the frozen tests are intact."""
        await _git("add", "-N", ".", cwd=self.path)
        out = await _git("diff", "--name-only", ref, cwd=self.path)
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def read_base(self, rel_path: str) -> str | None:
        """The content of `rel_path` at the base commit, or None if it didn't
        exist there (i.e. the Writer created it new)."""
        res = await _run_git("show", f"{self._base_commit}:{rel_path}", cwd=self.path)
        return res.stdout if res.code == 0 else None

    async def merge_back(self, *, message: str | None = None, ff_only: bool = False) -> str:
        """Merge this branch into the main repo's current branch. Returns the
        resulting HEAD sha. Only call after evaluation accepts the change."""
        args = ["merge", "--no-edit"]
        if ff_only:
            args.append("--ff-only")
        if message:
            args += ["-m", message]
        args.append(self.branch)
        await _git(*args, cwd=self.repo_root)
        return await _git("rev-parse", "HEAD", cwd=self.repo_root)

    async def remove(self, *, keep_branch: bool = False) -> None:
        """Tear down the worktree dir; delete its branch unless keep_branch (used
        when delivered commits must stay reachable). Best-effort (non-raising)."""
        if not self._entered:
            return
        await _run_git("worktree", "remove", "--force", str(self.path), cwd=self.repo_root)
        if not keep_branch:
            await _run_git("branch", "-D", self.branch, cwd=self.repo_root)
        self._entered = False
