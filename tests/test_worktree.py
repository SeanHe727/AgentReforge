from __future__ import annotations

import asyncio
import subprocess

from metaimprove.improve.worktree import WorktreeSession


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_task_and_loop_diff_use_their_own_fixed_boundaries(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "agent.py")
    _git(repo, "commit", "-m", "base")

    async def exercise():
        worktree = WorktreeSession(
            repo,
            worktrees_root=tmp_path / "worktrees",
            keep=False,
        )
        async with worktree:
            loop_base = await worktree.head()
            task_start = await worktree.snapshot()
            (worktree.path / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

            task_diff = await worktree.diff_since(task_start)
            loop_diff = await worktree.diff_since(loop_base)

            assert "-VALUE = 1" in task_diff
            assert "+VALUE = 2" in task_diff
            assert task_diff == loop_diff
            candidate_tree = await worktree.snapshot()
            (worktree.path / "agent.py").write_text("VALUE = 999\n", encoding="utf-8")
            (worktree.path / "reviewer.tmp").write_text("side effect\n", encoding="utf-8")
            await worktree.restore_snapshot(candidate_tree)
            assert (worktree.path / "agent.py").read_text(encoding="utf-8") == "VALUE = 2\n"
            assert not (worktree.path / "reviewer.tmp").exists()
            commit = await worktree.commit("loop", expected_tree=candidate_tree)
            assert commit is not None

    asyncio.run(exercise())
