from __future__ import annotations

import subprocess
import sys

from scripts.create_mini_demo import create_demo


def test_create_mini_demo_builds_a_runnable_git_repo(tmp_path):
    destination = tmp_path / "mini-agent"

    create_demo(destination)

    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=destination,
        capture_output=True,
        text=True,
        check=True,
    )
    run = subprocess.run(
        [sys.executable, "-m", "mini_agent", "echo hello", "add 2 3", "reverse abc"],
        cwd=destination,
        capture_output=True,
        text=True,
        check=True,
    )
    assert inside.stdout.strip() == "true"
    assert run.stdout.splitlines() == ["hello", "5", "cba"]
