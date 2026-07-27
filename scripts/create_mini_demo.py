"""Create a fresh standalone Git repository from the tracked mini-agent template."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _git(destination: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    )


def create_demo(destination: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    template = project_root / "examples" / "mini-agent"
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; choose a new path to preserve existing data"
        )
    shutil.copytree(template, destination)
    _git(destination, "init", "-b", "main")
    _git(destination, "add", ".")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AgentReforge Demo",
            "-c",
            "user.email=demo@agentreforge.local",
            "commit",
            "-m",
            "baseline mini-agent",
        ],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    create_demo(destination)
    print(f"created: {destination}")
    print("baseline: python3 -m mini_agent 'echo hello' 'add 2 3' 'reverse abc'")
    print(f"improve: meta-improve improve --cwd {destination} --loops 10 --keep ...")


if __name__ == "__main__":
    main()
