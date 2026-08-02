"""Run paired demo_agent versions on the deterministic mini benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .cases import MiniCase, cases


@dataclass
class CaseResult:
    arm: str
    case_id: str
    repeat: int
    passed: bool
    agent_exit_code: int
    test_exit_code: int
    duration_seconds: float
    agent_summary: str
    test_output: str
    solution_dir: str


def _write_starter(case: MiniCase, destination: Path) -> None:
    for relative, content in case.starter_files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _agent_commit(agent_repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=agent_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def run_case(
    *,
    arm: str,
    agent_repo: Path,
    case: MiniCase,
    repeat: int,
    output_root: Path,
    max_steps: int,
    timeout: int,
) -> CaseResult:
    solution_dir = output_root / "solutions" / arm / f"repeat_{repeat}" / case.id
    if solution_dir.exists():
        shutil.rmtree(solution_dir)
    solution_dir.mkdir(parents=True)
    _write_starter(case, solution_dir)

    started = time.monotonic()
    agent = subprocess.run(
        [
            sys.executable,
            "-m",
            "demo_agent",
            case.prompt,
            "--dir",
            str(solution_dir),
            "--max-steps",
            str(max_steps),
        ],
        cwd=agent_repo,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )

    with tempfile.TemporaryDirectory(prefix="agentreforge-mini-test-") as temp_dir:
        hidden_test = Path(temp_dir) / f"test_{case.id}.py"
        hidden_test.write_text(case.hidden_test, encoding="utf-8")
        test_env = os.environ.copy()
        test_env["PYTHONDONTWRITEBYTECODE"] = "1"
        test_env["PYTHONPATH"] = str(solution_dir)
        tested = subprocess.run(
            [sys.executable, str(hidden_test)],
            cwd=solution_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=test_env,
        )

    duration = round(time.monotonic() - started, 3)
    summary = (agent.stdout or agent.stderr).strip()
    test_output = (tested.stdout + tested.stderr).strip()
    return CaseResult(
        arm=arm,
        case_id=case.id,
        repeat=repeat,
        passed=agent.returncode == 0 and tested.returncode == 0,
        agent_exit_code=agent.returncode,
        test_exit_code=tested.returncode,
        duration_seconds=duration,
        agent_summary=summary[-2000:],
        test_output=test_output[-4000:],
        solution_dir=str(solution_dir),
    )


def _write_summary(
    output_root: Path,
    results: list[CaseResult],
    metadata: dict,
) -> None:
    lines = [
        "# AgentReforge mini benchmark",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Max steps: {metadata['max_steps']}",
        f"- Repeats: {metadata['repeats']}",
        "- Primary metric: deterministic hidden-test case pass rate",
        "",
        "| Arm | Passed | Total | Pass rate | Mean seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in metadata["arms"]:
        arm_results = [result for result in results if result.arm == arm]
        passed = sum(result.passed for result in arm_results)
        total = len(arm_results)
        rate = 100 * passed / total if total else 0
        mean = (
            sum(result.duration_seconds for result in arm_results) / total
            if total
            else 0
        )
        lines.append(f"| {arm} | {passed} | {total} | {rate:.1f}% | {mean:.1f} |")
    lines += [
        "",
        "## Paired results",
        "",
        "| Case | Repeat | " + " | ".join(metadata["arms"]) + " |",
        "|---|---:|" + "|".join("---:" for _ in metadata["arms"]) + "|",
    ]
    for case in metadata["cases"]:
        for repeat in range(1, metadata["repeats"] + 1):
            cells = []
            for arm in metadata["arms"]:
                match = next(
                    result
                    for result in results
                    if result.arm == arm
                    and result.case_id == case
                    and result.repeat == repeat
                )
                cells.append("PASS" if match.passed else "FAIL")
            lines.append(f"| {case} | {repeat} | " + " | ".join(cells) + " |")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=AGENT_REPO",
        help="repeat for each paired demo_agent version",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args = parser.parse_args()

    arms: dict[str, Path] = {}
    for value in args.arm:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("--arm must use NAME=AGENT_REPO")
        arms[name] = Path(path).resolve()
    selected = [
        case
        for case in cases()
        if not args.selected_cases or case.id in args.selected_cases
    ]
    if not selected:
        parser.error("no benchmark cases selected")

    output_root = Path(args.out).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": os.environ.get("CODER_MODEL", "gpt-5.4-mini"),
        "max_steps": args.max_steps,
        "repeats": args.repeats,
        "arms": list(arms),
        "agent_repos": {name: str(path) for name, path in arms.items()},
        "agent_commits": {name: _agent_commit(path) for name, path in arms.items()},
        "cases": [case.id for case in selected],
    }
    results: list[CaseResult] = []
    for repeat in range(1, args.repeats + 1):
        for case in selected:
            for arm, agent_repo in arms.items():
                print(f"[{arm}] repeat={repeat} case={case.id}", flush=True)
                result = run_case(
                    arm=arm,
                    agent_repo=agent_repo,
                    case=case,
                    repeat=repeat,
                    output_root=output_root,
                    max_steps=args.max_steps,
                    timeout=args.timeout,
                )
                results.append(result)
                print(
                    f"  {'PASS' if result.passed else 'FAIL'} "
                    f"agent={result.agent_exit_code} tests={result.test_exit_code} "
                    f"{result.duration_seconds:.1f}s",
                    flush=True,
                )
                payload = {
                    "metadata": metadata,
                    "results": [asdict(item) for item in results],
                }
                (output_root / "results.json").write_text(
                    json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8",
                )
    _write_summary(output_root, results, metadata)
    print(f"results: {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
