from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from eval.mini.cases import cases
from eval.mini.run import _write_starter


def test_mini_case_ids_are_unique_and_starters_fail_before_agent_changes():
    benchmark_cases = cases()

    assert len(benchmark_cases) == 4
    assert len({case.id for case in benchmark_cases}) == len(benchmark_cases)

    for case in benchmark_cases:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_starter(case, root)
            hidden = root.parent / f"test_{case.id}.py"
            hidden.write_text(case.hidden_test, encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(root)
            result = subprocess.run(
                [sys.executable, str(hidden)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

        assert result.returncode != 0, f"{case.id} has no baseline failure/headroom"
