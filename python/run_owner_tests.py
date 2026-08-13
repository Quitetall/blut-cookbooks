#!/usr/bin/env python3
"""Run BLUT backend tests covered by owner contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OWNER_PYTHON_TESTS = (
    "python/tests/test_run_manifest.py",
    "python/blut_core/tests/test_async_compute.py",
    "python/blut_core/tests/test_async_io_contract.py",
    "python/blut_core/tests/test_parallel_strategy.py",
)


def _run(argv: Sequence[str], *, env: dict[str, str] | None = None) -> int:
    print(f"owner-tests: running {' '.join(argv)}", flush=True)
    try:
        return subprocess.run(argv, cwd=ROOT, env=env, check=False).returncode
    except FileNotFoundError:
        print(f"owner-tests: command not found: {argv[0]}", file=sys.stderr)
        return 127


def main() -> int:
    for test_path in OWNER_PYTHON_TESTS:
        if not (ROOT / test_path).is_file():
            print(
                f"owner-tests: missing declared suite: {test_path}",
                file=sys.stderr,
            )
            return 2

    rust_status = _run(("cargo", "test", "--workspace", "--no-fail-fast"))
    python_env = os.environ.copy()
    python_root = str(ROOT / "python")
    existing_pythonpath = python_env.get("PYTHONPATH")
    python_env["PYTHONPATH"] = (
        f"{python_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else python_root
    )
    python_status = _run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *OWNER_PYTHON_TESTS,
        ),
        env=python_env,
    )
    return rust_status or python_status


if __name__ == "__main__":
    raise SystemExit(main())
