#!/usr/bin/env python3
"""Run BLUT backend tests covered by owner contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
A02_PYTHON_TEST = "python/tests/test_run_manifest.py"


def _run(argv: Sequence[str]) -> int:
    print(f"owner-tests: running {' '.join(argv)}", flush=True)
    try:
        return subprocess.run(argv, cwd=ROOT, check=False).returncode
    except FileNotFoundError:
        print(f"owner-tests: command not found: {argv[0]}", file=sys.stderr)
        return 127


def main() -> int:
    if not (ROOT / A02_PYTHON_TEST).is_file():
        print(
            f"owner-tests: missing declared A02 suite: {A02_PYTHON_TEST}",
            file=sys.stderr,
        )
        return 2

    rust_status = _run(("cargo", "test", "--workspace", "--no-fail-fast"))
    python_status = _run((sys.executable, "-m", "pytest", "-q", A02_PYTHON_TEST))
    return rust_status or python_status


if __name__ == "__main__":
    raise SystemExit(main())
