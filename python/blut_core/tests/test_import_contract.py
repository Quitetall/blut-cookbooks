"""Compatibility tests for the public ``blut_core`` import surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_keeps_torch_lazy():
    python_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import blut_core; assert 'torch' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_cli_submodule_remains_importable():
    from blut_core import read_metric

    assert read_metric.__name__ == "blut_core.read_metric"
