#!/usr/bin/env python3
"""Stands in for `python -m blut_core.trainer` in the Linux containment check.

`linux.py` points `BLUT_PYTHON` here, so the core cookbook's `train_model`
stage runs this instead of a real trainer: no torch, a few tens of MiB of RAM.
It reads the stage's config, commits 64 MiB so the reported peak is visibly
above an idle interpreter, writes a checkpoint file where the stage expects
one, and exits 0. With `BLUT_STUB_HANG=<file>` it instead starts a grandchild
process, records both pids in `<file>`, and sleeps, so the check can watch a
stage timeout or a cancel take the whole tree down.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    if args[:2] != ["-m", "blut_core.trainer"] or "--config" not in args:
        print(f"stub_trainer: unexpected arguments {args}", file=sys.stderr)
        return 2
    config = json.loads(Path(args[args.index("--config") + 1]).read_text())
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"[stub-trainer] pid {os.getpid()} tag {os.environ.get('BLUT_STUB_TAG')}",
          flush=True)

    hang = os.environ.get("BLUT_STUB_HANG")
    if hang:
        grandchild = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(900)"])
        partial = Path(hang + ".tmp")
        partial.write_text(json.dumps(
            {"trainer": os.getpid(), "grandchild": grandchild.pid}))
        os.replace(partial, hang)
        time.sleep(900)
        return 1

    block = bytearray(64 * 1024 * 1024)
    for i in range(0, len(block), 4096):
        block[i] = 1
    (out / "model.pt").write_bytes(b"stub checkpoint\n")
    print(f"[stub-trainer] wrote {out / 'model.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
