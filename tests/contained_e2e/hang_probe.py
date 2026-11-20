"""A model ingredient that never finishes building, for the Windows
containment end-to-end check (`run.py`).

Built by the trainer through the `custom_module` model ingredient, it starts
a grandchild process, records both pids, then sleeps. That gives the check a
real training process tree, held at a known point, whose fate it can observe
from outside after BLUT times the stage out or its driver dies.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


class HangingNet(torch.nn.Module):
    def __init__(self, pid_file: str):
        super().__init__()
        grandchild = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(900)"])
        target = Path(pid_file)
        partial = target.with_suffix(".tmp")
        partial.write_text(json.dumps(
            {"trainer": os.getpid(), "grandchild": grandchild.pid}))
        os.replace(partial, target)
        print(f"[hang_probe] trainer {os.getpid()} grandchild {grandchild.pid}",
              flush=True)
        time.sleep(900)
