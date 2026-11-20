"""checkpoint_slot — the trainer half of BLUT's mid-stage checkpoint contract
(engine CONTRACT.md §8).

The engine gives each attempt of a stage that opts in a durable directory (the
*slot*) and tells the trainer, through the environment, whether to resume:

- ``BLUT_CHECKPOINT_DIR`` — the slot. Checkpoints and the ``state.json``
  marker live here.
- ``BLUT_RESUME`` — ``1``: resume from the newest valid checkpoint; ``0``:
  start from step 0.
- ``BLUT_RUN_ID`` / ``BLUT_ATTEMPT_ID`` — copied into the marker so the engine
  can tell which attempt wrote it.

With no ``BLUT_CHECKPOINT_DIR`` the trainer runs exactly as before.

What this module does for the trainer:

- saves ``step-<N>.pt`` atomically through ``blut_core.checkpoint.save`` (temp
  file, fsync, rename, then a SHA-256 sidecar) and fsyncs the directory;
- keeps the newest ``keep`` checkpoints, so a corrupt newest one still leaves
  an older one to resume from;
- on resume, loads the newest checkpoint that has a matching sidecar, carries
  the same config fingerprint and world size, and has every envelope key;
  anything else is skipped with a message saying why;
- writes the marker (``status``, ``run_id``, ``pid``, ``host``,
  ``heartbeat_unix``, ``step``, ``epoch``, ``attempt_id``,
  ``resumed_from_step``) atomically, and refreshes its heartbeat from a
  background thread every ``HEARTBEAT_SECS``.

Only rank 0 touches the slot. torch is imported lazily.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import checkpoint

__all__ = [
    "ENV_CHECKPOINT_DIR",
    "ENV_RESUME",
    "ENV_RUN_ID",
    "ENV_ATTEMPT_ID",
    "ENVELOPE_KEYS",
    "HEARTBEAT_SECS",
    "Slot",
    "config_fingerprint",
]

ENV_CHECKPOINT_DIR = "BLUT_CHECKPOINT_DIR"
ENV_RESUME = "BLUT_RESUME"
ENV_RUN_ID = "BLUT_RUN_ID"
ENV_ATTEMPT_ID = "BLUT_ATTEMPT_ID"

# CONTRACT.md §4 canonical envelope keys a resumable checkpoint must carry.
ENVELOPE_KEYS = ("model", "opt", "config", "step", "rng")

# Well under the engine's 60 s heartbeat interval (CONTRACT.md §3).
HEARTBEAT_SECS = 30

MARKER = "state.json"
_STEP_RE = re.compile(r"^step-(\d{10})\.pt$")

# Paths differ between jobs that run the same training (a requeued job is a new
# job directory); the engine's slot key already covers the inputs' content.
_UNFINGERPRINTED = ("dataset_path", "output_dir")


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of ``config`` without its path keys. A
    checkpoint carrying a different fingerprint is never resumed."""
    import hashlib
    body = {k: v for k, v in config.items() if k not in _UNFINGERPRINTED}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _fsync_dir(path: Path) -> None:
    # Makes a rename in `path` durable. Not every platform can open a directory.
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    body = (json.dumps(obj, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


class Slot:
    """One attempt's view of its checkpoint slot."""

    def __init__(self, directory, resume: bool, run_id: str = "", attempt_id: str = "",
                 keep: int = 2):
        self.dir = Path(directory)
        self.resume = bool(resume)
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.keep = max(1, int(keep))
        self._lock = threading.Lock()
        self._marker: Dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, keep: int = 2) -> Optional["Slot"]:
        """The slot the engine assigned, or ``None`` when it assigned none."""
        env = os.environ if environ is None else environ
        directory = env.get(ENV_CHECKPOINT_DIR, "").strip()
        if not directory:
            return None
        resume = env.get(ENV_RESUME, "0").strip()
        if resume not in ("0", "1"):
            raise ValueError(f"{ENV_RESUME}={resume!r}: expected 0 or 1")
        return cls(directory, resume == "1", env.get(ENV_RUN_ID, ""),
                   env.get(ENV_ATTEMPT_ID, ""), keep=keep)

    # -- checkpoints ---------------------------------------------------------

    def path_for(self, step: int) -> Path:
        return self.dir / f"step-{int(step):010d}.pt"

    def checkpoints(self) -> List[Tuple[int, Path]]:
        """``(step, path)`` of every checkpoint file, newest first."""
        if not self.dir.is_dir():
            return []
        found = []
        for entry in self.dir.iterdir():
            m = _STEP_RE.match(entry.name)
            if m and entry.is_file():
                found.append((int(m.group(1)), entry))
        return sorted(found, reverse=True)

    def discard_all(self) -> None:
        """Delete every checkpoint, sidecar and leftover temp file: the start
        of a fresh attempt. The engine clears the slot too; this guards a
        trainer launched by hand with ``BLUT_RESUME=0``."""
        if not self.dir.is_dir():
            return
        for entry in self.dir.iterdir():
            name = entry.name
            if name.startswith("step-") and entry.is_file():
                entry.unlink(missing_ok=True)

    def remove_temp_files(self) -> None:
        """Delete temp files a killed save left behind."""
        if not self.dir.is_dir():
            return
        for entry in self.dir.iterdir():
            if entry.name.startswith("step-") and entry.name.endswith(".tmp"):
                entry.unlink(missing_ok=True)

    def load_latest(self, fingerprint: str, world_size: int,
                    required: Iterable[str] = ENVELOPE_KEYS) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
        """The newest checkpoint that verifies, or ``(None, None)``.

        A checkpoint is skipped, with a line on stderr saying why, when its
        sidecar is missing or does not match, it does not load, it lacks an
        envelope key, or it was written for another config or world size."""
        required = tuple(required)
        for step, path in self.checkpoints():
            try:
                payload = checkpoint.load(path, require_sidecar=True)
            except checkpoint.CheckpointError as e:
                print(f"[trainer] skipping checkpoint {path.name}: {e}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                print(f"[trainer] skipping checkpoint {path.name}: not a dict", file=sys.stderr)
                continue
            missing = [k for k in required if k not in payload]
            if missing:
                print(f"[trainer] skipping checkpoint {path.name}: missing {missing}",
                      file=sys.stderr)
                continue
            if payload.get("fingerprint") != fingerprint:
                print(f"[trainer] skipping checkpoint {path.name}: written for another config",
                      file=sys.stderr)
                continue
            if int(payload.get("world_size", 1)) != int(world_size):
                print(f"[trainer] skipping checkpoint {path.name}: written by "
                      f"{payload.get('world_size')} ranks, this run has {world_size}",
                      file=sys.stderr)
                continue
            if int(payload["step"]) != step:
                print(f"[trainer] skipping checkpoint {path.name}: holds step {payload['step']}",
                      file=sys.stderr)
                continue
            return payload, path
        return None, None

    def save(self, payload: Dict[str, Any], step: int, epoch: int) -> Path:
        """Write ``payload`` as the step-``step`` checkpoint, make it durable,
        record it in the marker, then drop checkpoints beyond ``keep``."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(step)
        checkpoint.save(payload, path, contract=ENVELOPE_KEYS)
        _fsync_dir(self.dir)
        self.update(step=int(step), epoch=int(epoch))
        for _, old in self.checkpoints()[self.keep:]:
            for victim in (old, old.with_suffix(old.suffix + ".sha256")):
                try:
                    victim.unlink()
                except OSError:
                    pass
        return path

    # -- marker --------------------------------------------------------------

    def begin(self, step: Optional[int], epoch: Optional[int],
              resumed_from: Optional[int]) -> None:
        """Write the marker for this attempt and start the heartbeat."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._marker = {
                "status": "running",
                "run_id": self.run_id,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "heartbeat_unix": int(time.time()),
                "step": step,
                "epoch": epoch,
                "attempt_id": self.attempt_id or None,
                "resumed_from_step": resumed_from,
            }
            _write_json_atomic(self.dir / MARKER, self._marker)
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._beat, name="blut-heartbeat",
                                            daemon=True)
            self._thread.start()

    def update(self, **fields: Any) -> None:
        with self._lock:
            if not self._marker:
                return
            self._marker.update(fields)
            self._marker["heartbeat_unix"] = int(time.time())
            _write_json_atomic(self.dir / MARKER, self._marker)

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECS):
            try:
                self.update()
            except OSError as e:
                print(f"[trainer] heartbeat write failed: {e}", file=sys.stderr)

    def stop_heartbeat(self) -> None:
        """Stop the heartbeat thread, leaving the marker as it is."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def end(self) -> None:
        """Stop the heartbeat and stamp the marker once more. The marker stays
        ``running`` with the final step: if the engine dies before it commits
        the output, the next attempt resumes from the final checkpoint instead
        of training again."""
        self.stop_heartbeat()
        self.update()
