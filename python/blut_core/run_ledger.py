"""Write to BLUT's append-only run ledger from a Python trainer.

WHY THIS EXISTS. ADR 0154 gave BLUT a run ledger with a Rust writer, a CLI, and
a rendered doc tree — and nothing that a trainer calls. The consequence was
measured on 2026-08-03: `lineage.db` held 167 runs whose newest was six weeks
old, a live PCCP-bound production run wrote to neither the ledger nor the
database, and answering "has experiment E1 been run?" meant reading prose in a
stage card. The prose said "never-run"; the card's own caveat field recorded an
earlier attempt at R≈0.17 that had been discarded as invalid. A record nothing
writes to is a record that gets contradicted by the thing it was supposed to
describe.

THE TWO FIELDS THAT ONLY THE LAUNCHER KNOWS. `intent` and `experiment` cannot be
recovered downstream. Whether a run is a smoke test or a campaign, and which
hypothesis it tests, are facts about why it was started — invisible in its
outputs. `lineage.db` has an `experiment` column and it is populated 0 times in
167 rows, which is what happens when a field is available but nothing at the
launch site fills it.

`seed` is here for the same reason and one more: it is the replicate axis. Runs
sharing (experiment, config_fingerprint) and differing only by seed are repeats,
and repeats are the only thing that turns a difference between two numbers into
a claim. See `blut::run_ledger::compare`.

THE WIRE FORMAT IS RUST'S. Records are read by `blut ledger` and by
`tools/training_log.py`; a shape mismatch here is silent data loss, so the field
names, the `kind` tag and the snake_case enums below mirror
`training/engine/src/run_ledger.rs` exactly and are covered by a round-trip test.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA = "blut.run-ledger/v1"
LEDGER_FILENAME = "run-ledger.jsonl"

# Mirrors the Rust enums. Kept as frozensets rather than a class so a typo is a
# loud rejection at the call site instead of a record the reader silently drops.
INTENTS = frozenset({"smoke", "probe", "campaign"})
OUTCOMES = frozenset({"completed", "diverged", "killed", "oom", "failed"})
TIERS = frozenset({"scratch", "recorded", "canonical"})


def data_dir() -> Path:
    """Mirror of `blut::paths::data_dir()`.

    Must agree with the Rust side exactly: a shim that writes somewhere the
    reader does not look fails silently, which is the failure mode this whole
    module exists to end.
    """
    override = os.environ.get("LAMU_TRAIN_DATA_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "lamu" / "train-data"


def ledger_path() -> Path:
    return data_dir() / LEDGER_FILENAME


@contextmanager
def _locked_append(path: Path) -> Iterator[Any]:
    """Append under an advisory exclusive lock.

    O_APPEND alone is not enough. POSIX guarantees atomicity only up to
    PIPE_BUF (4096 bytes), and a record carrying a long argv or a checkpoint
    list exceeds that — the legacy `experiment_log.py` relied on the unstated
    guarantee with a per-process-only lock. Concurrent trainers are normal here
    (a sweep launches several), so the interleaving is not theoretical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _append(record: Mapping[str, Any], path: Path | None = None) -> None:
    target = path or ledger_path()
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _locked_append(target) as handle:
        handle.write(line + "\n")


def _identity(
    blut_job_id: str,
    trainer_run_id: str | None = None,
    git_sha: str | None = None,
    config_fingerprint: str | None = None,
    checkpoint_sha256: list[str] | None = None,
    pccp_change_id: str | None = None,
    attestation_ref: str | None = None,
    ledger_rows: list[str] | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {"blut_job_id": blut_job_id}
    # Omit empties rather than writing nulls: the Rust side uses
    # skip_serializing_if, and matching it keeps a round-trip byte-stable.
    for key, value in (
        ("trainer_run_id", trainer_run_id),
        ("git_sha", git_sha),
        ("config_fingerprint", config_fingerprint),
        ("pccp_change_id", pccp_change_id),
        ("attestation_ref", attestation_ref),
    ):
        if value:
            identity[key] = value
    if checkpoint_sha256:
        identity["checkpoint_sha256"] = list(checkpoint_sha256)
    if ledger_rows:
        identity["ledger_rows"] = list(ledger_rows)
    return identity


class RunLedger:
    """One run's lifecycle. Start on launch, end exactly once.

    Use `track()` unless you need manual control — an un-ended run is worse than
    an unrecorded one, because it reads as still-running forever.
    """

    def __init__(
        self,
        run_uid: str,
        recipe: str,
        *,
        tenant: str = "",
        intent: str = "probe",
        experiment: str | None = None,
        seed: int | None = None,
        path: Path | None = None,
        **identity_fields: Any,
    ) -> None:
        if intent not in INTENTS:
            raise ValueError(f"intent must be one of {sorted(INTENTS)}, got {intent!r}")
        self.run_uid = run_uid
        self.recipe = recipe
        self.tenant = tenant
        self.intent = intent
        self.experiment = experiment
        self.seed = seed
        self.path = path
        self._identity = _identity(
            identity_fields.pop("blut_job_id", run_uid), **identity_fields
        )
        self._started_at: float | None = None
        self._ended = False

    def start(self) -> None:
        self._started_at = time.time()
        record: dict[str, Any] = {
            "kind": "run_started",
            "schema": SCHEMA,
            "run_uid": self.run_uid,
            "recipe": self.recipe,
            "intent": self.intent,
            "started_unix": int(self._started_at),
            "tenant": self.tenant,
            "identity": self._identity,
        }
        if self.experiment:
            record["experiment"] = self.experiment
        if self.seed is not None:
            record["seed"] = int(self.seed)
        _append(record, self.path)

    def update_identity(self, **fields: Any) -> None:
        """Fill join-key fields that only exist once the run has started.

        A trainer computes its own run id and config hash while setting up, not
        at launch, and the ledger has to be opened BEFORE that so a crash during
        setup is still recorded. Rather than delay the start record — which would
        lose exactly the runs most worth having — identity is completed here and
        written on the `run_ended` record.

        Only the start record is already on disk with the partial identity, and
        that is correct: it is what was known at the time. The ledger is
        append-only; nothing is rewritten.
        """
        self._identity.update(
            {k: v for k, v in _identity(self._identity["blut_job_id"], **fields).items()
             if k != "blut_job_id"}
        )

    def end(
        self,
        outcome: str,
        *,
        tier: str = "scratch",
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        if self._ended:
            return
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}, got {outcome!r}")
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {sorted(TIERS)}, got {tier!r}")
        self._ended = True
        now = time.time()
        record: dict[str, Any] = {
            "kind": "run_ended",
            "schema": SCHEMA,
            "run_uid": self.run_uid,
            "ended_unix": int(now),
            "duration_secs": int(max(0.0, now - (self._started_at or now))),
            "outcome": outcome,
            "tier": tier,
            "identity": self._identity,
        }
        if metrics:
            # Headline numbers only. Curves stay in wandb (ADR 0034, unchanged
            # by 0154) — a ledger that grows without bound stops being readable.
            record["metrics"] = {str(k): float(v) for k, v in metrics.items()}
        _append(record, self.path)


@contextmanager
def track(
    run_uid: str,
    recipe: str,
    *,
    tenant: str = "",
    intent: str = "probe",
    experiment: str | None = None,
    seed: int | None = None,
    path: Path | None = None,
    **identity_fields: Any,
) -> Iterator[RunLedger]:
    """Record a run whatever happens to it.

    A ledger that only captures clean exits is a ledger of successes, and the
    expensive knowledge is in the other column: a 300-epoch divergence is worth
    recording precisely so nobody repeats it. So SIGTERM and SIGINT are trapped
    and an OOM kill is inferred, because the common ways a training run dies are
    exactly the ways an ordinary `finally` never runs.

    `metrics` and the final tier are set by calling `.end()` explicitly inside
    the block; leaving the block without doing so records the outcome the exit
    implies.
    """
    ledger = RunLedger(
        run_uid,
        recipe,
        tenant=tenant,
        intent=intent,
        experiment=experiment,
        seed=seed,
        path=path,
        **identity_fields,
    )
    ledger.start()

    def _on_signal(signum: int, _frame: Any) -> None:
        ledger.end("killed" if signum == signal.SIGTERM else "killed")
        # Re-raise as the default action so the caller still dies the way it was
        # told to; swallowing a SIGTERM to finish bookkeeping is worse.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, _on_signal)
        except (ValueError, OSError):
            # Not the main thread, or a platform without it. Not fatal.
            pass
    atexit.register(lambda: ledger.end("failed"))

    try:
        yield ledger
    except MemoryError:
        ledger.end("oom")
        raise
    except BaseException:
        ledger.end("failed")
        raise
    else:
        ledger.end("completed")
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass


def _cli() -> int:
    """`python -m blut_core.run_ledger --self-check` — prove the shim writes
    what the Rust reader expects, without needing a trainer."""
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()

    if not args.self_check:
        print(f"ledger path: {ledger_path()}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / LEDGER_FILENAME
        with track(
            "selfcheck:1",
            "lamquant_joint_codec",
            tenant="lamquant",
            intent="smoke",
            experiment="SELF-CHECK",
            seed=7,
            path=target,
            trainer_run_id="selfcheck",
            config_fingerprint="deadbeef",
        ) as run:
            run.end("completed", tier="scratch", metrics={"best_val_r": 0.5})
        lines = [json.loads(l) for l in target.read_text().splitlines()]

    assert [l["kind"] for l in lines] == ["run_started", "run_ended"], lines
    assert lines[0]["schema"] == SCHEMA
    assert lines[0]["experiment"] == "SELF-CHECK"
    assert lines[0]["seed"] == 7
    assert lines[1]["metrics"]["best_val_r"] == 0.5
    print(json.dumps(lines, indent=2))
    print("self-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
