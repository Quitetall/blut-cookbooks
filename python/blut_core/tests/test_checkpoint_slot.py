"""Mid-stage checkpoint slots (engine CONTRACT.md §8), trainer side.

A run that crashes part-way and is resumed from its slot must produce the same
per-step losses, the same epoch losses and the same final weights as a run that
was never interrupted. A checkpoint that is partial, corrupt, or written for
another config is skipped.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import blut_core.trainer as trainer  # noqa: E402
from blut_core import checkpoint as ckpt  # noqa: E402
from blut_core.checkpoint_slot import (  # noqa: E402
    ENV_ATTEMPT_ID, ENV_CHECKPOINT_DIR, ENV_RESUME, ENV_RUN_ID, Slot, config_fingerprint,
)

STEPS_PER_EPOCH = 10
EPOCHS = 2


class Crash(Exception):
    """Stands in for the node going away."""


def _rows():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(40, 4, generator=g)
    y = x.sum(dim=1) * 0.5
    return [{"x": x[i].tolist(), "y": float(y[i])} for i in range(40)]


def _model():
    torch.manual_seed(0)
    # Dropout makes the loss depend on the RNG, so a resume that restored the
    # weights but not the RNG would drift.
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3),
                               torch.nn.Linear(8, 1))


def _config(out: Path, **extra):
    cfg = {
        "dataset_path": "unused",
        "output_dir": str(out),
        "optimizer": {"kind": "optimizer", "name": "adamw", "config": {"lr": 1e-2}},
        "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
        "epochs": EPOCHS,
        "batch_size": 4,
        "device": "cpu",
        "checkpoint_every_steps": 3,
    }
    cfg.update(extra)
    return cfg


def _run(out: Path, losses: list, epoch_losses: list, crash_at=None, **extra):
    """One trainer run; `crash_at` raises on that step call of this run."""
    calls = [0]

    def step(model, optimizer, loss_fn, batch, **kw):
        calls[0] += 1
        if crash_at is not None and calls[0] == crash_at:
            raise Crash(f"node lost at call {crash_at}")
        x = torch.tensor([r["x"] for r in batch])
        y = torch.tensor([[r["y"]] for r in batch])
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        return loss

    def emit(metrics, kind=None):
        if kind == "epoch":
            epoch_losses.append(metrics["loss"])

    ingredients = {
        "model": _model(),
        "loss": None,
        "step": step,
        "logging": emit,
        "checkpoint": lambda payload, path: torch.save(payload, path),
    }
    trainer.train_loop(_config(out, **extra), ingredients, _rows())
    return torch.load(out / "model.pt", weights_only=True)["model"]


def _slot_env(monkeypatch, slot_dir: Path, resume: bool, attempt: str):
    monkeypatch.setenv(ENV_CHECKPOINT_DIR, str(slot_dir))
    monkeypatch.setenv(ENV_RESUME, "1" if resume else "0")
    monkeypatch.setenv(ENV_RUN_ID, "job-1")
    monkeypatch.setenv(ENV_ATTEMPT_ID, attempt)


def _baseline(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_CHECKPOINT_DIR, raising=False)
    losses, epochs = [], []
    weights = _run(tmp_path / "baseline", losses, epochs)
    assert len(losses) == STEPS_PER_EPOCH * EPOCHS
    return losses, epochs, weights


def _same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_resume_continues_the_step_count_the_losses_and_the_weights(tmp_path, monkeypatch, capsys):
    base_losses, base_epochs, base_weights = _baseline(tmp_path, monkeypatch)
    slot_dir = tmp_path / "slot"

    # Attempt 1 dies in epoch 2, on its 14th step; step 12 is the newest checkpoint.
    _slot_env(monkeypatch, slot_dir, resume=False, attempt="job-1/a/1")
    first, first_epochs = [], []
    with pytest.raises(Crash):
        _run(tmp_path / "a1", first, first_epochs, crash_at=14)
    assert len(first) == 13
    marker = json.loads((slot_dir / "state.json").read_text())
    assert marker["status"] == "running"
    assert marker["step"] == 12 and marker["epoch"] == 2
    assert marker["attempt_id"] == "job-1/a/1"
    assert marker["resumed_from_step"] is None
    assert marker["pid"] == os.getpid() and marker["host"]
    assert [s for s, _ in Slot(slot_dir, False).checkpoints()] == [12, 9], "keeps the newest two"

    # Attempt 2 resumes.
    _slot_env(monkeypatch, slot_dir, resume=True, attempt="job-1/a/2")
    capsys.readouterr()
    second, second_epochs = [], []
    weights = _run(tmp_path / "a2", second, second_epochs)
    out = capsys.readouterr().out
    assert "resumed from step-0000000012.pt: step 12/20, epoch 2, batch 2/10" in out
    assert len(second) == 8, "steps 13..20 only"
    assert first[:12] + second == base_losses, "per-step losses continue, not restart"
    assert first_epochs + second_epochs == base_epochs, "epoch losses match"
    _same_weights(weights, base_weights)
    marker = json.loads((slot_dir / "state.json").read_text())
    assert marker["resumed_from_step"] == 12
    assert marker["step"] == 20, "a final checkpoint is recorded"
    assert marker["attempt_id"] == "job-1/a/2"


def test_a_corrupt_newest_checkpoint_falls_back_to_the_previous_one(tmp_path, monkeypatch, capsys):
    base_losses, _, base_weights = _baseline(tmp_path, monkeypatch)
    slot_dir = tmp_path / "slot"
    _slot_env(monkeypatch, slot_dir, resume=False, attempt="job-1/a/1")
    first = []
    with pytest.raises(Crash):
        _run(tmp_path / "a1", first, [], crash_at=14)
    newest = slot_dir / "step-0000000012.pt"
    newest.write_bytes(newest.read_bytes()[: newest.stat().st_size // 2])  # torn write

    _slot_env(monkeypatch, slot_dir, resume=True, attempt="job-1/a/2")
    capsys.readouterr()
    second = []
    weights = _run(tmp_path / "a2", second, [])
    captured = capsys.readouterr()
    assert "skipping checkpoint step-0000000012.pt" in captured.err
    assert "resumed from step-0000000009.pt" in captured.out
    assert first[:9] + second == base_losses
    _same_weights(weights, base_weights)


def test_resume_with_no_valid_checkpoint_starts_at_step_zero(tmp_path, monkeypatch, capsys):
    base_losses, _, _ = _baseline(tmp_path, monkeypatch)
    slot_dir = tmp_path / "slot"
    slot_dir.mkdir()
    (slot_dir / "step-0000000006.pt").write_bytes(b"not a checkpoint")
    _slot_env(monkeypatch, slot_dir, resume=True, attempt="job-1/a/1")
    losses = []
    _run(tmp_path / "a1", losses, [])
    assert "no valid checkpoint to resume from" in capsys.readouterr().out
    assert losses == base_losses


def test_a_fresh_attempt_discards_old_checkpoints(tmp_path, monkeypatch):
    slot_dir = tmp_path / "slot"
    slot_dir.mkdir()
    (slot_dir / "step-0000000099.pt").write_bytes(b"stale")
    (slot_dir / "step-0000000099.pt.sha256").write_text("x\n")
    _slot_env(monkeypatch, slot_dir, resume=False, attempt="job-1/a/1")
    _run(tmp_path / "a1", [], [])
    assert not (slot_dir / "step-0000000099.pt").exists()
    assert [s for s, _ in Slot(slot_dir, False).checkpoints()] == [20, 18]


def test_without_a_slot_the_trainer_writes_no_checkpoints(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_CHECKPOINT_DIR, raising=False)
    _run(tmp_path / "out", [], [])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["model.pt"]


def _payload(step, fingerprint="fp", world=1):
    return {"model": {"w": torch.ones(2) * step}, "opt": {}, "config": {}, "step": step,
            "rng": {"torch": torch.get_rng_state()}, "fingerprint": fingerprint,
            "world_size": world, "epoch": 1, "batch_in_epoch": step}


def test_load_latest_skips_partial_corrupt_and_foreign_checkpoints(tmp_path, capsys):
    slot = Slot(tmp_path, resume=True, keep=10)
    for step in (3, 6, 9, 12, 15):
        ckpt.save(_payload(step), slot.path_for(step))
    ckpt.save(_payload(18, fingerprint="other-config"), slot.path_for(18))
    ckpt.save(_payload(21, world=2), slot.path_for(21))
    # 15: sidecar missing (killed between the rename and the sidecar write).
    slot.path_for(15).with_suffix(".pt.sha256").unlink()
    # 12: bytes changed after the sidecar was written.
    raw = bytearray(slot.path_for(12).read_bytes())
    raw[-1] ^= 0xFF
    slot.path_for(12).write_bytes(bytes(raw))
    # A temp file from a killed save is not a checkpoint.
    (tmp_path / "step-0000000024.abc.pt.tmp").write_bytes(b"partial")

    payload, path = slot.load_latest("fp", world_size=1)
    assert path == slot.path_for(9) and payload["step"] == 9
    err = capsys.readouterr().err
    for reason in ("step-0000000021.pt: written by 2 ranks",
                   "step-0000000018.pt: written for another config",
                   "step-0000000015.pt: no .sha256 sidecar",
                   "step-0000000012.pt: checkpoint SHA mismatch"):
        assert reason in err, reason
    slot.remove_temp_files()
    assert not (tmp_path / "step-0000000024.abc.pt.tmp").exists()


def test_fingerprint_ignores_paths_but_not_settings():
    a = {"dataset_path": "/jobs/1/d.jsonl", "output_dir": "/jobs/1/out", "lr": 1e-3}
    b = {"dataset_path": "/jobs/2/d.jsonl", "output_dir": "/jobs/2/out", "lr": 1e-3}
    c = dict(a, lr=2e-3)
    assert config_fingerprint(a) == config_fingerprint(b)
    assert config_fingerprint(a) != config_fingerprint(c)


def test_from_env_reads_the_contract(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_CHECKPOINT_DIR, raising=False)
    assert Slot.from_env() is None
    env = {ENV_CHECKPOINT_DIR: str(tmp_path), ENV_RESUME: "1", ENV_RUN_ID: "r",
           ENV_ATTEMPT_ID: "r/i/2"}
    s = Slot.from_env(env)
    assert s.dir == tmp_path and s.resume and s.run_id == "r" and s.attempt_id == "r/i/2"
    with pytest.raises(ValueError):
        Slot.from_env(dict(env, **{ENV_RESUME: "yes"}))


# --- DDP over gloo, two processes ---------------------------------------------

_WORKER = Path(__file__).with_name("ckpt_ddp_worker.py")


def _ddp(tmp_path, tag, crash_at, slot_dir=None, resume=False):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(trainer.__file__).resolve().parents[1])
    env.pop(ENV_CHECKPOINT_DIR, None)
    if slot_dir is not None:
        env.update({ENV_CHECKPOINT_DIR: str(slot_dir), ENV_RESUME: "1" if resume else "0",
                    ENV_RUN_ID: "job-ddp", ENV_ATTEMPT_ID: f"job-ddp/{tag}"})
    out = tmp_path / tag
    out.mkdir()
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", "2", str(_WORKER), str(out), str(crash_at)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    losses = json.loads((out / "losses.json").read_text()) if (out / "losses.json").exists() else None
    return proc, losses, out


@pytest.mark.skipif(not torch.distributed.is_available(), reason="no torch.distributed")
def test_ddp_resume_matches_an_uninterrupted_run(tmp_path):
    proc, base, base_out = _ddp(tmp_path, "base", 0)
    assert proc.returncode == 0, proc.stderr[-3000:]
    slot_dir = tmp_path / "slot"
    proc, first, _ = _ddp(tmp_path, "a1", 4, slot_dir)
    assert proc.returncode != 0, "the first attempt crashes"
    proc, second, out = _ddp(tmp_path, "a2", 0, slot_dir, resume=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "resumed from step-0000000003.pt: step 3/10" in proc.stdout, proc.stdout + proc.stderr[-2000:]
    assert len(base) == 10 and len(second) == 7
    assert first[:3] + second == base, "rank 0's losses continue across the resume"
    _same_weights(torch.load(out / "model.pt", weights_only=True)["model"],
                  torch.load(base_out / "model.pt", weights_only=True)["model"])
