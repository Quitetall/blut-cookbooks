"""The generic trainer reports its progress in the engine's update format.

`train_model` forwards every ``BLUT_STATUS <json>`` stdout line to the engine
as the stage's progress (what `blut log` and `blut jobs` read) and leaves the
``[trainer]`` lines to the terminal. These tests pin what the trainer sends:
each logged step with the loss, learning rate and GPU memory it measured, a
save, the epoch, and the final loss, and never a value it did not measure.
"""
import json
import math
import sys

import pytest

from blut_core import status


def _relayed(text):
    return [json.loads(line[len(status.RELAY_PREFIX):])
            for line in text.splitlines() if line.startswith(status.RELAY_PREFIX)]


# --- the channel -----------------------------------------------------------------

def test_relay_writes_one_marked_json_line(capsys):
    line = status.relay("step", step=1, total=2, loss=0.5, lr=1e-3, vram_mb=0)
    out = capsys.readouterr().out
    assert out == line + "\n"
    assert _relayed(out) == [
        {"kind": "step", "step": 1, "total": 2, "loss": 0.5, "lr": 1e-3, "vram_mb": 0}]


def test_relay_drops_what_json_cannot_carry_and_never_raises(capsys):
    # The engine's update has no NaN; a number standing in for one would be a lie.
    assert status.relay("done", final_loss=math.nan, checkpoint_dir="/x") is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not sent" in captured.err


def test_relay_refuses_a_kind_the_engine_does_not_know():
    with pytest.raises(ValueError):
        status.relay("epoch", loss=1.0)


# --- the trainer -----------------------------------------------------------------

torch = pytest.importorskip("torch")

import blut_core.trainer as trainer  # noqa: E402


def _config(tmp_path, **extra):
    return {
        "dataset_path": "unused",
        "output_dir": str(tmp_path / "out"),
        "optimizer": {"kind": "optimizer", "name": "adamw", "config": {"lr": 1e-3}},
        "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu",
        **extra,
    }


def _ingredients(losses):
    it = iter(losses)

    def fixed_step(model, optimizer, loss_fn, batch, **kw):
        return torch.tensor(next(it))

    return {
        "model": torch.nn.Linear(2, 1),
        "loss": None,
        "step": fixed_step,
        "logging": lambda *a, **k: None,
        "checkpoint": lambda payload, path: torch.save(payload, path),
    }


def test_a_run_reports_its_steps_save_epoch_and_final_loss(tmp_path, capsys):
    rows = [{"x": i} for i in range(4)]  # batch_size 2 -> two steps
    trainer.train_loop(_config(tmp_path, log_every_steps=1), _ingredients([1.0, 3.0]), rows)
    out = capsys.readouterr().out
    model_pt = str(tmp_path / "out" / "model.pt")
    assert _relayed(out) == [
        {"kind": "step", "step": 1, "total": 2, "loss": 1.0, "lr": 1e-3, "vram_mb": 0},
        {"kind": "step", "step": 2, "total": 2, "loss": 3.0, "lr": 1e-3, "vram_mb": 0},
        {"kind": "heartbeat", "phase": "epoch 1/1 loss=2.0000"},
        {"kind": "saved", "path": model_pt},
        {"kind": "done", "final_loss": 2.0, "checkpoint_dir": str(tmp_path / "out")},
    ]
    # The terminal still reads as it did.
    assert "[trainer] step 2/2 loss=3.0000" in out
    assert "[trainer] epoch 1/1: loss=2.0000" in out


def test_without_step_logging_only_the_epoch_and_the_end_are_reported(tmp_path, capsys):
    rows = [{"x": i} for i in range(4)]
    trainer.train_loop(_config(tmp_path), _ingredients([1.0, 3.0]), rows)
    kinds = [u["kind"] for u in _relayed(capsys.readouterr().out)]
    assert kinds == ["heartbeat", "saved", "done"]


def test_an_unmeasured_value_is_never_filled_in(tmp_path, capsys):
    # A NaN step loss is still progress, but not a step the engine can carry;
    # an epoch whose mean is NaN has no final loss to report.
    rows = [{"x": i} for i in range(4)]
    trainer.train_loop(
        _config(tmp_path, log_every_steps=1), _ingredients([1.0, math.nan]), rows)
    updates = _relayed(capsys.readouterr().out)
    assert updates[1] == {"kind": "heartbeat", "phase": "step 2/2 loss=nan", "vram_mb": 0}
    assert [u["kind"] for u in updates] == ["step", "heartbeat", "heartbeat", "saved"]


def test_an_optimizer_without_a_learning_rate_reports_a_heartbeat(capsys):
    trainer._report_step(3, 8, 0.25, None, 0)
    assert _relayed(capsys.readouterr().out) == [
        {"kind": "heartbeat", "phase": "step 3/8 loss=0.25", "vram_mb": 0}]


def test_a_failure_is_reported_and_still_raised(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["trainer", "--config", str(tmp_path / "absent.json")])
    with pytest.raises(Exception):
        trainer.main()
    updates = _relayed(capsys.readouterr().out)
    assert [u["kind"] for u in updates] == ["failed"]
    assert "absent.json" in updates[0]["error"]
