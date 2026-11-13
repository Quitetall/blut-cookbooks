"""The LR each epoch trains at, from its first step.

The schedule tests need no torch: a scheduler only writes `param_groups`. The
last test runs the real training loop, which does.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # blut/python
from blut_core import build_ingredient  # noqa: E402


def _lrs(name, epochs, peak=1.0, **config):
    """Each epoch's LR, set as the trainer sets it: `step(epoch)` before it trains."""
    opt = types.SimpleNamespace(param_groups=[{"lr": peak}])
    sched = build_ingredient("scheduler", name, config, optimizer=opt)
    out = []
    for epoch in range(1, epochs + 1):
        sched.step(epoch)
        out.append(opt.param_groups[0]["lr"])
    return out


def test_constant_ramps_from_the_first_epoch_and_holds_the_peak():
    assert _lrs("constant", 6, warmup_epochs=4) == pytest.approx(
        [0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


def test_warmup_starts_from_its_start_lr_and_the_first_epoch_still_trains():
    assert _lrs("constant", 2, warmup_epochs=2, warmup_start_lr=0.5) == pytest.approx(
        [0.75, 1.0])
    # A one-epoch run with the defaults must not train at LR 0.
    assert _lrs("constant", 1) == pytest.approx([0.2])


def test_no_warmup_is_the_peak_from_the_first_epoch():
    assert _lrs("constant", 2, warmup_epochs=0) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("name", ["cosine", "linear"])
def test_decay_starts_at_the_peak_and_the_last_epoch_still_trains(name):
    lrs = _lrs(name, 6, warmup_epochs=2, total_epochs=6)
    assert lrs[:3] == pytest.approx([0.5, 1.0, 1.0])
    assert lrs[2] > lrs[3] > lrs[4] > lrs[5] > 0.0


def test_linear_decay_values():
    assert _lrs("linear", 5, warmup_epochs=1, total_epochs=5) == pytest.approx(
        [1.0, 1.0, 0.75, 0.5, 0.25])


def test_the_training_loop_trains_each_epoch_at_its_scheduled_lr(tmp_path):
    # The ordering lives in the loop, so this one needs torch (CI installs it).
    torch = pytest.importorskip("torch")
    import blut_core.trainer as trainer

    seen = []

    def recording_step(model, optimizer, loss_fn, batch, **kw):
        seen.append(optimizer.param_groups[0]["lr"])
        return torch.tensor(1.0)

    config = {
        "dataset_path": "unused",
        "output_dir": str(tmp_path / "out"),
        "optimizer": {"kind": "optimizer", "name": "adamw", "config": {"lr": 1.0}},
        "scheduler": {"kind": "scheduler", "name": "constant",
                      "config": {"warmup_epochs": 2}},
        "epochs": 3,
        "batch_size": 2,
        "device": "cpu",
    }
    ingredients = {
        "model": torch.nn.Linear(2, 1),
        "loss": None,
        "step": recording_step,
        "logging": lambda *a, **k: None,
        "checkpoint": lambda payload, path: torch.save(payload, path),
    }
    trainer.train_loop(config, ingredients, [{"x": i} for i in range(4)])  # 2 steps per epoch
    assert seen == pytest.approx([0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
