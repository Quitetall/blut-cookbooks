from __future__ import annotations

import copy

import pytest

from blut_core.ingredients.scheduler.wsd import WSDScheduler


class _Optimizer:
    def __init__(self, *rates: float):
        self.param_groups = [{"lr": rate} for rate in rates]


def _scheduler(optimizer: _Optimizer, *, peak_lr: float = 1e-3) -> WSDScheduler:
    return WSDScheduler(
        optimizer,
        total_epochs=100,
        peak_lr=peak_lr,
        warmup_frac=0.1,
        decay_frac=0,
        min_lr=1e-5,
    )


def test_state_round_trip_preserves_manual_decay_trajectory():
    original_optimizer = _Optimizer(1e-3, 5e-4)
    original = _scheduler(original_optimizer)
    original.step(25)
    original.trigger_decay(20)
    original.step(31)

    state = copy.deepcopy(original.state_dict())
    resumed_optimizer = _Optimizer(1e-3, 5e-4)
    resumed = _scheduler(resumed_optimizer)
    resumed.load_state_dict(state)

    assert resumed.epoch == original.epoch
    assert resumed.phase == original.phase
    assert resumed.get_last_lr() == original.get_last_lr()
    assert [g["lr"] for g in resumed_optimizer.param_groups] == original.get_last_lr()

    original.step()
    resumed.step()
    assert resumed.get_last_lr() == pytest.approx(original.get_last_lr())


def test_load_rejects_incompatible_configuration():
    state = _scheduler(_Optimizer(1e-3)).state_dict()
    incompatible = _scheduler(_Optimizer(2e-3), peak_lr=2e-3)
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.load_state_dict(state)


def test_state_round_trip_preserves_automatic_decay_trajectory():
    original_optimizer = _Optimizer(1e-3)
    original = WSDScheduler(
        original_optimizer,
        total_epochs=100,
        peak_lr=1e-3,
        warmup_frac=0.1,
        decay_frac=0.2,
        min_lr=1e-5,
    )
    original.step(85)

    resumed_optimizer = _Optimizer(1e-3)
    resumed = WSDScheduler(
        resumed_optimizer,
        total_epochs=100,
        peak_lr=1e-3,
        warmup_frac=0.1,
        decay_frac=0.2,
        min_lr=1e-5,
    )
    resumed.load_state_dict(copy.deepcopy(original.state_dict()))

    assert resumed.phase == "decay"
    assert resumed.get_last_lr() == original.get_last_lr()
    original.step()
    resumed.step()
    assert resumed.get_last_lr() == pytest.approx(original.get_last_lr())


def test_load_rejects_optimizer_group_mismatch():
    state = _scheduler(_Optimizer(1e-3)).state_dict()
    with pytest.raises(ValueError, match="optimizer groups"):
        _scheduler(_Optimizer(1e-3, 5e-4)).load_state_dict(state)


def test_load_rejects_unknown_state_version():
    scheduler = _scheduler(_Optimizer(1e-3))
    state = scheduler.state_dict()
    state["version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        scheduler.load_state_dict(state)
