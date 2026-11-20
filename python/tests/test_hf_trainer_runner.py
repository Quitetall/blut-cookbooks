"""The HF trainer wrapper's status stream and distributed defaults.

The wrapper ships inside the Rust crate (src/backends/hf_trainer/python) and is
loaded here by path. Nothing in these tests imports transformers or torch at
module scope, so they run without a training stack.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_WRAPPER = (Path(__file__).resolve().parents[2]
            / "src/backends/hf_trainer/python/hf_trainer_runner.py")


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("hf_trainer_runner", _WRAPPER)
    assert spec is not None and spec.loader is not None, f"cannot load {_WRAPPER}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lines(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.splitlines() if l]


def test_only_the_local_leader_reports_progress(runner, monkeypatch, capsys):
    # Under torchrun every rank shares the node's stdout, which the Rust
    # runner parses. Unguarded, N ranks wrote N copies of every line.
    monkeypatch.setenv("LOCAL_RANK", "1")
    runner.emit({"kind": "step", "step": 1, "total": 2})
    runner.emit({"kind": "done", "checkpoint_dir": "/x", "final_loss": 1.0})
    assert _lines(capsys) == []

    monkeypatch.setenv("LOCAL_RANK", "0")
    runner.emit({"kind": "done", "checkpoint_dir": "/x", "final_loss": 1.0})
    assert [l["kind"] for l in _lines(capsys)] == ["done"]


def test_the_leader_is_local_rank_zero_not_global_rank_zero(runner, monkeypatch, capsys):
    # Node 1's leader has RANK=2 on a 2x2 job. Each node's runner reads only
    # its own stdout; if only global rank 0 reported, node 1 would never see
    # a done line and its stage would fail.
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    runner.emit({"kind": "done", "checkpoint_dir": "/x", "final_loss": 1.0})
    assert len(_lines(capsys)) == 1


def test_any_rank_reports_its_own_failure(runner, monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_RANK", "3")
    runner.emit({"kind": "failed", "error": "CUDA out of memory"})
    assert _lines(capsys) == [{"kind": "failed", "error": "CUDA out of memory"}]


def test_a_multi_node_job_saves_on_every_node(runner):
    # No shared filesystem is assumed: each node's stage needs its own copy.
    assert runner._distributed_defaults({"nnodes": 2}) == {"save_on_each_node": True}
    assert runner._distributed_defaults({"nnodes": 1}) == {}
    assert runner._distributed_defaults({}) == {}


def test_vram_is_zero_off_the_gpu_and_measured_on_it(runner, monkeypatch):
    # The Rust side relays each step as the engine's trainer update, whose
    # vram field `blut log` prints; it must be a measurement, never a guess.
    import sys
    import types

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert runner._vram_mb() == 0

    cuda = types.SimpleNamespace(is_available=lambda: True,
                                 memory_allocated=lambda: 3 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    assert runner._vram_mb() == 3

    cuda.is_available = lambda: False
    assert runner._vram_mb() == 0


@pytest.mark.parametrize("metric", ["loss", "eval_loss"])
@pytest.mark.parametrize("value, token", [
    (float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")])
def test_a_non_finite_loss_is_a_diverged_line(runner, metric, value, token):
    lines = runner._LogLines()
    lines.total = 10
    out = lines.lines({metric: value, "learning_rate": 1e-4}, 5)
    assert out == [{"kind": "diverged", "step": 5, "total": 10,
                    "metric": metric, "value": token}]
    json.dumps(out, allow_nan=False)  # valid JSON, unlike a bare NaN
    assert lines.last_loss is None


def test_a_log_event_becomes_its_status_lines(runner, monkeypatch):
    # HF logs training, evaluation and the end-of-run summary through one
    # callback. An evaluation is the engine's `eval` update; the others are
    # steps, with the latest training loss and, from a training log, the
    # learning rate. Nothing HF did not log is filled in.
    monkeypatch.setattr(runner, "_vram_mb", lambda: 7)
    lines = runner._LogLines()
    lines.total = 10
    assert lines.lines({"eval_loss": 1.5, "eval_runtime": 0.2}, 0) == [
        {"kind": "eval", "step": 0, "eval_loss": 1.5}]
    assert lines.lines({"loss": 2.0, "learning_rate": 1e-4, "epoch": 0.5}, 5) == [
        {"kind": "step", "step": 5, "total": 10, "loss": 2.0, "lr": 1e-4, "vram_mb": 7}]
    assert lines.lines({"eval_loss": 1.25}, 5) == [
        {"kind": "eval", "step": 5, "eval_loss": 1.25}]
    assert lines.lines({"train_runtime": 3.0, "train_loss": 1.9}, 10) == [
        {"kind": "step", "step": 10, "total": 10, "loss": 2.0, "lr": None, "vram_mb": 7}]
