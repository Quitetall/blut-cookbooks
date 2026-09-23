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
