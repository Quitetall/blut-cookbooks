"""Unit tests for the FSDP/DDP parallel-strategy dispatch in blut_core.trainer.

These run on CPU with no torchrun / NCCL: `_parallel_strategy` is pure config
validation, and the wrap-dispatch is checked by monkeypatching the torch
distributed entry points so no real sharding/process-group is needed.
"""
import pytest

import blut_core.trainer as trainer


# --- _parallel_strategy: pure config validation -------------------------------

def test_parallel_strategy_defaults_to_ddp():
    assert trainer._parallel_strategy({}) == "ddp"


def test_parallel_strategy_reads_fsdp():
    assert trainer._parallel_strategy({"parallel_strategy": "fsdp"}) == "fsdp"


def test_parallel_strategy_is_case_insensitive():
    assert trainer._parallel_strategy({"parallel_strategy": "FSDP"}) == "fsdp"


def test_parallel_strategy_rejects_unknown():
    with pytest.raises(ValueError, match="unknown parallel_strategy"):
        trainer._parallel_strategy({"parallel_strategy": "zero3"})


# --- wrap dispatch: the right backend is invoked per strategy ------------------

def test_fsdp_wrap_calls_fully_shard(monkeypatch):
    """`_fsdp_wrap_model` must route through torch's fully_shard, not DDP."""
    calls = {"fully_shard": 0}

    class _FakeModule:
        def to(self, _dev):
            return self

    def _fake_fully_shard(model, **kwargs):
        calls["fully_shard"] += 1
        # bf16 mixed-precision policy must be threaded through.
        assert "mp_policy" in kwargs
        return model

    class _FakeMPPolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    import torch.distributed.fsdp as fsdp_mod
    monkeypatch.setattr(fsdp_mod, "fully_shard", _fake_fully_shard, raising=False)
    monkeypatch.setattr(fsdp_mod, "MixedPrecisionPolicy", _FakeMPPolicy, raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")

    out = trainer._fsdp_wrap_model(_FakeModule())
    assert calls["fully_shard"] == 1
    assert out is not None


def test_ddp_unwrap_is_identity_for_plain_model():
    """FSDP models have no `.module`; unwrap must return them unchanged."""
    sentinel = object()
    assert trainer._ddp_unwrap(sentinel) is sentinel


# --- _ddp_init: the process group comes up and the rank's device is selected --

def test_ddp_init_selects_the_local_rank_device(monkeypatch):
    """Every distributed run went through `_ddp_init`, which raised
    NameError: name 'torch' is not defined. Observed on a two-node Slurm run;
    nothing here had ever executed it."""
    import torch
    import torch.distributed as dist

    calls = {}
    monkeypatch.setattr(dist, "init_process_group", lambda backend: calls.update(backend=backend))
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.update(device=i))
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert trainer._ddp_init() == "cuda:1"
    assert calls == {"backend": "nccl", "device": 1}


def test_cpu_ranks_use_gloo_and_never_touch_cuda(monkeypatch):
    """`device: cpu` under DDP used to call init_process_group("nccl") and
    torch.cuda.set_device, so CPU-only multi-node training could not start."""
    import torch
    import torch.distributed as dist

    calls = {}
    monkeypatch.setattr(dist, "init_process_group", lambda backend: calls.update(backend=backend))
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.update(device=i))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert trainer._ddp_init("cpu") == "cpu"
    assert calls == {"backend": "gloo"}


def test_gpu_request_without_cuda_is_an_error_not_a_cpu_fallback(monkeypatch):
    import pytest
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device: cpu"):
        trainer._ddp_device("cuda")


def test_params_hash_changes_with_any_parameter_and_handles_bf16():
    import torch

    model = torch.nn.Linear(4, 3)
    first = trainer._params_sha256(model)
    assert first == trainer._params_sha256(model)
    with torch.no_grad():
        model.bias[0] += 1e-6
    assert trainer._params_sha256(model) != first
    assert len(trainer._params_sha256(model.to(torch.bfloat16))) == 64


def test_diverged_replicas_fail_the_run(monkeypatch):
    import pytest
    import torch
    import torch.distributed as dist

    def gather(out, mine):
        out[0], out[1] = mine, "f" * 64  # rank 1 holds different weights

    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(RuntimeError, match="replicas diverged"):
        trainer._check_replicas_agree(torch.nn.Linear(2, 2))
