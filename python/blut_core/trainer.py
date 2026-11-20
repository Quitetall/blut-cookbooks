"""Generic ingredient-based trainer for BLUT core cookbook.

Reads an ingredient config JSON (written by the TrainModel Rust stage),
builds all ingredients via the registry, and runs a training loop.

Supports DDP: when launched via torchrun (WORLD_SIZE > 1), automatically
initializes process group, DDP-wraps the model, shards data, and restricts
saves/metrics to rank 0.

Usage:
    # Single GPU
    python -m blut_core.trainer --config train_config.json

    # Multi-GPU DDP (launched by the TrainModel stage via torchrun)
    torchrun --nproc_per_node=N -m blut_core.trainer --config train_config.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure blut_core is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from blut_core import build_ingredient, list_ingredients, status
from blut_core.lm_data import is_causal_lm, pack_token_blocks, texts_from_rows
from blut_core.spec import KINDS


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_ddp() -> bool:
    """True when launched under torchrun (WORLD_SIZE > 1)."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0() -> bool:
    return _ddp_rank() == 0


# ---------------------------------------------------------------------------
# Progress for the launching stage
# ---------------------------------------------------------------------------
#
# `train_model` reads this process's stdout and forwards `BLUT_STATUS` lines
# (``status.relay``) to the engine as the stage's progress, which is what
# `blut log` and `blut jobs` show; the `[trainer]` lines stay the terminal's.
# Rank 0 reports progress; a failure is reported by the rank that hit it.

def _current_lr(optimizer) -> "float | None":
    """The learning rate the first parameter group steps with now, or None
    when the optimizer has none to read."""
    try:
        return float(optimizer.param_groups[0]["lr"])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def _vram_mb(device) -> int:
    """GPU memory this process holds on ``device``, in MiB; 0 off the GPU."""
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated(device) / (1024 * 1024))


def _report_step(step: int, total: int, loss: float, lr, vram_mb: int) -> None:
    """One logged step. The engine's step update needs a finite loss and a
    learning rate; a step without them is reported as a heartbeat that names
    it, never with a value filled in.

    A non-finite loss also goes as the string ``loss: "nan"`` (or ``"inf"``,
    ``"-inf"``). JSON has no such number, and the string is how the engine
    takes one: ``KillOnNaN`` and the HPO scheduler kill a diverged trial on it,
    and ``blut log`` still reads the update as a heartbeat."""
    phase = f"step {step}/{total} loss={loss}"
    if not math.isfinite(loss):
        status.relay("heartbeat", phase=phase, vram_mb=vram_mb, loss=str(loss))
    elif lr is not None and math.isfinite(lr):
        status.relay("step", step=step, total=total, loss=loss, lr=lr, vram_mb=vram_mb)
    else:
        status.relay("heartbeat", phase=phase, vram_mb=vram_mb)


def _report_epoch(epoch: int, epochs: int, loss: float) -> None:
    """One finished epoch, its mean loss in the label. A non-finite mean goes
    as in ``_report_step``: with no step logging this is where a diverged run
    is first seen."""
    phase = f"epoch {epoch}/{epochs} loss={loss:.4f}"
    if math.isfinite(loss):
        status.relay("heartbeat", phase=phase)
    else:
        status.relay("heartbeat", phase=phase, loss=str(loss))


def _ddp_device(device: str) -> str:
    """The device this rank trains on: its local GPU, or the CPU when the
    config asks for `device: cpu`. A GPU request on a host without CUDA is an
    error, not a silent move to the CPU."""
    import torch
    if device == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} needs CUDA, which is not available on this host; "
            "set `device: cpu` to train on the CPU (gloo backend)")
    return f"cuda:{_ddp_local_rank()}"


def _ddp_init(device: str = "cuda"):
    """Initialize the process group: NCCL for GPU ranks, gloo for CPU ranks
    (NCCL has no CPU transport)."""
    # `import torch.distributed as dist` binds only `dist`; `torch` itself
    # must be imported here too, since this module imports it lazily.
    import torch
    import torch.distributed as dist
    rank_device = _ddp_device(device)
    on_gpu = rank_device.startswith("cuda")
    dist.init_process_group(backend="nccl" if on_gpu else "gloo")
    local_rank = _ddp_local_rank()
    if on_gpu:
        torch.cuda.set_device(local_rank)
    print(f"[trainer] DDP initialized: rank={_ddp_rank()}, "
          f"local_rank={local_rank}, world_size={_ddp_world_size()}, "
          f"device={rank_device}")
    return rank_device


def _ddp_cleanup():
    """Destroy the DDP process group."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _ddp_wrap_model(model, rank_device: str = None):
    """Wrap model with DistributedDataParallel on this rank's device."""
    import torch.nn.parallel as parallel
    local_rank = _ddp_local_rank()
    rank_device = rank_device or f"cuda:{local_rank}"
    model = model.to(rank_device)
    if rank_device.startswith("cuda"):
        return parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    return parallel.DistributedDataParallel(model)


def _ddp_shard_dataset(dataset, batch_size: int, seed: int, dl_kwargs: dict = None):
    """Create a DistributedSampler and DataLoader for DDP. `dl_kwargs` carries
    the async knobs (num_workers/prefetch_factor/pin_memory/persistent_workers);
    when omitted, defaults to the old `num_workers=0, pin_memory=True`."""
    from torch.utils.data import DataLoader, DistributedSampler
    if dl_kwargs is None:
        dl_kwargs = {"num_workers": 0, "pin_memory": True}
    sampler = DistributedSampler(dataset, seed=seed, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, **dl_kwargs)
    return sampler, loader


def _params_sha256(model) -> str:
    """sha256 over every parameter and buffer, in name order, as raw bytes."""
    import hashlib
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        # Reinterpret as bytes first: numpy has no bfloat16.
        raw = tensor.detach().to("cpu").contiguous().view(-1).view(torch.uint8)
        h.update(raw.numpy().tobytes())
    return h.hexdigest()


def _check_replicas_agree(model) -> str:
    """DDP keeps one replica per rank and they must stay bit-identical. Every
    rank hashes its parameters and rank 0 refuses to save if any differ, so a
    replica that silently diverged (a skipped all-reduce, a rank that trained
    on different data) fails the run instead of shipping one rank's weights."""
    import torch.distributed as dist
    mine = _params_sha256(model)
    hashes = [None] * _ddp_world_size()
    dist.all_gather_object(hashes, mine)
    if len(set(hashes)) != 1:
        raise RuntimeError(f"DDP replicas diverged; parameter sha256 per rank: {hashes}")
    if _is_rank0():
        print(f"[trainer] replicas agree across {len(hashes)} ranks: params sha256 {mine}")
        return {"params_sha256": mine, "ranks": len(hashes)}
    return None


def _ddp_unwrap(model):
    """Unwrap DDP wrapper to get the base model."""
    return model.module if hasattr(model, 'module') else model


# ---------------------------------------------------------------------------
# FSDP helpers (FSDP2 / fully_shard — torch >= 2.5)
# ---------------------------------------------------------------------------
#
# FSDP2 shards parameters, gradients, and optimizer state across ranks
# (ZeRO-3), trading NCCL traffic for memory so a model larger than one GPU's
# VRAM can train. Unlike DDP it does NOT wrap the module in a `.module`
# container — `fully_shard` mutates the module in place and registers its
# sharded params, so unwrap is the identity and the saved state-dict must be
# GATHERED via torch.distributed.checkpoint (DCP) rather than read directly.

def _fsdp_wrap_model(model):
    """Shard the model in place with FSDP2 fully_shard + bf16 mixed precision.

    bf16 matches the LamQuant convention (stable on Ada/Hopper, no GradScaler).
    `fully_shard` returns the same (now sharded) module object.
    """
    import torch
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    local_rank = _ddp_local_rank()
    model = model.to(f"cuda:{local_rank}")
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model, mp_policy=mp)
    return model


def _fsdp_full_state_dict(model):
    """Gather a full (unsharded), CPU state-dict from an FSDP2 model via the DCP
    state-dict API. This is a COLLECTIVE — every rank must call it — and with
    `full_state_dict=True` each rank receives the full dict; callers then write
    on rank 0 only (the existing `_is_rank0()` guard)."""
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        StateDictOptions,
    )
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def _parallel_strategy(config: dict) -> str:
    """Read the parallel strategy from the config. 'ddp' (default) or 'fsdp'.
    Only meaningful under torchrun (WORLD_SIZE > 1)."""
    strat = str(config.get("parallel_strategy", "ddp")).lower()
    if strat not in ("ddp", "fsdp"):
        raise ValueError(
            f"unknown parallel_strategy '{strat}' (expected 'ddp' or 'fsdp')")
    return strat


# ---------------------------------------------------------------------------
# Async compute (all opt-in via config; defaults reproduce the old behaviour)
# ---------------------------------------------------------------------------
#
# The GPU stalls on three things: waiting for the next batch (decode/transfer),
# the synchronous checkpoint write, and inline eval. These helpers overlap the
# first two with compute. Each is gated by a config flag and defaults OFF, so an
# unchanged config trains byte-for-byte as before.
#
# Async EVAL is intentionally NOT here: this generic trainer has no separate
# eval phase (it emits one epoch metric inline), so there is nothing to overlap.
# The valuable async-eval target is train_joint.py's `validate_joint`, which is
# GPU-bound — truly overlapping it needs a weight snapshot + a second CUDA
# stream, a larger change deferred to that trainer (see ADR 0065 follow-ups).

def _dataloader_kwargs(config: dict) -> dict:
    """DataLoader knobs from config. Defaults reproduce the old naive loader
    (`num_workers=0`), so omitting them changes nothing. Set `num_workers > 0`
    to decode on worker processes; `prefetch_factor`/`persistent_workers` then
    overlap decode with the model step and avoid per-epoch worker re-forks."""
    nw = int(config.get("num_workers", 0))
    kw = {"num_workers": nw, "pin_memory": bool(config.get("pin_memory", False))}
    if nw > 0:
        # prefetch_factor / persistent_workers are only valid with workers > 0.
        kw["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        kw["persistent_workers"] = bool(config.get("persistent_workers", True))
    return kw


class CudaPrefetcher:
    """Double-buffer the next batch onto a side CUDA stream while the model
    computes the current one. Hides the host→device copy behind compute.

    Opt-in (`config["cuda_prefetch"]`); the call site only constructs it on a
    CUDA device (`torch.cuda.is_available()`), so the constructor assumes CUDA.
    Handles dict / list / tuple batches of tensors (recursively). Wraps an
    existing DataLoader; the yielded batches are identical (same order, same
    values, now on-device) — only the transfer timing changes, so the trained
    result is unaffected."""

    def __init__(self, loader, device):
        import torch
        self._loader = loader
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._torch = torch

    def _to_device(self, batch):
        # Recurse through dict / list / tuple containers so a tuple batch
        # (e.g. (input_ids, labels) from a custom collate) is transferred too,
        # not silently left on CPU. Named tuples keep their type. `is_tensor`
        # (not duck-typed `.to`) so an nn.Module in a batch isn't moved wholesale.
        if self._torch.is_tensor(batch):
            return batch.to(self._device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            mapped = [self._to_device(v) for v in batch]
            if isinstance(batch, tuple):
                # Preserve namedtuple type (it takes positional args, not an iterable).
                return type(batch)(*mapped) if hasattr(batch, "_fields") else tuple(mapped)
            return mapped
        return batch

    def _record_stream(self, obj, stream):
        # `record_stream` on EVERY tensor (any container) keeps the allocator
        # from reusing the staging buffer before the default stream finishes
        # reading it — a use-after-free guard that must not skip tuple batches.
        if hasattr(obj, "record_stream"):
            obj.record_stream(stream)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._record_stream(v, stream)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._record_stream(v, stream)

    def __iter__(self):
        torch = self._torch
        it = iter(self._loader)
        try:
            nxt = next(it)
        except StopIteration:
            return
        with torch.cuda.stream(self._stream):
            nxt = self._to_device(nxt)
        while True:
            # Make the default stream wait for the staging copy, then hand off
            # the batch and stage the following one on the side stream.
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
            cur = nxt
            self._record_stream(cur, torch.cuda.current_stream(self._device))
            try:
                nxt = next(it)
            except StopIteration:
                yield cur
                return
            with torch.cuda.stream(self._stream):
                nxt = self._to_device(nxt)
            yield cur


class AsyncCheckpointSaver:
    """Offload `torch.save` off the training-loop thread. The state-dict is
    snapshotted to CPU on the loop thread (cheap), then written by a single
    background worker. At most one write is in flight: the next save joins the
    previous one first, so writes never overlap or race the file.

    Opt-in (`config["async_checkpoint"]`). `save_fn` is the same checkpoint
    ingredient used synchronously; only the thread it runs on changes."""

    def __init__(self, save_fn):
        from concurrent.futures import ThreadPoolExecutor
        self._save_fn = save_fn
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending = None

    @staticmethod
    def _to_cpu(obj):
        # Snapshot tensors to CPU so the background write sees a stable copy
        # the training loop can't mutate underneath it. Recurse through dict /
        # list / tuple so tensors nested in a list aren't left on the GPU
        # (where the loop could mutate them mid-write). `is_tensor` (not
        # duck-typed `.detach`) so only real tensors are copied.
        import torch
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: AsyncCheckpointSaver._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            mapped = [AsyncCheckpointSaver._to_cpu(v) for v in obj]
            if isinstance(obj, tuple):
                # Preserve namedtuple type (consistent with CudaPrefetcher._to_device).
                return type(obj)(*mapped) if hasattr(obj, "_fields") else tuple(mapped)
            return mapped
        return obj

    def save(self, payload, path):
        if self._pending is not None:
            # Bound to one in-flight write: join the previous before the next.
            self._pending.result()
        snapshot = self._to_cpu(payload)
        self._pending = self._pool.submit(self._save_fn, snapshot, path)

    def close(self):
        """Join any in-flight write and shut the worker down. The pool is shut
        down even if the join raises (a background save error), so the executor
        never leaks; the error is re-raised after shutdown."""
        try:
            if self._pending is not None:
                self._pending.result()  # may raise the background save's error
                self._pending = None
        finally:
            self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        # Always join + shut down the pool, even if a save raised — so the
        # ThreadPoolExecutor never leaks (this is the reusable primitive).
        self.close()
        return False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate the ingredient config JSON."""
    with open(config_path) as f:
        config = json.load(f)
    required = ["dataset_path", "output_dir", "model", "optimizer", "scheduler", "loss"]
    for key in required:
        if key not in config:
            raise ValueError(f"missing required key '{key}' in config")
    return config


# ---------------------------------------------------------------------------
# Ingredient building
# ---------------------------------------------------------------------------

def build_all_ingredients(config: dict) -> dict:
    """Build all ingredients from the config."""
    ingredients = {}

    # Build model
    model_cfg = config["model"]
    if _is_rank0():
        print(f"[trainer] building model: {model_cfg['kind']}:{model_cfg['name']}")
    ingredients["model"] = build_ingredient(
        model_cfg["kind"], model_cfg["name"], model_cfg.get("config"))

    # The optimizer and scheduler are NOT built here: they must be built from
    # the parameters the model has after it is placed and wrapped. See
    # build_optimization.

    # Build loss
    loss_cfg = config["loss"]
    if _is_rank0():
        print(f"[trainer] building loss: {loss_cfg['kind']}:{loss_cfg['name']}")
    ingredients["loss"] = build_ingredient(
        loss_cfg["kind"], loss_cfg["name"], loss_cfg.get("config"))

    # Build step
    step_cfg = config.get("step", {"kind": "step", "name": "standard", "config": {}})
    if _is_rank0():
        print(f"[trainer] building step: {step_cfg['kind']}:{step_cfg['name']}")
    ingredients["step"] = build_ingredient(
        step_cfg["kind"], step_cfg["name"], step_cfg.get("config"))

    # Build logging (rank-0 only for metrics emission)
    if _is_rank0():
        print("[trainer] building logging: logging:blut_metric")
    ingredients["logging"] = build_ingredient("logging", "blut_metric", {})

    # Build checkpoint (rank-0 only for saves)
    if _is_rank0():
        print("[trainer] building checkpoint: checkpoint:atomic_save")
    ingredients["checkpoint"] = build_ingredient("checkpoint", "atomic_save", {})

    return ingredients


def build_optimization(config: dict, model) -> tuple:
    """Build the optimizer and scheduler over `model`'s CURRENT parameters.

    Call this after the model is moved and wrapped. FSDP2's `fully_shard`
    replaces every parameter with a sharded DTensor, so an optimizer built
    beforehand (as this trainer used to) holds the old, unsharded tensors: its
    `step()` updates tensors the model no longer uses, and training silently
    changes nothing. DDP keeps the same parameter objects, but names them
    `module.<name>`; building from the unwrapped module keeps name-based
    parameter-group rules (no weight decay on biases, say) matching.
    """
    opt_cfg = config["optimizer"]
    if _is_rank0():
        print(f"[trainer] building optimizer: {opt_cfg['kind']}:{opt_cfg['name']}")
    base = _ddp_unwrap(model)
    named_params = list(base.named_parameters()) if hasattr(base, "named_parameters") else []
    optimizer = build_ingredient(
        opt_cfg["kind"], opt_cfg["name"], opt_cfg.get("config"),
        named_params=named_params)

    sched_cfg = config["scheduler"]
    if _is_rank0():
        print(f"[trainer] building scheduler: {sched_cfg['kind']}:{sched_cfg['name']}")
    scheduler = build_ingredient(
        sched_cfg["kind"], sched_cfg["name"], sched_cfg.get("config"),
        optimizer=optimizer)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(config: dict) -> list:
    """Read the dataset rows from `config["dataset_path"]`.

    Accepts JSONL (one object per line) or a JSON array. Every failure raises.
    This used to print a warning and return `[]` when the file could not be
    read, and the training loop then ran zero batches and reported success —
    a missing dataset produced a checkpoint.
    """
    path = Path(config["dataset_path"])
    if not path.is_file():
        raise FileNotFoundError(f"dataset_path is not a file: {path}")
    if path.suffix == ".json":
        with open(path) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array of rows")
    else:
        rows = []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: not valid JSON: {e}") from e
    if not rows:
        raise ValueError(f"{path} holds no rows; refusing to train on nothing")
    if _is_rank0():
        print(f"[trainer] loaded {len(rows)} rows from {path}")
    return rows


def _keep_rows(batch):
    """Collate that hands the step the rows unchanged, as a list.

    Module-level (not a lambda) so DataLoader worker processes can pickle it.
    """
    return batch


def _stack_blocks(batch):
    """Collate packed causal-LM blocks into `input_ids`/`labels` tensors.

    The labels are the inputs: a HuggingFace causal LM shifts them by one
    position internally when it computes its loss.
    """
    import torch
    ids = torch.tensor(batch, dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


def prepare_dataset(config: dict, rows: list, model):
    """Turn raw rows into what the step consumes. Returns (dataset, collate, tokenizer).

    For a HuggingFace causal LM, text rows are tokenized with the model's own
    tokenizer and packed into `max_seq_len` blocks (see `blut_core.lm_data`).
    Any other model gets the rows untouched, as before, and `tokenizer` is None.
    """
    if not is_causal_lm(model):
        return rows, _keep_rows, None

    from transformers import AutoTokenizer
    # The Rust stage writes these keys as JSON null when unset.
    text_field = config.get("text_field") or "text"
    block_size = int(config.get("max_seq_len") or 512)
    source = config.get("tokenizer") or _ddp_unwrap(model).name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source)
    blocks = pack_token_blocks(texts_from_rows(rows, text_field), tokenizer, block_size)
    if _is_rank0():
        print(f"[trainer] packed {len(rows)} rows into {len(blocks)} blocks "
              f"of up to {block_size} tokens (tokenizer: {source})")
    return blocks, _stack_blocks, tokenizer


# ---------------------------------------------------------------------------
# Mid-stage checkpoints (engine CONTRACT.md §8)
# ---------------------------------------------------------------------------
#
# When the engine assigns a checkpoint slot (BLUT_CHECKPOINT_DIR), the loop
# saves a step checkpoint every `checkpoint_every_steps` steps (or, when that is
# unset, every `checkpoint_every_secs` seconds, default 600) and once more at
# the end. With BLUT_RESUME=1 it restores model, optimizer, scheduler, RNG and
# the position inside the epoch from the newest valid checkpoint and carries on
# from there: same step count, same data order, same losses as a run that was
# never interrupted.

class _ResumableSampler:
    """Wraps a sampler so one epoch can start part-way through.

    `skip` indices are dropped from the next iteration only (a resumed epoch
    skips the batches it already trained on), so with `skip == 0` the draws
    and the order are exactly those of the wrapped sampler."""

    def __init__(self, base):
        self.base = base
        self.skip = 0

    def set_epoch(self, epoch):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        skip, self.skip = self.skip, 0
        it = iter(self.base)
        for _ in range(skip):
            if next(it, None) is None:
                return
        yield from it


def _checkpoint_slot(config: dict, ddp: bool, strategy: str):
    """The slot the engine assigned, or None. FSDP checkpoints are sharded and
    not written here, so an FSDP run ignores the slot and restarts from step 0
    when interrupted, as before."""
    from blut_core.checkpoint_slot import Slot
    slot = Slot.from_env(keep=int(config.get("checkpoint_keep") or 2))
    if slot is not None and ddp and strategy == "fsdp":
        if _is_rank0():
            print("[trainer] checkpoint slot ignored: mid-stage checkpoints are not "
                  "written for parallel_strategy=fsdp")
        return None
    return slot


def _rng_state() -> dict:
    """Every RNG the loop draws from, as plain values torch.load accepts
    with weights_only=True."""
    import random
    import torch
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
        kind, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = [kind, keys.tolist(), int(pos), int(has_gauss), float(cached)]
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    import random
    import torch
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np
            kind, keys, pos, has_gauss, cached = state["numpy"]
            np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
        except ImportError:
            pass
    if "cuda" in state and torch.cuda.is_available() \
            and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def _broadcast_payload(payload, rank_device: str):
    """Send rank 0's checkpoint to every rank. Tensors go one by one with
    `dist.broadcast`; only the structure around them is pickled, so a large
    optimizer state is not pickled whole."""
    import torch
    import torch.distributed as dist
    tensors = []

    def strip(obj):
        if torch.is_tensor(obj):
            tensors.append(obj)
            return {"__tensor__": len(tensors) - 1, "shape": list(obj.shape),
                    "dtype": str(obj.dtype).replace("torch.", "")}
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [strip(v) for v in obj]
        return obj

    box = [strip(payload) if _is_rank0() else None]
    dist.broadcast_object_list(box, src=0)
    skeleton = box[0]
    if skeleton is None:
        return None
    metas = []

    def collect(obj):
        if isinstance(obj, dict) and "__tensor__" in obj:
            metas.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)

    collect(skeleton)
    metas.sort(key=lambda m: m["__tensor__"])
    received = []
    for meta in metas:
        dtype = getattr(torch, meta["dtype"])
        if _is_rank0():
            t = tensors[meta["__tensor__"]].contiguous()
        else:
            t = torch.empty(meta["shape"], dtype=dtype)
        # gloo has no bool type; bool tensors travel as uint8.
        wire = t.to(torch.uint8) if dtype is torch.bool else t
        if rank_device.startswith("cuda"):
            wire = wire.to(rank_device)
        dist.broadcast(wire, src=0)
        wire = wire.to("cpu")
        received.append(wire.to(torch.bool) if dtype is torch.bool else wire)

    def rebuild(obj, like):
        if isinstance(obj, dict) and "__tensor__" in obj:
            return received[obj["__tensor__"]]
        if isinstance(obj, dict):
            return {k: rebuild(v, like.get(k) if isinstance(like, dict) else None)
                    for k, v in obj.items()}
        if isinstance(obj, list):
            src = like if isinstance(like, (list, tuple)) else [None] * len(obj)
            out = [rebuild(v, l) for v, l in zip(obj, src)]
            return tuple(out) if isinstance(like, tuple) else out
        return obj

    return rebuild(skeleton, payload if _is_rank0() else None)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(config: dict, ingredients: dict, dataset):
    """Run the training loop."""
    import torch

    model = ingredients["model"]
    loss_fn = ingredients["loss"]
    step_fn = ingredients["step"]
    emit = ingredients["logging"]
    save_fn = ingredients["checkpoint"]

    epochs = config.get("epochs", 1)
    device = config.get("device", "cuda")
    output_dir = Path(config["output_dir"])
    seed = config.get("seed", 42)

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Distributed initialization (DDP replicate, or FSDP2 shard)
    ddp = _is_ddp()
    strategy = _parallel_strategy(config) if ddp else "ddp"
    if ddp:
        rank_device = _ddp_init(device)
        if strategy == "fsdp" and not rank_device.startswith("cuda"):
            raise RuntimeError("parallel_strategy=fsdp needs CUDA; use ddp on the CPU")
        # Wrap the model per strategy. Both wraps move it to the local GPU;
        # the data-parallel axis (DistributedSampler + loss all-reduce) is
        # identical — FSDP shards params, not the data dimension.
        if strategy == "fsdp":
            model = _fsdp_wrap_model(model)
        else:
            model = _ddp_wrap_model(model, rank_device)
        # Create output dir (rank 0 only)
        if _is_rank0():
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Single-GPU: move model to device
        if device != "cpu" and torch.cuda.is_available():
            model = model.to(device)
        else:
            model = model.to("cpu")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer and scheduler come after placement and wrapping, from the
    # parameters the model will actually train with (see build_optimization).
    optimizer, scheduler = build_optimization(config, model)

    batch_size = int(config.get("batch_size", 32))
    world = _ddp_world_size()
    if _is_rank0():
        print(f"[trainer] starting training: {epochs} epochs, "
               f"device={device}, ddp={ddp}, strategy={strategy}, "
               f"world_size={world}, batch_size={batch_size} per rank, "
               f"global_batch={batch_size * world}")
        print(f"[trainer] available ingredients: {list_ingredients()}")

    dataset, collate, tokenizer = prepare_dataset(config, dataset, model)

    # One loading path for every dataset. JSONL rows used to be sliced by hand
    # on every rank — each DDP rank iterated the WHOLE dataset, so N GPUs did
    # the same work N times and the global batch was never what it claimed.
    dl_kwargs = _dataloader_kwargs(config)
    generator = None
    if ddp:
        # Preserve the old DDP default (pin_memory=True) unless the config
        # explicitly set pin_memory — otherwise existing DDP runs silently
        # lose pinned H2D copies (a perf regression, not a correctness one).
        if "pin_memory" not in config:
            dl_kwargs["pin_memory"] = True
        from torch.utils.data import DistributedSampler
        sampler = _ResumableSampler(DistributedSampler(dataset, seed=seed, shuffle=True))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate, **dl_kwargs)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        # What `shuffle=True, generator=generator` builds, made explicit so a
        # resumed epoch can skip the batches it already trained on.
        sampler = _ResumableSampler(
            torch.utils.data.RandomSampler(dataset, generator=generator))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, generator=generator,
            collate_fn=collate, **dl_kwargs)
    # Opt-in CUDA-stream prefetch: double-buffer the next batch onto a side
    # stream while the model computes the current one (GPU only).
    if (config.get("cuda_prefetch", False)
            and str(device) != "cpu" and torch.cuda.is_available()):
        dataloader = CudaPrefetcher(dataloader, device)

    # Mid-stage checkpoints: the slot, the cadence, and where to start.
    slot = _checkpoint_slot(config, ddp, strategy)
    rank_dev = rank_device if ddp else ("cpu" if device == "cpu" else device)
    batches_per_epoch = len(dataloader)
    total_steps = batches_per_epoch * epochs
    every_steps = int(config.get("checkpoint_every_steps") or 0)
    every_secs = float(config.get("checkpoint_every_secs") or 600)
    log_every = int(config.get("log_every_steps") or 0)
    fingerprint = None
    start_epoch, skip_batches, global_step = 1, 0, 0
    # (python-float loss sum, on-device loss sum, batches) of the interrupted
    # epoch. The device sum is kept as the tensor it was so the epoch loss
    # adds up in the same order and precision as an uninterrupted run.
    resume_partial = None
    epoch_gen_state = None
    resume_rng = None
    if slot is not None:
        from blut_core.checkpoint_slot import config_fingerprint
        fingerprint = config_fingerprint(config)
        payload, loaded_from = None, None
        if slot.resume:
            if _is_rank0():
                slot.remove_temp_files()
                payload, loaded_from = slot.load_latest(fingerprint, world)
            if ddp:
                payload = _broadcast_payload(payload, rank_dev)
        elif _is_rank0():
            slot.discard_all()
        if payload is not None:
            _ddp_unwrap(model).load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["opt"])
            if scheduler is not None and payload.get("sched") is not None \
                    and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(payload["sched"])
            global_step = int(payload["step"])
            start_epoch = int(payload["epoch"])
            skip_batches = int(payload["batch_in_epoch"])
            sums = payload.get("epoch_loss_sums") or []
            py_sum, dev_sum = sums[_ddp_rank()] if _ddp_rank() < len(sums) else (0.0, None)
            resume_partial = (float(py_sum), dev_sum, skip_batches)
            epoch_gen_state = payload.get("epoch_generator")
            resume_rng = payload["rng"]
            if _is_rank0():
                print(f"[trainer] resumed from {loaded_from.name}: step {global_step}/"
                      f"{total_steps}, epoch {start_epoch}, batch {skip_batches}/"
                      f"{batches_per_epoch}")
        elif slot.resume and _is_rank0():
            print("[trainer] no valid checkpoint to resume from; starting at step 0")
        if _is_rank0():
            slot.begin(step=global_step if payload is not None else None,
                       epoch=start_epoch if payload is not None else None,
                       resumed_from=global_step if payload is not None else None)
            cadence = (f"every {every_steps} steps" if every_steps > 0
                       else f"every {every_secs:g} s")
            print(f"[trainer] checkpoints {cadence} in {slot.dir}")
    last_saved_step = global_step
    last_save_time = time.monotonic()
    # The last epoch's mean training loss: what `done` reports. None until an
    # epoch completes in this attempt.
    final_loss = None

    def _checkpoint_due() -> bool:
        if every_steps > 0:
            return global_step % every_steps == 0
        due = time.monotonic() - last_save_time >= every_secs
        if ddp:
            # Rank 0's clock decides, so every rank enters the save together.
            import torch.distributed as dist
            flag = torch.tensor([1 if (due and _is_rank0()) else 0], device=rank_dev)
            dist.broadcast(flag, src=0)
            due = bool(flag.item())
        return due

    def _save_checkpoint(epoch_no: int, batch_no: int, partial_sum, gen_state):
        nonlocal last_saved_step, last_save_time
        if ddp:
            import torch.distributed as dist
            sums = [None] * world
            dist.all_gather_object(sums, partial_sum)
        else:
            sums = [partial_sum]
        if _is_rank0():
            base = _ddp_unwrap(model)
            payload = {
                "model": base.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": (scheduler.state_dict()
                          if scheduler is not None and hasattr(scheduler, "state_dict")
                          else None),
                "config": config,
                "fingerprint": fingerprint,
                "world_size": world,
                "step": global_step,
                "epoch": epoch_no,
                "batch_in_epoch": batch_no,
                "epoch_loss_sums": sums,
                "epoch_generator": gen_state,
                "rng": _rng_state(),
            }
            path = slot.save(payload, step=global_step, epoch=epoch_no)
            print(f"[trainer] checkpoint step {global_step}/{total_steps} -> {path.name}")
            status.relay("saved", path=str(path))
        last_saved_step = global_step
        last_save_time = time.monotonic()

    try:
        # Training loop
        for epoch in range(start_epoch, epochs + 1):
            # The LR for this epoch, set before it trains. Stepping only after
            # an epoch left epoch 1 at the optimizer's peak LR, so a warmup
            # began by dropping from the peak, one epoch late. Every rank.
            if scheduler is not None:
                scheduler.step(epoch)

            model.train()
            epoch_loss = 0.0
            n_batches = 0
            resuming = resume_partial is not None and epoch == start_epoch

            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            if resuming and generator is not None and epoch_gen_state is not None:
                generator.set_state(epoch_gen_state)
            this_epoch_gen = generator.get_state() if generator is not None else None
            if resuming:
                sampler.skip = skip_batches * batch_size
                epoch_loss, _, n_batches = resume_partial

            # No exception is caught here. This loop used to catch every batch's
            # exception, print a warning, and continue — so a run in which every
            # batch failed averaged zero losses into `loss=0.0000`, saved a
            # checkpoint, and exited 0. A failed batch now fails the run. Under
            # DDP that is also the only safe choice: a rank that skips a batch
            # its peers train on deadlocks the next gradient all-reduce.
            target_device = (rank_device if ddp
                             else ("cpu" if device == "cpu" else device))
            # Losses accumulate on the device. `.item()` per batch forces a
            # host-device sync every step, stalling the queue of launched kernels
            # for a number that is only reported once per epoch.
            device_loss_sum = None
            if resuming and resume_partial[1] is not None:
                device_loss_sum = resume_partial[1].to(target_device)
            batches = iter(dataloader)
            if resuming:
                # After the iterator exists: creating it draws from the global RNG,
                # which the interrupted run had already done before it saved.
                _set_rng_state(resume_rng)
            for batch in batches:
                if isinstance(batch, dict):
                    batch = {k: v.to(target_device) if hasattr(v, "to") else v
                             for k, v in batch.items()}
                loss = step_fn(model, optimizer, loss_fn, batch)
                if hasattr(loss, "detach"):
                    step_loss = loss.detach()
                    device_loss_sum = (step_loss if device_loss_sum is None
                                       else device_loss_sum + step_loss)
                else:
                    step_loss = float(loss)
                    epoch_loss += step_loss
                n_batches += 1
                global_step += 1
                if log_every > 0 and global_step % log_every == 0:
                    value = float(step_loss.item() if hasattr(step_loss, "item") else step_loss)
                    if ddp:
                        import torch.distributed as dist
                        t = torch.tensor([value], device=rank_dev)
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        value = t.item() / world
                    if _is_rank0():
                        emit({"step": global_step, "loss": value, "epoch": epoch}, kind="step")
                        print(f"[trainer] step {global_step}/{total_steps} loss={value:.4f}")
                        _report_step(global_step, total_steps, value,
                                     _current_lr(optimizer), _vram_mb(rank_dev))
                if slot is not None and _checkpoint_due():
                    partial = (epoch_loss, device_loss_sum.detach().to("cpu").clone()
                               if device_loss_sum is not None else None)
                    _save_checkpoint(epoch, n_batches, partial, this_epoch_gen)
            if device_loss_sum is not None:
                epoch_loss += device_loss_sum.item()

            if n_batches == 0:
                raise RuntimeError(
                    f"epoch {epoch} ran zero batches; the dataset yielded nothing "
                    "to train on for this rank")

            avg_loss = epoch_loss / n_batches

            # Reduce avg_loss across DDP ranks for consistent reporting
            if ddp:
                import torch.distributed as dist
                # SUM then divide: gloo (the CPU backend) has no AVG reduction.
                loss_tensor = torch.tensor([avg_loss], device=rank_device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / _ddp_world_size()

            # Emit metrics (rank 0 only)
            if _is_rank0():
                metrics = {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "n_batches": n_batches,
                    "rank": _ddp_rank(),
                }
                if scheduler is not None and hasattr(scheduler, 'get_last_lr'):
                    metrics["lr"] = scheduler.get_last_lr()[0]
                emit(metrics, kind="epoch")
                print(f"[trainer] epoch {epoch}/{epochs}: loss={avg_loss:.4f}")
                _report_epoch(epoch, epochs, avg_loss)
                final_loss = avg_loss
    finally:
        # Stop the heartbeat however the loop ends. After a failure the marker
        # keeps the last durable step, so the next attempt can resume from it.
        if slot is not None and _is_rank0():
            slot.stop_heartbeat()

    # A final step checkpoint, so an output the engine never got to commit is
    # recovered by resuming at the last step instead of training again.
    if slot is not None:
        if global_step != last_saved_step:
            _save_checkpoint(epochs + 1, 0, (0.0, None), None)
        if _is_rank0():
            slot.end()

    # Save checkpoint. For FSDP the state-dict GATHER is a collective — every
    # rank must call it — but only rank 0 holds the full dict and writes.
    if strategy == "fsdp":
        model_state = _fsdp_full_state_dict(model)  # collective; all ranks
        base_model = _ddp_unwrap(model)  # identity for FSDP (no .module)
    else:
        base_model = _ddp_unwrap(model) if ddp else model
        if ddp:
            replica = _check_replicas_agree(base_model)  # collective; all ranks
            if replica is not None:
                emit(replica, kind="replicas")
        model_state = base_model.state_dict() if _is_rank0() else None

    if _is_rank0():
        ckpt_path = output_dir / "model.pt"
        # Opt-in async checkpoint. NOTE: this trainer saves ONCE at the end, so
        # here the async path is functionally a synchronous save (snapshot →
        # write → join, with nothing left to overlap). AsyncCheckpointSaver is
        # the reusable primitive; its overlap payoff lands in a trainer that
        # saves DURING the loop (construct once, save() per epoch, close() at
        # cleanup) — e.g. the train_joint integration. Kept wired here so the
        # config flag + the primitive are exercised end-to-end.
        if config.get("async_checkpoint", False):
            # Context manager: close() (join + pool shutdown) always runs, even
            # if save() raises — no ThreadPoolExecutor leak.
            with AsyncCheckpointSaver(save_fn) as saver:
                saver.save({"model": model_state}, str(ckpt_path))
        else:
            save_fn({"model": model_state}, str(ckpt_path))
        print(f"[trainer] checkpoint saved to {ckpt_path}")
        status.relay("saved", path=str(ckpt_path))

        # Also save in HuggingFace format if possible. FSDP's sharded module
        # can't `save_pretrained` directly (params are DTensors); skip it —
        # the gathered model.pt above is the portable artifact.
        try:
            if strategy != "fsdp" and hasattr(base_model, 'save_pretrained'):
                hf_dir = output_dir / "hf"
                base_model.save_pretrained(str(hf_dir))
                # Without its tokenizer an HF checkpoint cannot be loaded for
                # inference by name alone; ship the one training used.
                if tokenizer is not None:
                    tokenizer.save_pretrained(str(hf_dir))
                print(f"[trainer] HF checkpoint saved to {hf_dir}")
        except Exception as e:
            print(f"[trainer] warning: could not save HF checkpoint: {e}")

    # DDP cleanup
    if ddp:
        _ddp_cleanup()

    if _is_rank0():
        print("[trainer] training complete")
        # An attempt that resumed past its last epoch trained none here and
        # has no loss of its own to report, so it sends no `done`.
        if final_loss is not None:
            status.relay("done", final_loss=final_loss, checkpoint_dir=str(output_dir))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # `train_model` reads this stdout through a pipe, where Python buffers by
    # the block: the `[trainer]` lines would reach the terminal late, and a
    # killed run would lose them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="BLUT core generic trainer")
    parser.add_argument("--config", required=True, help="Path to ingredient config JSON")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        ingredients = build_all_ingredients(config)
        dataset = load_dataset(config)
        train_loop(config, ingredients, dataset)
    except Exception as e:
        status.relay("failed", error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
