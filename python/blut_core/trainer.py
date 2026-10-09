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
import os
import sys
from pathlib import Path

# Ensure blut_core is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from blut_core import build_ingredient, list_ingredients
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


def _ddp_init():
    """Initialize the DDP process group (NCCL backend)."""
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = _ddp_local_rank()
    torch.cuda.set_device(local_rank)
    print(f"[trainer] DDP initialized: rank={_ddp_rank()}, "
          f"local_rank={local_rank}, world_size={_ddp_world_size()}")


def _ddp_cleanup():
    """Destroy the DDP process group."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _ddp_wrap_model(model):
    """Wrap model with DistributedDataParallel."""
    import torch.nn.parallel as parallel
    local_rank = _ddp_local_rank()
    model = model.to(f"cuda:{local_rank}")
    model = parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)
    return model


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
        _ddp_init()
        # Wrap the model per strategy. Both wraps move it to the local GPU;
        # the data-parallel axis (DistributedSampler + loss all-reduce) is
        # identical — FSDP shards params, not the data dimension.
        if strategy == "fsdp":
            model = _fsdp_wrap_model(model)
        else:
            model = _ddp_wrap_model(model)
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
    sampler = None
    if ddp:
        # Preserve the old DDP default (pin_memory=True) unless the config
        # explicitly set pin_memory — otherwise existing DDP runs silently
        # lose pinned H2D copies (a perf regression, not a correctness one).
        if "pin_memory" not in config:
            dl_kwargs["pin_memory"] = True
        sampler, dataloader = _ddp_shard_dataset(
            dataset, batch_size, seed, dict(dl_kwargs, collate_fn=collate))
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator,
            collate_fn=collate, **dl_kwargs)
    # Opt-in CUDA-stream prefetch: double-buffer the next batch onto a side
    # stream while the model computes the current one (GPU only).
    if (config.get("cuda_prefetch", False)
            and str(device) != "cpu" and torch.cuda.is_available()):
        dataloader = CudaPrefetcher(dataloader, device)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if sampler is not None:
            sampler.set_epoch(epoch)

        # No exception is caught here. This loop used to catch every batch's
        # exception, print a warning, and continue — so a run in which every
        # batch failed averaged zero losses into `loss=0.0000`, saved a
        # checkpoint, and exited 0. A failed batch now fails the run. Under
        # DDP that is also the only safe choice: a rank that skips a batch
        # its peers train on deadlocks the next gradient all-reduce.
        target_device = (f"cuda:{_ddp_local_rank()}" if ddp
                         else ("cpu" if device == "cpu" else device))
        # Losses accumulate on the device. `.item()` per batch forces a
        # host-device sync every step, stalling the queue of launched kernels
        # for a number that is only reported once per epoch.
        device_loss_sum = None
        for batch in dataloader:
            if isinstance(batch, dict):
                batch = {k: v.to(target_device) if hasattr(v, "to") else v
                         for k, v in batch.items()}
            loss = step_fn(model, optimizer, loss_fn, batch)
            if hasattr(loss, "detach"):
                step_loss = loss.detach()
                device_loss_sum = (step_loss if device_loss_sum is None
                                   else device_loss_sum + step_loss)
            else:
                epoch_loss += float(loss)
            n_batches += 1
        if device_loss_sum is not None:
            epoch_loss += device_loss_sum.item()

        if n_batches == 0:
            raise RuntimeError(
                f"epoch {epoch} ran zero batches; the dataset yielded nothing "
                "to train on for this rank")

        # Step scheduler (all ranks)
        if scheduler is not None:
            scheduler.step(epoch)

        avg_loss = epoch_loss / n_batches

        # Reduce avg_loss across DDP ranks for consistent reporting
        if ddp:
            import torch.distributed as dist
            loss_tensor = torch.tensor([avg_loss], device=f"cuda:{_ddp_local_rank()}")
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

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

    # Save checkpoint. For FSDP the state-dict GATHER is a collective — every
    # rank must call it — but only rank 0 holds the full dict and writes.
    if strategy == "fsdp":
        model_state = _fsdp_full_state_dict(model)  # collective; all ranks
        base_model = _ddp_unwrap(model)  # identity for FSDP (no .module)
    else:
        base_model = _ddp_unwrap(model) if ddp else model
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BLUT core generic trainer")
    parser.add_argument("--config", required=True, help="Path to ingredient config JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    ingredients = build_all_ingredients(config)
    dataset = load_dataset(config)
    train_loop(config, ingredients, dataset)


if __name__ == "__main__":
    main()
