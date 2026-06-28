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


def _ddp_shard_dataset(dataset, batch_size: int, seed: int):
    """Create a DistributedSampler and DataLoader for DDP."""
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    sampler = DistributedSampler(dataset, seed=seed, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True)
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
    """Gather a full (unsharded), CPU, rank-0-only state-dict from an FSDP2
    model via the DCP state-dict API. Other ranks get an empty dict; callers
    save on rank 0 only (the existing `_is_rank0()` guard)."""
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

    # Build optimizer (needs model params)
    opt_cfg = config["optimizer"]
    if _is_rank0():
        print(f"[trainer] building optimizer: {opt_cfg['kind']}:{opt_cfg['name']}")
    model = ingredients["model"]
    named_params = list(model.named_parameters()) if hasattr(model, 'named_parameters') else []
    ingredients["optimizer"] = build_ingredient(
        opt_cfg["kind"], opt_cfg["name"], opt_cfg.get("config"),
        named_params=named_params)

    # Build scheduler (needs optimizer)
    sched_cfg = config["scheduler"]
    if _is_rank0():
        print(f"[trainer] building scheduler: {sched_cfg['kind']}:{sched_cfg['name']}")
    ingredients["scheduler"] = build_ingredient(
        sched_cfg["kind"], sched_cfg["name"], sched_cfg.get("config"),
        optimizer=ingredients["optimizer"])

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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(config: dict):
    """Load the dataset from the config."""
    dataset_path = config["dataset_path"]

    # Try to load as JSONL
    if str(dataset_path).endswith(".jsonl"):
        texts = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    texts.append(json.loads(line))
        if _is_rank0():
            print(f"[trainer] loaded {len(texts)} examples from {dataset_path}")
        return texts

    # Fallback: try HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("json", data_files=str(dataset_path), split="train")
        if _is_rank0():
            print(f"[trainer] loaded {len(ds)} examples from {dataset_path}")
        return ds
    except Exception as e:
        if _is_rank0():
            print(f"[trainer] warning: could not load dataset: {e}")
        return []


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(config: dict, ingredients: dict, dataset):
    """Run the training loop."""
    import torch

    model = ingredients["model"]
    optimizer = ingredients["optimizer"]
    scheduler = ingredients["scheduler"]
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

    if _is_rank0():
        print(f"[trainer] starting training: {epochs} epochs, "
               f"device={device}, ddp={ddp}, strategy={strategy}, "
               f"world_size={_ddp_world_size()}")
        print(f"[trainer] available ingredients: {list_ingredients()}")

    # Create dataloader
    if isinstance(dataset, list):
        # JSONL dataset — simple batch iteration
        dataloader = None  # handled inline
    else:
        batch_size = config.get("batch_size", 32)
        if ddp:
            sampler, dataloader = _ddp_shard_dataset(dataset, batch_size, seed)
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Set epoch for DDP sampler
        if ddp and isinstance(dataset, list) is False and 'sampler' in dir():
            sampler.set_epoch(epoch)

        if isinstance(dataset, list):
            # JSONL dataset — iterate in batches
            batch_size = config.get("batch_size", 32)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                try:
                    loss = step_fn(model, optimizer, loss_fn, batch)
                    epoch_loss += loss.item() if hasattr(loss, 'item') else float(loss)
                    n_batches += 1
                except Exception as e:
                    if _is_rank0():
                        print(f"[trainer] warning: batch {i} failed: {e}")
                    continue
        else:
            # HuggingFace dataset with DataLoader
            for batch in dataloader:
                if isinstance(batch, dict):
                    local_rank = _ddp_local_rank() if ddp else (0 if device == "cpu" else None)
                    target_device = f"cuda:{local_rank}" if ddp and local_rank is not None else device
                    batch = {k: v.to(target_device) if hasattr(v, 'to') else v
                             for k, v in batch.items()}
                try:
                    loss = step_fn(model, optimizer, loss_fn, batch)
                    epoch_loss += loss.item() if hasattr(loss, 'item') else float(loss)
                    n_batches += 1
                except Exception as e:
                    if _is_rank0():
                        print(f"[trainer] warning: batch failed: {e}")
                    continue

        # Step scheduler (all ranks)
        if scheduler is not None:
            scheduler.step(epoch)

        avg_loss = epoch_loss / max(n_batches, 1)

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
        save_fn({"model": model_state}, str(ckpt_path))
        print(f"[trainer] checkpoint saved to {ckpt_path}")

        # Also save in HuggingFace format if possible. FSDP's sharded module
        # can't `save_pretrained` directly (params are DTensors); skip it —
        # the gathered model.pt above is the portable artifact.
        try:
            if strategy != "fsdp" and hasattr(base_model, 'save_pretrained'):
                hf_dir = output_dir / "hf"
                base_model.save_pretrained(str(hf_dir))
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
