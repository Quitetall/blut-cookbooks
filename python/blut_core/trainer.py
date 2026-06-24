"""Generic ingredient-based trainer for BLUT core cookbook.

Reads an ingredient config JSON (written by the TrainModel Rust stage),
builds all ingredients via the registry, and runs a training loop.

Usage:
    python -m blut_core.trainer --config train_config.json
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


def load_config(config_path: str) -> dict:
    """Load and validate the ingredient config JSON."""
    with open(config_path) as f:
        config = json.load(f)
    required = ["dataset_path", "output_dir", "model", "optimizer", "scheduler", "loss"]
    for key in required:
        if key not in config:
            raise ValueError(f"missing required key '{key}' in config")
    return config


def build_all_ingredients(config: dict) -> dict:
    """Build all ingredients from the config."""
    ingredients = {}

    # Build model
    model_cfg = config["model"]
    print(f"[trainer] building model: {model_cfg['kind']}:{model_cfg['name']}")
    ingredients["model"] = build_ingredient(
        model_cfg["kind"], model_cfg["name"], model_cfg.get("config"))

    # Build optimizer (needs model params)
    opt_cfg = config["optimizer"]
    print(f"[trainer] building optimizer: {opt_cfg['kind']}:{opt_cfg['name']}")
    model = ingredients["model"]
    named_params = list(model.named_parameters()) if hasattr(model, 'named_parameters') else []
    ingredients["optimizer"] = build_ingredient(
        opt_cfg["kind"], opt_cfg["name"], opt_cfg.get("config"),
        named_params=named_params)

    # Build scheduler (needs optimizer)
    sched_cfg = config["scheduler"]
    print(f"[trainer] building scheduler: {sched_cfg['kind']}:{sched_cfg['name']}")
    ingredients["scheduler"] = build_ingredient(
        sched_cfg["kind"], sched_cfg["name"], sched_cfg.get("config"),
        optimizer=ingredients["optimizer"])

    # Build loss
    loss_cfg = config["loss"]
    print(f"[trainer] building loss: {loss_cfg['kind']}:{loss_cfg['name']}")
    ingredients["loss"] = build_ingredient(
        loss_cfg["kind"], loss_cfg["name"], loss_cfg.get("config"))

    # Build step
    step_cfg = config.get("step", {"kind": "step", "name": "standard", "config": {}})
    print(f"[trainer] building step: {step_cfg['kind']}:{step_cfg['name']}")
    ingredients["step"] = build_ingredient(
        step_cfg["kind"], step_cfg["name"], step_cfg.get("config"))

    # Build logging
    print("[trainer] building logging: logging:blut_metric")
    ingredients["logging"] = build_ingredient("logging", "blut_metric", {})

    # Build checkpoint
    print("[trainer] building checkpoint: checkpoint:atomic_save")
    ingredients["checkpoint"] = build_ingredient("checkpoint", "atomic_save", {})

    return ingredients


def load_dataset(config: dict):
    """Load the dataset from the config."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 32)

    # Try to load as JSONL
    if str(dataset_path).endswith(".jsonl"):
        texts = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    texts.append(json.loads(line))
        # Simple tokenization fallback — real usage should provide a proper dataset
        print(f"[trainer] loaded {len(texts)} examples from {dataset_path}")
        return texts

    # Fallback: try HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("json", data_files=str(dataset_path), split="train")
        print(f"[trainer] loaded {len(ds)} examples from {dataset_path}")
        return ds
    except Exception as e:
        print(f"[trainer] warning: could not load dataset: {e}")
        return []


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

    # Move model to device
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to("cpu")

    print(f"[trainer] starting training: {epochs} epochs, device={device}")
    print(f"[trainer] available ingredients: {list_ingredients()}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Simple training loop — real usage should provide a proper DataLoader
        if isinstance(dataset, list):
            # JSONL dataset — iterate in batches
            batch_size = config.get("batch_size", 32)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                # Convert batch to tensors (placeholder — real usage needs proper encoding)
                try:
                    loss = step_fn(model, optimizer, loss_fn, batch)
                    epoch_loss += loss.item() if hasattr(loss, 'item') else float(loss)
                    n_batches += 1
                except Exception as e:
                    print(f"[trainer] warning: batch {i} failed: {e}")
                    continue
        else:
            # HuggingFace dataset with DataLoader
            try:
                batch_size = config.get("batch_size", 32)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for batch in dataloader:
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                    try:
                        loss = step_fn(model, optimizer, loss_fn, batch)
                        epoch_loss += loss.item() if hasattr(loss, 'item') else float(loss)
                        n_batches += 1
                    except Exception as e:
                        print(f"[trainer] warning: batch failed: {e}")
                        continue
            except Exception as e:
                print(f"[trainer] warning: dataloader failed: {e}")

        # Step scheduler
        if scheduler is not None:
            scheduler.step(epoch)

        avg_loss = epoch_loss / max(n_batches, 1)
        metrics = {
            "epoch": epoch,
            "loss": avg_loss,
            "n_batches": n_batches,
        }
        if scheduler is not None and hasattr(scheduler, 'get_last_lr'):
            metrics["lr"] = scheduler.get_last_lr()[0]

        emit(metrics, kind="epoch")
        print(f"[trainer] epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

    # Save checkpoint
    ckpt_path = output_dir / "model.pt"
    save_fn({"model": model.state_dict()}, str(ckpt_path))
    print(f"[trainer] checkpoint saved to {ckpt_path}")

    # Also save in HuggingFace format if possible
    try:
        if hasattr(model, 'save_pretrained'):
            hf_dir = output_dir / "hf"
            model.save_pretrained(str(hf_dir))
            print(f"[trainer] HF checkpoint saved to {hf_dir}")
    except Exception as e:
        print(f"[trainer] warning: could not save HF checkpoint: {e}")


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
