"""Generic ingredient-based evaluator for BLUT core cookbook.

Reads an eval config JSON (written by the Rust evaluation stages), loads the
checkpoint and dataset, prepares the rows exactly as training does, and runs
the eval ingredient over them.

Usage:
    python -m blut_core.evaluator --config eval_config.json --output eval_report.json

Every failure exits non-zero with the reason. This module used to swallow a
failed HuggingFace load into a warning, and it called its eval ingredient
without the loss function that ingredient requires, so no report it could
produce had ever been computed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blut_core import build_ingredient
from blut_core.lm_data import is_causal_lm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_model(checkpoint_path: Path):
    """Load the HuggingFace model the trainer saved under `<checkpoint>/hf`.

    The generic trainer also writes `model.pt`, a bare state dict. Loading one
    needs the model's architecture, which a checkpoint directory does not
    record, so only the HuggingFace format is evaluable here.
    """
    hf_dir = checkpoint_path / "hf"
    if not hf_dir.is_dir():
        raise FileNotFoundError(
            f"no HuggingFace checkpoint at {hf_dir}; the generic evaluator can "
            "only load models the trainer saved with save_pretrained")
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(hf_dir))


def resolve_device(requested: str) -> str:
    """`requested`, or the CPU when CUDA was asked for and is absent."""
    import torch
    if requested != "cpu" and not torch.cuda.is_available():
        print(f"[evaluator] {requested} requested but CUDA is unavailable; using cpu")
        return "cpu"
    return requested


def evaluate(config: dict) -> dict:
    """Run the configured evaluation and return its metrics."""
    import torch

    from blut_core.trainer import load_dataset, prepare_dataset

    eval_cfg = config["eval"]
    print(f"[evaluator] building eval: {eval_cfg['kind']}:{eval_cfg['name']}")
    evaluate_fn = build_ingredient(eval_cfg["kind"], eval_cfg["name"], eval_cfg.get("config"))

    checkpoint_path = Path(config["checkpoint_path"])
    print(f"[evaluator] loading checkpoint from {checkpoint_path}")
    model = load_model(checkpoint_path)
    device = resolve_device(config.get("device") or "cuda")
    model = model.to(device)

    rows = load_dataset({"dataset_path": config["dataset_path"]})
    dataset, collate, tokenizer = prepare_dataset(config, rows, model)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(config.get("batch_size") or 64),
        shuffle=False, collate_fn=collate)

    # The loss training used, when the stage passed it: an eval ingredient
    # that scores with a loss (`loss_eval`) needs one.
    loss_fn = None
    loss_cfg = config.get("loss")
    if loss_cfg:
        loss_fn = build_ingredient(loss_cfg["kind"], loss_cfg["name"], loss_cfg.get("config"))
    elif is_causal_lm(model):
        loss_fn = build_ingredient("loss", "causal_lm", {})

    unit = "token blocks" if tokenizer is not None else "rows"
    print(f"[evaluator] evaluating {len(rows)} rows as {len(dataset)} {unit} on {device}")
    metrics = dict(evaluate_fn(model, loader, device=device, loss_fn=loss_fn))
    metrics["n_rows"] = len(rows)
    metrics["n_items"] = len(dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BLUT core generic evaluator")
    parser.add_argument("--config", required=True, help="Path to eval config JSON")
    parser.add_argument("--output", required=True, help="Path to output eval report JSON")
    args = parser.parse_args()

    metrics = evaluate(load_config(args.config))
    print(f"[evaluator] metrics: {metrics}")
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluator] report written to {args.output}")


if __name__ == "__main__":
    main()
