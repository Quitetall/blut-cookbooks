"""Generic ingredient-based evaluator for BLUT core cookbook.

Reads an eval config JSON (written by the EvaluateModel Rust stage),
builds the eval ingredient, and runs evaluation.

Usage:
    python -m blut_core.evaluator --config eval_config.json --output eval_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blut_core import build_ingredient


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="BLUT core generic evaluator")
    parser.add_argument("--config", required=True, help="Path to eval config JSON")
    parser.add_argument("--output", required=True, help="Path to output eval report JSON")
    args = parser.parse_args()

    config = load_config(args.config)

    checkpoint_path = config["checkpoint_path"]
    dataset_path = config["dataset_path"]
    eval_cfg = config["eval"]
    batch_size = config.get("batch_size", 64)
    device = config.get("device", "cuda")

    # Build eval ingredient
    print(f"[evaluator] building eval: {eval_cfg['kind']}:{eval_cfg['name']}")
    evaluate_fn = build_ingredient(eval_cfg["kind"], eval_cfg["name"], eval_cfg.get("config"))

    # Load checkpoint
    import torch
    print(f"[evaluator] loading checkpoint from {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path)

    # Try to load as HuggingFace model
    model = None
    hf_dir = checkpoint_path / "hf"
    if hf_dir.exists():
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(str(hf_dir))
            print(f"[evaluator] loaded HF model from {hf_dir}")
        except Exception as e:
            print(f"[evaluator] warning: could not load HF model: {e}")

    if model is None:
        # Try loading as torch checkpoint
        pt_path = checkpoint_path / "model.pt"
        if pt_path.exists():
            print(f"[evaluator] found torch checkpoint at {pt_path}")
            # Need a model architecture to load into — this is a limitation
            # of the generic evaluator. Real usage should provide the model.
            print("[evaluator] error: no model architecture available for torch checkpoint")
            sys.exit(1)
        else:
            print(f"[evaluator] error: no checkpoint found at {checkpoint_path}")
            sys.exit(1)

    # Load dataset
    print(f"[evaluator] loading dataset from {dataset_path}")
    try:
        from datasets import load_dataset
        ds = load_dataset("json", data_files=str(dataset_path), split="train")
        print(f"[evaluator] loaded {len(ds)} examples")
    except Exception as e:
        print(f"[evaluator] error: could not load dataset: {e}")
        sys.exit(1)

    # Run evaluation
    print(f"[evaluator] running evaluation...")
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device)

    metrics = evaluate_fn(model, ds, device=device)
    print(f"[evaluator] metrics: {metrics}")

    # Write output
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluator] report written to {args.output}")


if __name__ == "__main__":
    main()
