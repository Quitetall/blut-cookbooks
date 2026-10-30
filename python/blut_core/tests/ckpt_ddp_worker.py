"""Worker for test_checkpoint_slot's DDP test: two gloo ranks under torchrun.

Usage: python -m torch.distributed.run --standalone --nproc_per_node 2 \
           ckpt_ddp_worker.py OUT_DIR CRASH_AT

Rank 0 writes its per-step losses to OUT_DIR/losses.json, also when the run
crashes. CRASH_AT > 0 makes rank 0 raise on that step call of this run.
"""
import json
import os
import sys
from pathlib import Path

import torch

import blut_core.trainer as trainer


def main():
    out, crash_at = Path(sys.argv[1]), int(sys.argv[2])
    g = torch.Generator().manual_seed(7)
    x = torch.randn(40, 4, generator=g)
    rows = [{"x": x[i].tolist(), "y": float(x[i].sum() * 0.5)} for i in range(40)]
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3),
                                torch.nn.Linear(8, 1))
    losses, calls = [], [0]

    def step(model, optimizer, loss_fn, batch, **kw):
        calls[0] += 1
        # Only rank 0 dies, after its synchronous save of the previous
        # checkpoint; rank 1 is left blocked in the next all-reduce until
        # torchrun tears the group down, as when a node goes away.
        if crash_at and calls[0] == crash_at and os.environ.get("RANK") == "0":
            raise RuntimeError(f"node lost at call {crash_at}")
        xb = torch.tensor([r["x"] for r in batch])
        yb = torch.tensor([[r["y"]] for r in batch])
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        return loss

    config = {
        "dataset_path": "unused", "output_dir": str(out / "ckpt"),
        "optimizer": {"kind": "optimizer", "name": "adamw", "config": {"lr": 1e-2}},
        "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
        "epochs": 1, "batch_size": 2, "device": "cpu", "pin_memory": False,
        "checkpoint_every_steps": 3,
    }
    ingredients = {"model": model, "loss": None, "step": step,
                   "logging": lambda *a, **k: None,
                   "checkpoint": lambda payload, path: torch.save(payload, path)}
    try:
        trainer.train_loop(config, ingredients, rows)
    finally:
        if os.environ.get("RANK", "0") == "0":
            (out / "losses.json").write_text(json.dumps(losses))
    if os.environ.get("RANK", "0") == "0":
        os.replace(out / "ckpt" / "model.pt", out / "model.pt")


if __name__ == "__main__":
    main()
