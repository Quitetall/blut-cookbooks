"""Generic evaluation ingredient specs.

Provides common evaluation metrics: loss-based eval and accuracy.
Domain cookbooks override with domain-specific evaluation (e.g. LamQuant's
joint codec eval with Pearson-R, PRD, spectral metrics).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- Loss eval ----------------------------------------------------------
@dataclass(frozen=True)
class LossEvalConfig:
    reduction: str = "mean"


def _build_loss_eval(cfg: LossEvalConfig):
    """Return an evaluator that computes loss on a dataset."""
    import torch

    def evaluate(model, dataloader, loss_fn, device="cpu"):
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, 'to') else v
                             for k, v in batch.items()}
                else:
                    batch = batch.to(device) if hasattr(batch, 'to') else batch
                output = model(batch)
                loss = loss_fn(output, batch) if callable(loss_fn) else loss_fn
                total_loss += loss.item()
                n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        return {"loss": avg_loss}

    return evaluate


@register_ingredient
def _loss_eval():
    return IngredientSpec(
        name="loss_eval", kind="eval", config_cls=LossEvalConfig,
        cache_relevant=False,
        build=_build_loss_eval,
    )


# ---- Accuracy -----------------------------------------------------------
@dataclass(frozen=True)
class AccuracyConfig:
    top_k: tuple = (1,)         # e.g. (1, 5) for top-1 and top-5
    ignore_index: int = -100


def _build_accuracy(cfg: AccuracyConfig):
    """Return an evaluator that computes top-K accuracy."""
    import torch

    def evaluate(model, dataloader, device="cpu"):
        model.eval()
        correct = {k: 0 for k in cfg.top_k}
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids", batch.get("input"))
                    labels = batch.get("labels", batch.get("target"))
                    if inputs is not None:
                        inputs = inputs.to(device)
                    if labels is not None:
                        labels = labels.to(device)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                if logits.dim() == 3:  # (B, T, V) -> flatten
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                mask = labels != cfg.ignore_index
                logits = logits[mask]
                labels = labels[mask]
                for k in cfg.top_k:
                    topk = logits.topk(k, dim=-1).indices
                    correct[k] += (topk == labels.unsqueeze(-1)).any(-1).sum().item()
                total += labels.size(0)
        return {f"top{k}_acc": correct[k] / max(total, 1) for k in cfg.top_k}

    return evaluate


@register_ingredient
def _accuracy():
    return IngredientSpec(
        name="accuracy", kind="eval", config_cls=AccuracyConfig,
        cache_relevant=False,
        build=_build_accuracy,
    )


# ---- Perplexity ---------------------------------------------------------
@dataclass(frozen=True)
class PerplexityConfig:
    pass


def _build_perplexity(cfg: PerplexityConfig):
    """Return an evaluator that computes perplexity."""
    import math
    import torch

    def evaluate(model, dataloader, device="cpu"):
        model.eval()
        total_loss = 0.0
        n_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    labels = batch.get("labels", input_ids)
                    if input_ids is not None:
                        input_ids = input_ids.to(device)
                    if labels is not None:
                        labels = labels.to(device)
                else:
                    input_ids, labels = batch
                    input_ids, labels = input_ids.to(device), labels.to(device)
                outputs = model(input_ids, labels=labels)
                total_loss += outputs.loss.item() * labels.numel()
                n_tokens += labels.numel()
        avg_loss = total_loss / max(n_tokens, 1)
        return {"perplexity": math.exp(avg_loss), "loss": avg_loss}

    return evaluate


@register_ingredient
def _perplexity():
    return IngredientSpec(
        name="perplexity", kind="eval", config_cls=PerplexityConfig,
        cache_relevant=False,
        build=_build_perplexity,
    )
