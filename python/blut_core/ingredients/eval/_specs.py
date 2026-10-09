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
def _to_device(batch, device):
    """Move a tensor, or the tensors in a dict batch, to `device`."""
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    return batch.to(device) if hasattr(batch, "to") else batch


@dataclass(frozen=True)
class LossEvalConfig:
    reduction: str = "mean"


def _build_loss_eval(cfg: LossEvalConfig):
    """Return an evaluator that computes loss on a dataset."""
    import torch

    from blut_core.ingredients.step._specs import _forward

    def evaluate(model, dataloader, device="cpu", loss_fn=None, **kwargs):
        """Mean per-batch loss.

        Scores with `loss_fn` — the evaluator passes the loss training used.
        Without one, a model that computes its own loss (a HuggingFace model
        given `labels`) is scored by that. `loss_fn` used to be a required
        positional argument the evaluator never supplied, so this ingredient
        had never run.
        """
        model.eval()
        total = None
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = _to_device(batch, device)
                output = _forward(model, batch)
                if loss_fn is not None:
                    loss = loss_fn(output, batch)
                elif getattr(output, "loss", None) is not None:
                    loss = output.loss
                else:
                    raise ValueError(
                        "loss_eval needs a loss: pass the training loss "
                        "ingredient, or use a model that returns its own")
                loss = loss.detach()
                total = loss if total is None else total + loss
                n_batches += 1
        if n_batches == 0:
            raise ValueError("loss_eval: the dataset yielded no batches")
        return {"loss": total.item() / n_batches}

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

    from blut_core.lm_data import is_causal_lm

    def evaluate(model, dataloader, device="cpu", **kwargs):
        model.eval()
        causal = is_causal_lm(model)
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
                # A HuggingFace model returns an output object; the scores
                # are its `.logits`.
                logits = getattr(logits, "logits", logits)
                if causal:
                    # Position t predicts token t + 1. Comparing unshifted
                    # logits with labels scored every prediction against the
                    # token it was conditioned on.
                    logits = logits[:, :-1, :]
                    labels = labels[:, 1:]
                if logits.dim() == 3:  # (B, T, V) -> flatten
                    # reshape, not view: the causal shift above leaves a
                    # non-contiguous slice that view() rejects.
                    logits = logits.reshape(-1, logits.size(-1))
                    labels = labels.reshape(-1)
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

    def evaluate(model, dataloader, device="cpu", **kwargs):
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
                # The model's loss is a mean over the tokens it predicted:
                # every position but the first, excluding ignored labels.
                # Weighting each batch by that count (not by labels.numel(),
                # as this did) makes the result a true per-token mean.
                predicted = int((labels[..., 1:] != -100).sum())
                total_loss += outputs.loss.item() * predicted
                n_tokens += predicted
        if n_tokens == 0:
            raise ValueError("perplexity: no predicted tokens in the dataset")
        avg_loss = total_loss / n_tokens
        return {"perplexity": math.exp(avg_loss), "loss": avg_loss}

    return evaluate


@register_ingredient
def _perplexity():
    return IngredientSpec(
        name="perplexity", kind="eval", config_cls=PerplexityConfig,
        cache_relevant=False,
        build=_build_perplexity,
    )
