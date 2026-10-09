"""Generic loss ingredient specs.

Provides common loss functions: cross-entropy, causal-LM next-token
cross-entropy, MSE, contrastive, and DPO.
Domain cookbooks override with domain-specific losses (e.g. LamQuant's
joint codec loss).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- Cross-entropy ------------------------------------------------------
@dataclass(frozen=True)
class CrossEntropyConfig:
    label_smoothing: float = 0.0
    ignore_index: int = -100
    reduction: str = "mean"


def _build_cross_entropy(cfg: CrossEntropyConfig):
    import torch.nn.functional as F

    def loss_fn(logits, targets):
        if hasattr(logits, "logits"):
            # A HuggingFace model output, not a logits tensor. Say which loss
            # was meant instead of letting F.cross_entropy reject the type.
            raise TypeError(
                "cross_entropy expects a logits tensor but got a "
                f"{type(logits).__name__}; for a HuggingFace causal LM use the "
                "'causal_lm' loss, which trains on next-token prediction")
        return F.cross_entropy(
            logits, targets,
            label_smoothing=cfg.label_smoothing,
            ignore_index=cfg.ignore_index,
            reduction=cfg.reduction,
        )

    return loss_fn


@register_ingredient
def _cross_entropy():
    return IngredientSpec(
        name="cross_entropy", kind="loss", config_cls=CrossEntropyConfig,
        cache_relevant=True,
        build=_build_cross_entropy,
    )


# ---- Causal LM (next-token prediction) ---------------------------------
@dataclass(frozen=True)
class CausalLmConfig:
    ignore_index: int = -100


def _build_causal_lm(cfg: CausalLmConfig):
    import torch.nn.functional as F

    def loss_fn(output, batch):
        """Next-token cross-entropy for a causal LM.

        A HuggingFace model given `labels` already returns this loss as
        `output.loss`, computed with the one-token shift; use it. Otherwise
        shift here: position t predicts token t + 1.
        """
        loss = getattr(output, "loss", None)
        if loss is not None:
            return loss
        logits = getattr(output, "logits", output)
        labels = batch["labels"]
        shifted = logits[..., :-1, :].contiguous()
        targets = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shifted.view(-1, shifted.size(-1)),
            targets.view(-1),
            ignore_index=cfg.ignore_index,
        )

    return loss_fn


@register_ingredient
def _causal_lm():
    return IngredientSpec(
        name="causal_lm", kind="loss", config_cls=CausalLmConfig,
        cache_relevant=True,
        build=_build_causal_lm,
    )


# ---- MSE ----------------------------------------------------------------
@dataclass(frozen=True)
class MseConfig:
    reduction: str = "mean"


def _build_mse(cfg: MseConfig):
    import torch.nn.functional as F

    def loss_fn(pred, target):
        return F.mse_loss(pred, target, reduction=cfg.reduction)

    return loss_fn


@register_ingredient
def _mse():
    return IngredientSpec(
        name="mse", kind="loss", config_cls=MseConfig,
        cache_relevant=True,
        build=_build_mse,
    )


# ---- Contrastive (InfoNCE / NT-Xent) -----------------------------------
@dataclass(frozen=True)
class ContrastiveConfig:
    temperature: float = 0.07


def _build_contrastive(cfg: ContrastiveConfig):
    import torch
    import torch.nn.functional as F

    def loss_fn(z_i, z_j):
        """NT-Xent loss. z_i, z_j are (N, D) tensors of paired embeddings."""
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        sim = z @ z.T / cfg.temperature  # (2N, 2N)
        # Mask out self-similarity
        mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)
        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])
        sim = sim.masked_select(mask).view(2 * N, -1)
        return F.cross_entropy(sim, labels)

    return loss_fn


@register_ingredient
def _contrastive():
    return IngredientSpec(
        name="contrastive", kind="loss", config_cls=ContrastiveConfig,
        cache_relevant=True,
        build=_build_contrastive,
    )


# ---- DPO (Direct Preference Optimization) ------------------------------
@dataclass(frozen=True)
class DpoConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0


def _build_dpo(cfg: DpoConfig):
    import torch
    import torch.nn.functional as F

    def loss_fn(policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps):
        """DPO loss. All inputs are log-probabilities (N,)."""
        logits = (policy_chosen_logps - reference_chosen_logps) - \
                 (policy_rejected_logps - reference_rejected_logps)
        if cfg.label_smoothing > 0:
            loss = -F.logsigmoid(cfg.beta * logits) * (1 - cfg.label_smoothing) \
                   - F.logsigmoid(-cfg.beta * logits) * cfg.label_smoothing
        else:
            loss = -F.logsigmoid(cfg.beta * logits)
        return loss.mean()

    return loss_fn


@register_ingredient
def _dpo():
    return IngredientSpec(
        name="dpo", kind="loss", config_cls=DpoConfig,
        cache_relevant=True,
        build=_build_dpo,
    )
