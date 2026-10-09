"""Generic training step ingredient specs.

Provides common training step patterns: standard, mixed precision, and
gradient accumulation. Domain cookbooks override with domain-specific steps
(e.g. LamQuant's QAT step).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


def _forward(model, batch):
    """Call `model` on `batch` the way the model expects.

    HuggingFace models take their inputs as keyword tensors
    (`input_ids=..., labels=...`); every other module here takes the batch as
    one positional argument. Passing a HuggingFace model its batch dict
    positionally binds the whole dict to `input_ids`, and the embedding layer
    rejects it.
    """
    from collections.abc import Mapping

    from blut_core.lm_data import is_causal_lm

    if isinstance(batch, Mapping) and is_causal_lm(model):
        return model(**batch)
    return model(batch)


# ---- Standard step (forward + backward + clip + step) -------------------
@dataclass(frozen=True)
class StandardStepConfig:
    max_grad_norm: float = 1.0
    grad_clip_value: Optional[float] = None


def _build_standard_step(cfg: StandardStepConfig):
    import torch

    def step_fn(model, optimizer, loss_fn, batch, **kwargs):
        """Standard training step: forward, backward, clip, step."""
        optimizer.zero_grad()
        output = _forward(model, batch)
        loss = loss_fn(output, batch) if callable(loss_fn) else loss_fn
        loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if cfg.grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip_value)
        optimizer.step()
        return loss

    return step_fn


@register_ingredient
def _standard_step():
    return IngredientSpec(
        name="standard", kind="step", config_cls=StandardStepConfig,
        cache_relevant=False,
        build=_build_standard_step,
    )


# ---- Mixed precision (AMP) ---------------------------------------------
@dataclass(frozen=True)
class MixedPrecisionConfig:
    max_grad_norm: float = 1.0
    dtype: str = "float16"            # "float16" or "bfloat16"


def _build_mixed_precision(cfg: MixedPrecisionConfig):
    import torch

    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    # One scaler for the whole run. Dynamic loss scaling works by carrying the
    # scale from step to step — backing off after an overflow, growing after a
    # run of clean steps. A scaler built inside the step (as this was) restarts
    # at its initial scale every call and never adapts at all.
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    def step_fn(model, optimizer, loss_fn, batch, **kwargs):
        """Mixed precision training step with AMP autocast + GradScaler."""
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            output = _forward(model, batch)
            loss = loss_fn(output, batch) if callable(loss_fn) else loss_fn
        scaler.scale(loss).backward()
        if cfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        return loss

    return step_fn


@register_ingredient
def _mixed_precision():
    return IngredientSpec(
        name="mixed_precision", kind="step", config_cls=MixedPrecisionConfig,
        cache_relevant=False,
        build=_build_mixed_precision,
    )


# ---- Gradient accumulation ---------------------------------------------
@dataclass(frozen=True)
class GradAccumConfig:
    accumulation_steps: int = 4
    max_grad_norm: float = 1.0


def _build_grad_accum(cfg: GradAccumConfig):
    import torch

    if cfg.accumulation_steps < 1:
        raise ValueError(
            f"accumulation_steps must be at least 1, got {cfg.accumulation_steps}")
    # The micro-step count lives here, in the closure. It used to be read from
    # a `_accum_step` keyword that the trainer never passes, so it was always
    # 0: gradients were zeroed on every call and `optimizer.step()` — guarded
    # by `(0 + 1) % accumulation_steps == 0` — never ran. The model never
    # updated, and the loss curve (computed before any update) looked normal.
    micro_step = 0

    def step_fn(model, optimizer, loss_fn, batch, **kwargs):
        """Training step with gradient accumulation."""
        nonlocal micro_step
        if micro_step == 0:
            optimizer.zero_grad()
        output = _forward(model, batch)
        loss = loss_fn(output, batch) if callable(loss_fn) else loss_fn
        (loss / cfg.accumulation_steps).backward()
        micro_step += 1
        if micro_step == cfg.accumulation_steps:
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            micro_step = 0
        return loss

    return step_fn


@register_ingredient
def _gradient_accumulation():
    return IngredientSpec(
        name="gradient_accumulation", kind="step", config_cls=GradAccumConfig,
        cache_relevant=False,
        build=_build_grad_accum,
    )
