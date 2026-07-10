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
        output = model(batch)
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

    def step_fn(model, optimizer, loss_fn, batch, **kwargs):
        """Mixed precision training step with AMP autocast + GradScaler."""
        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
        scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            output = model(batch)
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

    def step_fn(model, optimizer, loss_fn, batch, **kwargs):
        """Training step with gradient accumulation."""
        step = kwargs.get("_accum_step", 0)
        if step == 0:
            optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch) if callable(loss_fn) else loss_fn
        (loss / cfg.accumulation_steps).backward()
        if (step + 1) % cfg.accumulation_steps == 0:
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        return loss

    return step_fn


@register_ingredient
def _gradient_accumulation():
    return IngredientSpec(
        name="gradient_accumulation", kind="step", config_cls=GradAccumConfig,
        cache_relevant=False,
        build=_build_grad_accum,
    )
