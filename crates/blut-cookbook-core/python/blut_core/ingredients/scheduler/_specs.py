"""Generic scheduler ingredient specs.

Provides common LR schedulers: cosine, linear, constant, and WSD.
Domain cookbooks can override or extend with domain-specific schedulers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- Cosine annealing with warmup --------------------------------------
@dataclass(frozen=True)
class CosineConfig:
    total_epochs: int
    warmup_epochs: int = 5
    min_lr: float = 0.0
    warmup_start_lr: float = 0.0


def _build_cosine(cfg: CosineConfig, optimizer):
    """Return a cosine annealing scheduler with linear warmup."""
    import math

    class CosineWarmupScheduler:
        def __init__(self, opt, total_epochs, warmup_epochs, min_lr, warmup_start_lr):
            self.optimizer = opt
            self.total_epochs = total_epochs
            self.warmup_epochs = warmup_epochs
            self.min_lr = min_lr
            self.warmup_start_lr = warmup_start_lr
            self.epoch = 0
            self._last_lr = [pg['lr'] for pg in opt.param_groups]
            self._peak_lr = opt.param_groups[0]['lr']

        def step(self, epoch=None):
            if epoch is not None:
                self.epoch = epoch
            else:
                self.epoch += 1
            if self.epoch <= self.warmup_epochs:
                p = self.epoch / max(self.warmup_epochs, 1)
                lr = self.warmup_start_lr + (self._peak_lr - self.warmup_start_lr) * p
            else:
                progress = (self.epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                progress = min(progress, 1.0)
                lr = self.min_lr + 0.5 * (self._peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            self._last_lr = [lr] * len(self.optimizer.param_groups)

        def get_last_lr(self):
            return self._last_lr

    return CosineWarmupScheduler(
        optimizer, cfg.total_epochs, cfg.warmup_epochs,
        cfg.min_lr, cfg.warmup_start_lr)


@register_ingredient
def _cosine():
    return IngredientSpec(
        name="cosine", kind="scheduler", config_cls=CosineConfig,
        build=_build_cosine,
    )


# ---- Linear warmup then linear decay ------------------------------------
@dataclass(frozen=True)
class LinearConfig:
    total_epochs: int
    warmup_epochs: int = 5
    min_lr: float = 0.0
    warmup_start_lr: float = 0.0


def _build_linear(cfg: LinearConfig, optimizer):
    import math

    class LinearWarmupDecayScheduler:
        def __init__(self, opt, total_epochs, warmup_epochs, min_lr, warmup_start_lr):
            self.optimizer = opt
            self.total_epochs = total_epochs
            self.warmup_epochs = warmup_epochs
            self.min_lr = min_lr
            self.warmup_start_lr = warmup_start_lr
            self.epoch = 0
            self._last_lr = [pg['lr'] for pg in opt.param_groups]
            self._peak_lr = opt.param_groups[0]['lr']

        def step(self, epoch=None):
            if epoch is not None:
                self.epoch = epoch
            else:
                self.epoch += 1
            if self.epoch <= self.warmup_epochs:
                p = self.epoch / max(self.warmup_epochs, 1)
                lr = self.warmup_start_lr + (self._peak_lr - self.warmup_start_lr) * p
            else:
                progress = (self.epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                progress = min(progress, 1.0)
                lr = self._peak_lr + (self.min_lr - self._peak_lr) * progress
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            self._last_lr = [lr] * len(self.optimizer.param_groups)

        def get_last_lr(self):
            return self._last_lr

    return LinearWarmupDecayScheduler(
        optimizer, cfg.total_epochs, cfg.warmup_epochs,
        cfg.min_lr, cfg.warmup_start_lr)


@register_ingredient
def _linear():
    return IngredientSpec(
        name="linear", kind="scheduler", config_cls=LinearConfig,
        build=_build_linear,
    )


# ---- Constant LR with warmup --------------------------------------------
@dataclass(frozen=True)
class ConstantConfig:
    warmup_epochs: int = 5
    warmup_start_lr: float = 0.0


def _build_constant(cfg: ConstantConfig, optimizer):
    class ConstantWithWarmupScheduler:
        def __init__(self, opt, warmup_epochs, warmup_start_lr):
            self.optimizer = opt
            self.warmup_epochs = warmup_epochs
            self.warmup_start_lr = warmup_start_lr
            self.epoch = 0
            self._last_lr = [pg['lr'] for pg in opt.param_groups]
            self._peak_lr = opt.param_groups[0]['lr']

        def step(self, epoch=None):
            if epoch is not None:
                self.epoch = epoch
            else:
                self.epoch += 1
            if self.epoch <= self.warmup_epochs:
                p = self.epoch / max(self.warmup_epochs, 1)
                lr = self.warmup_start_lr + (self._peak_lr - self.warmup_start_lr) * p
            else:
                lr = self._peak_lr
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            self._last_lr = [lr] * len(self.optimizer.param_groups)

        def get_last_lr(self):
            return self._last_lr

    return ConstantWithWarmupScheduler(optimizer, cfg.warmup_epochs, cfg.warmup_start_lr)


@register_ingredient
def _constant():
    return IngredientSpec(
        name="constant", kind="scheduler", config_cls=ConstantConfig,
        build=_build_constant,
    )


# ---- WSD (Warmup-Stable-Decay) -----------------------------------------
@dataclass(frozen=True)
class WsdConfig:
    total_epochs: int
    peak_lr: float
    warmup_frac: float = 0.05
    decay_frac: float = 0.10
    min_lr: float = 1e-6
    warmup_kind: str = "cosine"


def _build_wsd(cfg: WsdConfig, optimizer):
    from blut_core.ingredients.scheduler.wsd import WSDScheduler
    return WSDScheduler(
        optimizer, total_epochs=cfg.total_epochs, peak_lr=cfg.peak_lr,
        warmup_frac=cfg.warmup_frac, decay_frac=cfg.decay_frac,
        min_lr=cfg.min_lr, warmup_kind=cfg.warmup_kind)


@register_ingredient
def _wsd():
    return IngredientSpec(
        name="wsd", kind="scheduler", config_cls=WsdConfig,
        build=_build_wsd,
    )
