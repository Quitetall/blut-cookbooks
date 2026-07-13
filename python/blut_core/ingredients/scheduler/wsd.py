"""Warmup-Stable-Decay LR scheduler (ADR 0050/0051 scheduler ingredient).

Relocated verbatim from ``lamquant/student/train_joint.py`` — it was defined in
the codec trainer and cross-imported by the SNN trainers
(``pretrain_ssl_tueg`` / ``train_4state_controller``), the worst coupling in the
training tree. It now lives in its own home; the SNN trainers import it here
instead of reaching into a sibling trainer.
"""
from __future__ import annotations

import math


class WSDScheduler:
    """Warmup-Stable-Decay scheduler for continual training.

    Three phases (user direction 2026-05-21, cosine-then-WSD):
      1. Warmup (cosine ramp): LR 0 → peak via half-cosine, smoother
         than linear at the start and end of the warmup window.
      2. Stable (constant): LR = peak, shippable any time.
      3. Decay (cosine): LR peak → min_lr over decay_epochs.

    Pass ``warmup_kind="linear"`` to restore the legacy linear ramp.

    Infinite mode (decay_frac=0):
      Stable phase runs forever. The model trains at peak LR
      indefinitely — every checkpoint is shippable. When you
      want to finalize, call trigger_decay(n_epochs) to start
      the cosine cooldown manually.

    For continual training: resume from any stable-phase checkpoint.
    No re-warming disruption since stable phase is at full LR.
    """

    def __init__(self, optimizer, total_epochs: int, peak_lr: float,
                 warmup_frac: float = 0.05, decay_frac: float = 0.10,
                 min_lr: float = 1e-6, warmup_kind: str = "cosine"):
        if warmup_kind not in ("cosine", "linear"):
            raise ValueError(f"warmup_kind must be 'cosine' or 'linear', got {warmup_kind!r}")
        self.warmup_kind = warmup_kind
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_epochs = max(1, int(total_epochs * warmup_frac))

        # decay_frac=0 → infinite stable phase (no automatic decay)
        if decay_frac <= 0:
            self.decay_epochs = 0
            self.decay_start = total_epochs + 1  # never reached
            self._infinite = True
        else:
            self.decay_epochs = max(1, int(total_epochs * decay_frac))
            self.decay_start = total_epochs - self.decay_epochs
            self._infinite = False

        self.stable_start = self.warmup_epochs
        self._configured_decay_epochs = self.decay_epochs
        self._configured_decay_start = self.decay_start
        self._last_lr = [peak_lr] * len(optimizer.param_groups)
        # Preserve each param group's relative LR at construction time (e.g.
        # an encoder group scaled to a fraction of peak_lr via
        # encoder_lr_scale before this scheduler is built) — step() applies
        # the schedule's absolute lr scaled by this per-group ratio instead
        # of overwriting every group with the identical value, which used to
        # silently discard any differential per-group rate from epoch 2
        # onward. A caller that never differentiates groups gets scale=1.0
        # everywhere (all groups start at peak_lr), so this is a no-op for
        # every existing single-rate use.
        self._group_scale = [
            (pg['lr'] / peak_lr) if peak_lr else 1.0
            for pg in optimizer.param_groups
        ]
        # Sanity-check the implicit contract this inference depends on: no
        # group should start ABOVE peak_lr unless a caller deliberately
        # wants a super-peak group. In practice this catches the far more
        # likely mistake -- passing a peak_lr that doesn't match how the
        # optimizer was actually constructed -- which would otherwise run
        # the entire schedule at a silently wrong rate with no signal.
        if peak_lr and any(s > 1.0 + 1e-6 for s in self._group_scale):
            print(
                f"[WSD] warning: a param group's initial lr exceeds "
                f"peak_lr={peak_lr:.2e} (scales={[round(s, 3) for s in self._group_scale]}) "
                f"-- if this wasn't a deliberate super-peak group, peak_lr likely "
                f"doesn't match how the optimizer was constructed"
            )
        self.epoch = 0
        self._decay_triggered = False
        self._decay_trigger_epoch = None

    def trigger_decay(self, n_epochs: int = 40):
        """Manually trigger the cosine decay phase.

        Call this when you want to finalize the model. The decay
        starts at the current epoch and runs for n_epochs.
        """
        self._decay_triggered = True
        self._decay_trigger_epoch = self.epoch
        self.decay_epochs = n_epochs
        self.decay_start = self.epoch
        print(f"[WSD] Decay triggered at epoch {self.epoch}, "
              f"will decay over {n_epochs} epochs to lr={self.min_lr:.1e}")

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            # progress in [0, 1] across warmup window
            p = self.epoch / max(self.warmup_epochs, 1)
            if self.warmup_kind == "cosine":
                # Half-cosine ramp: 0 → peak via 0.5*(1 - cos(pi*p)).
                # Same start/end values as linear but smoother derivative
                # at both edges (avoids the optimizer-state shock that
                # a sharp linear corner can trigger right at peak LR).
                lr = self.peak_lr * 0.5 * (1.0 - math.cos(math.pi * p))
            else:  # "linear"
                lr = self.peak_lr * p
        elif not self._decay_triggered and self._infinite:
            # Infinite stable — runs forever at peak LR
            lr = self.peak_lr
        elif self.epoch < self.decay_start:
            lr = self.peak_lr
        else:
            # Cosine decay
            progress = (self.epoch - self.decay_start) / max(self.decay_epochs, 1)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self._last_lr = []
        for pg, scale in zip(self.optimizer.param_groups, self._group_scale):
            pg['lr'] = lr * scale
            self._last_lr.append(pg['lr'])

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self) -> dict:
        """Return all mutable scheduler state needed for exact resume.

        Configuration values ride with the state so loading into an
        incompatible scheduler fails instead of silently changing the LR
        trajectory. The optimizer itself remains responsible for its own state.
        """
        return {
            "version": 1,
            "epoch": self.epoch,
            "last_lr": list(self._last_lr),
            "group_scale": list(self._group_scale),
            "decay_triggered": self._decay_triggered,
            "decay_trigger_epoch": self._decay_trigger_epoch,
            "decay_epochs": self.decay_epochs,
            "decay_start": self.decay_start,
            "infinite": self._infinite,
            "config": {
                "total_epochs": self.total_epochs,
                "peak_lr": self.peak_lr,
                "min_lr": self.min_lr,
                "warmup_epochs": self.warmup_epochs,
                "warmup_kind": self.warmup_kind,
                "configured_decay_epochs": self._configured_decay_epochs,
                "configured_decay_start": self._configured_decay_start,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler progress and optimizer group learning rates.

        Raises ``ValueError`` for malformed or incompatible state. Resume must
        never fall back to a fresh warmup schedule without an explicit caller
        decision.
        """
        if not isinstance(state, dict):
            raise ValueError("WSD scheduler state must be a mapping")
        if state.get("version") != 1:
            raise ValueError(
                f"unsupported WSD scheduler state version: {state.get('version')!r}"
            )

        expected_config = self.state_dict()["config"]
        if state.get("config") != expected_config:
            raise ValueError(
                "WSD scheduler state is incompatible with current configuration"
            )

        last_lr = state.get("last_lr")
        group_scale = state.get("group_scale")
        n_groups = len(self.optimizer.param_groups)
        if not isinstance(last_lr, list) or len(last_lr) != n_groups:
            raise ValueError(
                f"WSD scheduler state has {len(last_lr) if isinstance(last_lr, list) else 'invalid'} "
                f"learning rates for {n_groups} optimizer groups"
            )
        if group_scale != self._group_scale:
            raise ValueError("WSD scheduler optimizer group scales do not match")

        epoch = state.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"invalid WSD scheduler epoch: {epoch!r}")
        decay_epochs = state.get("decay_epochs")
        decay_start = state.get("decay_start")
        if not isinstance(decay_epochs, int) or decay_epochs < 0:
            raise ValueError(f"invalid WSD decay_epochs: {decay_epochs!r}")
        if not isinstance(decay_start, int):
            raise ValueError(f"invalid WSD decay_start: {decay_start!r}")
        if state.get("infinite") is not self._infinite:
            raise ValueError("WSD scheduler infinite-mode setting does not match")
        if not all(isinstance(lr, (int, float)) and math.isfinite(lr) for lr in last_lr):
            raise ValueError("WSD scheduler state contains a non-finite learning rate")
        decay_triggered = state.get("decay_triggered")
        decay_trigger_epoch = state.get("decay_trigger_epoch")
        if not isinstance(decay_triggered, bool):
            raise ValueError("WSD decay_triggered must be boolean")
        if decay_triggered:
            if not isinstance(decay_trigger_epoch, int) or decay_trigger_epoch < 0:
                raise ValueError(
                    f"invalid WSD decay_trigger_epoch: {decay_trigger_epoch!r}"
                )
        elif decay_trigger_epoch is not None:
            raise ValueError("WSD decay_trigger_epoch set before decay was triggered")

        self.epoch = epoch
        self._last_lr = list(last_lr)
        self._decay_triggered = decay_triggered
        self._decay_trigger_epoch = decay_trigger_epoch
        self.decay_epochs = decay_epochs
        self.decay_start = decay_start
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr

    @property
    def phase(self) -> str:
        if self.epoch <= self.warmup_epochs:
            return 'warmup'
        elif not self._decay_triggered and self._infinite:
            return 'stable∞'
        elif self.epoch < self.decay_start:
            return 'stable'
        else:
            return 'decay'
