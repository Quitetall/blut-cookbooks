"""Generic checkpoint ingredient specs.

Provides atomic save, durable resume, and best-K checkpoint management.
These are extracted from LamQuant's checkpoint ingredients — they have zero
domain knowledge and work for any ML training task.
"""
from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- atomic_save --------------------------------------------------------
_SAVE_EXECUTOR: Optional[ThreadPoolExecutor] = None
_SAVE_LOCK = threading.Lock()


def _ensure_save_executor() -> ThreadPoolExecutor:
    global _SAVE_EXECUTOR
    with _SAVE_LOCK:
        if _SAVE_EXECUTOR is None:
            _SAVE_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ckpt-atomic")
            atexit.register(_SAVE_EXECUTOR.shutdown, wait=True)
    return _SAVE_EXECUTOR


def _state_dict_to_cpu(sd):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v
    return out


def _atomic_torch_save(payload: dict, path: str) -> None:
    """torch.save to a temp file in the same dir, then atomic rename."""
    import torch
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _async_save(payload: dict, path: str):
    return _ensure_save_executor().submit(_atomic_torch_save, payload, path)


@dataclass(frozen=True)
class AtomicSaveConfig:
    async_: bool = False
    state_dict_to_cpu: bool = False


def _build_atomic_save(cfg: AtomicSaveConfig):
    """Return ``save(payload: dict, path: str) -> None``."""
    def save(payload: dict, path: str) -> None:
        if cfg.state_dict_to_cpu and isinstance(payload, dict) \
                and "state_dict" in payload:
            payload = dict(payload)
            payload["state_dict"] = _state_dict_to_cpu(payload["state_dict"])
        if cfg.async_:
            _async_save(payload, path)
        else:
            _atomic_torch_save(payload, path)
    return save


@register_ingredient
def _atomic_save_spec():
    return IngredientSpec(
        name="atomic_save", kind="checkpoint", config_cls=AtomicSaveConfig,
        cache_relevant=False,
        build=_build_atomic_save,
    )


# ---- durable_resume -----------------------------------------------------
@dataclass(frozen=True)
class DurableResumeConfig:
    pass


def _build_durable_resume(cfg: DurableResumeConfig, *, resume_dir,
                          run_id="", resume_key=""):
    """Return a DurableResume or None when resume_dir is falsy."""
    # Import from the blut engine's durable_resume module if available,
    # otherwise provide a minimal implementation.
    try:
        from blut.durable_resume import DurableResume
        return DurableResume(resume_dir, run_id, resume_key) if resume_dir else None
    except ImportError:
        # Minimal fallback: just return the resume_dir path for the trainer to handle
        return resume_dir if resume_dir else None


@register_ingredient
def _durable_resume_spec():
    return IngredientSpec(
        name="durable_resume", kind="checkpoint",
        config_cls=DurableResumeConfig,
        cache_relevant=False,
        build=_build_durable_resume,
    )


# ---- best_k (keep top-K checkpoints by metric) -------------------------
@dataclass(frozen=True)
class BestKConfig:
    k: int = 3
    metric: str = "val_loss"
    mode: str = "min"           # "min" or "max"


def _build_best_k(cfg: BestKConfig):
    """Return a BestKTracker that keeps top-K checkpoints by metric."""
    import heapq

    class BestKTracker:
        def __init__(self, k, metric, mode):
            self.k = k
            self.metric = metric
            self.mode = mode
            self.heap = []  # min-heap of (score, path)
            self._sign = 1 if mode == "min" else -1

        def update(self, metrics: dict, save_fn, payload: dict, path: str):
            score = metrics.get(self.metric)
            if score is None:
                return
            # Save if we have room or if this is better than the worst
            if len(self.heap) < self.k:
                save_fn(payload, path)
                heapq.heappush(self.heap, (self._sign * score, path))
            elif self._sign * score < self.heap[0][0]:
                # Better than the worst — evict worst, save this
                _, old_path = heapq.heappushpop(
                    self.heap, (self._sign * score, path))
                if os.path.exists(old_path):
                    os.remove(old_path)
                save_fn(payload, path)

        def best_path(self):
            if not self.heap:
                return None
            return max(self.heap, key=lambda x: x[0])[1]

    return BestKTracker(cfg.k, cfg.metric, cfg.mode)


@register_ingredient
def _best_k_spec():
    return IngredientSpec(
        name="best_k", kind="checkpoint", config_cls=BestKConfig,
        cache_relevant=False,
        build=_build_best_k,
    )
