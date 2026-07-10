"""Generic optimizer ingredient specs (ADR 0050 / 0051).

Provides the standard optimizers any ML training needs: AdamW, SGD,
Adam8bit, SOAP, and Muon. Domain cookbooks override with domain-specific
optimizers (e.g. LamQuant's ESOAP with suffix routing).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- AdamW --------------------------------------------------------------
@dataclass(frozen=True)
class AdamwConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    fused: bool = False


@register_ingredient
def _adamw():
    return IngredientSpec(
        name="adamw", kind="optimizer", config_cls=AdamwConfig,
        build_param_groups=lambda named, cfg: [
            {"params": [q for _n, q in named if q.requires_grad]}],
        construct=lambda groups, cfg: torch.optim.AdamW(
            groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas,
            fused=cfg.fused),
    )


# ---- SGD ----------------------------------------------------------------
@dataclass(frozen=True)
class SgdConfig:
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False


@register_ingredient
def _sgd():
    return IngredientSpec(
        name="sgd", kind="optimizer", config_cls=SgdConfig,
        build_param_groups=lambda named, cfg: [
            {"params": [q for _n, q in named if q.requires_grad]}],
        construct=lambda groups, cfg: torch.optim.SGD(
            groups, lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=cfg.nesterov),
    )


# ---- Adam 8-bit (bitsandbytes) -----------------------------------------
@dataclass(frozen=True)
class Adam8bitConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@register_ingredient
def _adam8bit():
    return IngredientSpec(
        name="adam8bit", kind="optimizer", config_cls=Adam8bitConfig,
        requires=("pkg:bitsandbytes",),
        build_param_groups=lambda named, cfg: [
            {"params": [q for _n, q in named if q.requires_grad]}],
        construct=lambda groups, cfg: __import__(
            "bitsandbytes.optim").Adam8bit(
                groups, lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=cfg.betas, eps=cfg.eps),
    )


# ---- SOAP ---------------------------------------------------------------
@dataclass(frozen=True)
class SoapConfig:
    lr: float = 3e-3
    betas: tuple = (0.95, 0.95)
    shampoo_beta: float = -1.0
    eps: float = 1e-8
    weight_decay: float = 0.01
    precondition_frequency: int = 10
    max_precond_dim: int = 10000
    merge_dims: bool = False
    precondition_1d: bool = False
    correct_bias: bool = True
    cautious_wd: bool = False


def _soap_groups(named, cfg):
    return [{"params": [q for _n, q in named if q.requires_grad]}]


@register_ingredient
def _soap():
    from blut_core.ingredients.optimizer.soap_optimizer import SOAP
    return IngredientSpec(
        name="soap", kind="optimizer", config_cls=SoapConfig,
        build_param_groups=_soap_groups,
        construct=lambda groups, cfg: SOAP(
            groups, lr=cfg.lr, betas=cfg.betas, shampoo_beta=cfg.shampoo_beta,
            eps=cfg.eps, weight_decay=cfg.weight_decay,
            precondition_frequency=cfg.precondition_frequency,
            max_precond_dim=cfg.max_precond_dim, merge_dims=cfg.merge_dims,
            precondition_1d=cfg.precondition_1d, correct_bias=cfg.correct_bias,
            cautious_wd=cfg.cautious_wd),
    )


# ---- Muon ---------------------------------------------------------------
@dataclass(frozen=True)
class MuonConfig:
    lr: float = 0.02
    momentum: float = 0.95
    weight_decay: float = 0.0
    adamw_lr: float = 1e-3
    adamw_betas: tuple = (0.95, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.0


def _muon_groups(named, cfg):
    named = list(named)
    muon_p = [q for _n, q in named if q.requires_grad and q.ndim >= 2]
    adamw_p = [q for _n, q in named if q.requires_grad and q.ndim < 2]
    return [
        dict(params=muon_p, lr=cfg.lr, momentum=cfg.momentum,
             weight_decay=cfg.weight_decay, use_muon=True),
        dict(params=adamw_p, lr=cfg.adamw_lr, betas=cfg.adamw_betas,
             eps=cfg.adamw_eps, weight_decay=cfg.adamw_weight_decay,
             use_muon=False),
    ]


@register_ingredient
def _muon():
    from blut_core.ingredients.optimizer.muon_optimizer import Muon
    return IngredientSpec(
        name="muon", kind="optimizer", config_cls=MuonConfig,
        build_param_groups=_muon_groups,
        construct=lambda groups, cfg: Muon(groups),
    )
