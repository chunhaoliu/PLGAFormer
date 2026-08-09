#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared training protocol helpers for experiment scripts."""

from __future__ import annotations

import math
import torch


def build_adamw_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """Create a standardized AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build linear-warmup + cosine-decay scheduler.

    Warmup: lr ratio from 0.1 -> 1.0
    Cosine: lr ratio from 1.0 -> min_lr_ratio
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def should_update_best(current: float, best: float, min_delta: float = 1e-6) -> bool:
    """Return True if current loss improves best loss by at least min_delta."""
    return current < (best - min_delta)


def build_optimizer_by_profile(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    profile: str = "enhanced",
) -> torch.optim.Optimizer:
    """
    Build optimizer by protocol profile.

    - enhanced: AdamW (current project default)
    - official_like: Adam (closer to Autoformer/FEDformer official scripts)
    """
    p = profile.lower().strip()
    if p == "official_like":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    return build_adamw_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler_by_profile(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    profile: str = "enhanced",
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build learning-rate scheduler by protocol profile.

    - enhanced: warmup + cosine
    - official_like: Autoformer/FEDformer-style epoch-step decay
    """
    p = profile.lower().strip()
    if p == "official_like":
        def lr_lambda(epoch: int) -> float:
            # type1: lr *= 0.5 every epoch (epoch starts from 0)
            return 0.5 ** max(0, epoch)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
    )


def clip_gradients_with_finite_check(
    model,
    max_norm: float | None,
    *,
    context: str,
) -> torch.Tensor:
    """Check all gradients with one aggregate reduction and optionally clip.

    This replaces per-parameter ``isnan().any()`` Python branches, which force
    a CUDA synchronization for every parameter tensor.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise FloatingPointError(f"No gradients were produced: {context}.")
    effective_max_norm = float(max_norm) if max_norm is not None else float("inf")
    try:
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=effective_max_norm,
            error_if_nonfinite=True,
            foreach=True,
        )
    except RuntimeError as exc:
        raise FloatingPointError(f"Non-finite gradient norm: {context}.") from exc
    return total_norm.detach()
