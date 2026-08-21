#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified model factory for experiment scripts."""

from __future__ import annotations

import torch

from utils.mainline_contract import ACTIVE_PLGAFORMER_FLAGS

from .baseline_models import create_baseline_model
from .public_baselines import (
    TSLIB_ADAPTER_DEFAULTS,
    TSLibForecastAdapter,
)
from .plgaformer import PLGAFormerTransformer


ACTIVE_MODEL_TYPES = (
    "transformer",
    "kinematic",
    "rotating_3dof",
    "dlinear",
    "plgaformer",
    "patchtst",
    "itransformer",
)


def get_supported_model_types() -> tuple[str, ...]:
    """Return the exact paper-mainline model registry."""
    return ACTIVE_MODEL_TYPES


def _create_tslib_model(
    model_type: str,
    *,
    input_dim: int,
    device: torch.device | str,
    source_kwargs: dict,
):
    """Construct one pinned TSLib model through the HGV adapter."""
    from . import HGVConfig  # delayed import to avoid circular dependency

    train_config = HGVConfig.get_train_config()
    model_config = HGVConfig.get_model_config(model_type)
    adapter_config = dict(TSLIB_ADAPTER_DEFAULTS[model_type])
    return TSLibForecastAdapter(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=model_config.get("output_dim", 3),
        seq_len=train_config.get("seq_len", 256),
        pred_len=train_config.get("pred_len", 256),
        d_model=adapter_config["d_model"],
        n_heads=adapter_config["n_heads"],
        e_layers=adapter_config["e_layers"],
        d_layers=adapter_config["d_layers"],
        d_ff=adapter_config["d_ff"],
        factor=adapter_config["factor"],
        dropout=adapter_config["dropout"],
        patch_len=model_config.get("patch_len", 16),
        stride=model_config.get("stride", 8),
        moving_avg=model_config.get("moving_avg", 25),
        source_root=source_kwargs.get("tslib_root"),
    ).to(device)


def create_registered_model(
    model_type: str,
    input_dim: int = 6,
    device: torch.device | str = torch.device("cpu"),
    plgaformer_kwargs: dict | None = None,
):
    """
    Create model by normalized model_type.

    Args:
        model_type: Registered model type (case-insensitive).
        input_dim: Input feature dimension.
        device: Target device.
        plgaformer_kwargs: Optional overrides for PLGAFormer constructor.
    """
    mt = model_type.lower().strip()
    supported = get_supported_model_types()
    if mt not in supported:
        raise ValueError(
            f"Unsupported active model type: {mt}; supported={supported}"
        )

    if mt == "transformer":
        source_kwargs = dict(plgaformer_kwargs or {})
        if str(source_kwargs.get("source", "")).lower() == "tslib":
            return _create_tslib_model(
                "transformer",
                input_dim=input_dim,
                device=device,
                source_kwargs=source_kwargs,
            )
        return create_baseline_model("transformer", input_dim=input_dim, device=device)
    if mt == "kinematic":
        return create_baseline_model(
            "kinematic",
            input_dim=input_dim,
            device=device,
            **(plgaformer_kwargs or {}),
        )
    if mt == "rotating_3dof":
        return create_baseline_model(
            "rotating_3dof",
            input_dim=input_dim,
            device=device,
            **(plgaformer_kwargs or {}),
        )
    if mt == "plgaformer":
        from . import HGVConfig  # delayed import to avoid circular dependency
        model_config = HGVConfig.get_model_config("plgaformer")
        kwargs = {
            "input_dim": input_dim,
            "d_model": model_config.get("d_model", 256),
            "nhead": model_config.get("nhead", 8),
            "num_encoder_layers": model_config.get("num_encoder_layers", 3),
            "num_decoder_layers": model_config.get("num_decoder_layers", 2),
            "dim_feedforward": model_config.get("dim_feedforward", 1024),
            "dropout": model_config.get("dropout", 0.1),
            "output_dim": model_config.get("output_dim", 3),
            **dict(ACTIVE_PLGAFORMER_FLAGS),
        }
        if plgaformer_kwargs:
            kwargs.update(plgaformer_kwargs)
        return PLGAFormerTransformer(**kwargs).to(device)
    return _create_tslib_model(
        mt,
        input_dim=input_dim,
        device=device,
        source_kwargs=dict(plgaformer_kwargs or {}),
    )
