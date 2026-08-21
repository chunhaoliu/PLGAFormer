#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified model factory for experiment scripts."""

from __future__ import annotations

import torch

from utils.mainline_contract import ACTIVE_PLGAFORMER_FLAGS

from .baseline_models import create_baseline_model
from .sota_models import create_sota_model
from .PIT import create_pit_model
from .external_baselines import AFCILNExternalAdapter
from .public_baselines import (
    PUBLIC_TSLIB_MODEL_TYPES,
    TSLIB_ADAPTER_DEFAULTS,
    TSLibForecastAdapter,
)
from .plgaformer import PLGAFormerTransformer


def get_supported_model_types() -> tuple[str, ...]:
    """Return supported model type names for experiments."""
    return (
        "baseline",      # alias of standard transformer baseline
        "transformer",   # standard transformer baseline
        "kalman",        # fixed-parameter kalman baseline
        "kinematic",     # frozen-state analytical spherical kinematics
        "rotating_3dof", # identified rotating-Earth 3-DOF RK4 propagation
        "dlinear",       # decomposition-linear forecasting baseline
        "pit",           # physics-informed transformer
        "plgaformer",    # proposed model
        "af_ciln",       # exact external AF-CILN checkout (no source redistribution)
        "autoformer",
        "informer",
        "patchtst",
        "fedformer",
        "timesnet",
        "itransformer",
    )


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
    mt = model_type.lower()

    if mt in ("baseline", "transformer"):
        source_kwargs = dict(plgaformer_kwargs or {})
        if str(source_kwargs.get("source", "")).lower() == "tslib":
            return _create_tslib_model(
                "transformer",
                input_dim=input_dim,
                device=device,
                source_kwargs=source_kwargs,
            )
        return create_baseline_model("transformer", input_dim=input_dim, device=device)
    if mt == "kalman":
        return create_baseline_model("kalman", input_dim=input_dim, device=device)
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
    if mt == "dlinear":
        source_kwargs = dict(plgaformer_kwargs or {})
        if str(source_kwargs.get("source", "")).lower() == "tslib":
            return _create_tslib_model(
                "dlinear",
                input_dim=input_dim,
                device=device,
                source_kwargs=source_kwargs,
            )
        return create_baseline_model("dlinear", input_dim=input_dim, device=device)
    if mt == "pit":
        return create_pit_model(input_dim=input_dim, device=device)
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
    if mt == "af_ciln":
        from . import HGVConfig  # delayed import to avoid circular dependency

        if not plgaformer_kwargs:
            raise ValueError("af_ciln requires input/output scaler metadata")
        train_config = HGVConfig.get_train_config()
        model_config = HGVConfig.get_model_config("af_ciln")
        scaler_keys = {
            "input_scaler_mean",
            "input_scaler_scale",
            "output_scaler_mean",
            "output_scaler_scale",
        }
        scaler_kwargs = {
            key: value for key, value in plgaformer_kwargs.items() if key in scaler_keys
        }
        return AFCILNExternalAdapter(
            input_dim=input_dim,
            seq_len=train_config.get("seq_len", 256),
            pred_len=train_config.get("pred_len", 256),
            output_dim=model_config.get("output_dim", 3),
            d_model=model_config.get("d_model", 256),
            dropout=model_config.get("dropout", 0.1),
            **scaler_kwargs,
        ).to(device)

    # SOTA models
    if mt in PUBLIC_TSLIB_MODEL_TYPES:
        source_kwargs = dict(plgaformer_kwargs or {})
        if str(source_kwargs.get("source", "")).lower() == "tslib":
            return _create_tslib_model(
                mt,
                input_dim=input_dim,
                device=device,
                source_kwargs=source_kwargs,
            )
    return create_sota_model(mt, input_dim=input_dim, device=device)
