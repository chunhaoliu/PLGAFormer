#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model provenance registry for reproducible comparison reporting."""

from __future__ import annotations

from typing import Any

MODEL_PROVENANCE: dict[str, dict[str, Any]] = {
    "transformer": {
        "source_type": "internal_baseline",
        "implementation_level": "simplified",
        "reference_family": "vanilla_transformer",
    },
    "baseline": {
        "source_type": "internal_baseline",
        "implementation_level": "standard_transformer_alias",
        "reference_family": "vanilla_transformer",
    },
    "plgaformer": {
        "source_type": "proposed_model",
        "implementation_level": "project_primary",
        "reference_family": "plgaformer",
    },
    "af_ciln": {
        "source_type": "official_external_checkout",
        "implementation_level": "exact_release_with_scaling_adapter",
        "reference_family": "af_ciln",
        "upstream_repository": "https://github.com/WEAPONCHUA/AF-CILN",
        "upstream_commit": "ee39368edf833866aa472ea0ef024ebb28e92557",
        "license_status": "no_license_file_in_upstream_checkout",
        "redistributed": False,
    },
    "pit": {
        "source_type": "reimplemented",
        "implementation_level": "faithful_min",
        "reference_family": "pit_hgv",
    },
    "kalman": {
        "source_type": "internal_baseline",
        "implementation_level": "classical_filter",
        "reference_family": "kalman_filter",
    },
    "kinematic": {
        "source_type": "analytical_baseline",
        "implementation_level": "parameter_free_frozen_state",
        "reference_family": "spherical_kinematics",
    },
    "rotating_3dof": {
        "source_type": "analytical_numerical_baseline",
        "implementation_level": "parameter_free_identified_rotating_earth_3dof",
        "reference_family": "rotating_spherical_earth_point_mass",
    },
    "dlinear": {
        "source_type": "reimplemented",
        "implementation_level": "architecture_faithful",
        "reference_family": "dlinear",
    },
    "informer": {
        "source_type": "reimplemented",
        "implementation_level": "lightweight_proxy_not_formal",
        "reference_family": "informer",
    },
    "autoformer": {
        "source_type": "reimplemented",
        "implementation_level": "lightweight_proxy_not_formal",
        "reference_family": "autoformer",
    },
    "patchtst": {
        "source_type": "reimplemented",
        "implementation_level": "architecture_faithful",
        "reference_family": "patchtst",
    },
    "fedformer": {
        "source_type": "reimplemented",
        "implementation_level": "lightweight_proxy_not_formal",
        "reference_family": "fedformer",
    },
    "timesnet": {
        "source_type": "reimplemented",
        "implementation_level": "lightweight_proxy_not_formal",
        "reference_family": "timesnet",
    },
    "itransformer": {
        "source_type": "reimplemented",
        "implementation_level": "architecture_faithful",
        "reference_family": "itransformer",
    },
}


def get_model_provenance(model_type: str) -> dict[str, Any]:
    """Return provenance metadata for model_type."""
    key = str(model_type).lower().strip()
    return MODEL_PROVENANCE.get(
        key,
        {
            "source_type": "unknown",
            "implementation_level": "unknown",
            "reference_family": key or "unknown",
        },
    )


def build_comparison_model_provenance(comparison_models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build provenance mapping keyed by display model name."""
    return {
        model_name: get_model_provenance(model_cfg.get("model_type", ""))
        for model_name, model_cfg in comparison_models.items()
    }


def validate_comparison_model_types(comparison_models: dict[str, dict[str, Any]]) -> list[str]:
    """Return unknown model types appearing in comparison settings."""
    unknown = []
    for _, model_cfg in comparison_models.items():
        model_type = str(model_cfg.get("model_type", "")).lower().strip()
        if not model_type:
            unknown.append("<empty>")
            continue
        if model_type not in MODEL_PROVENANCE:
            unknown.append(model_type)
    return sorted(set(unknown))
