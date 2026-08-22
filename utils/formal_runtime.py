from __future__ import annotations

from argparse import Namespace
import math
from pathlib import Path
from typing import Any

from utils.mainline_contract import ACTIVE_CONFIG_PATH, PROJECT_ROOT


# These shared defaults are the scientific/runtime contract of both formal unit
# runners. The canonical config path is enforced. Subset ratio is bounded, and
# final runs require exactly 1.0. Dataset/source/output paths, model selection,
# phase, selection/skip behavior, and legacy compatibility switches remain
# operational controls. The autocast dtype is outside the contract because AMP
# is frozen off and the central config contains no mixed-precision dtype value.
# Seed requests preserve caller order and may contain surrounding whitespace;
# empty tokens, duplicates, malformed integers, and non-contract seeds fail.


def parse_prediction_horizons(value: str, fallback: int = 64) -> list[int]:
    """Parse a comma-separated horizon list into sorted unique positive ints."""
    horizons = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        horizon = int(item)
        if horizon <= 0:
            raise ValueError(f"Prediction horizon must be positive: {horizon}")
        horizons.append(horizon)
    if not horizons:
        horizons = [int(fallback)]
    return sorted(set(horizons))


def _require_exact_type(
    section: dict[str, Any], field: str, expected_type: type
) -> Any:
    value = section[field]
    if type(value) is not expected_type:
        raise TypeError(
            f"{field} must have frozen config type {expected_type.__name__}; "
            f"got {type(value).__name__}"
        )
    return value


def _require_finite_real(section: dict[str, Any], field: str) -> float:
    value = section[field]
    if type(value) not in (int, float):
        raise TypeError(
            f"{field} must have frozen config type int or float; "
            f"got {type(value).__name__}"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return normalized


def _validate_frozen_config_types(config: dict[str, Any]) -> None:
    task = config["task"]
    training = config["training"]
    for field in (
        "input_dim",
        "output_dim",
        "seq_len",
        "pred_len",
        "label_len",
        "training_origins",
        "evaluation_stride",
        "window_stride",
    ):
        _require_exact_type(task, field, int)
    for field in (
        "target_space",
        "train_supervision_protocol",
        "eval_protocol",
        "eval_ar_seed_mode",
    ):
        _require_exact_type(task, field, str)
    horizons = _require_exact_type(task, "reporting_horizons", list)
    if not horizons or any(type(value) is not int for value in horizons):
        raise TypeError("reporting_horizons must be a non-empty list of int")

    for field in (
        "epochs_max",
        "batch_size",
        "patience",
        "warmup_epochs",
        "workers",
        "physics_prior_cache_batch_size",
    ):
        _require_exact_type(training, field, int)
    for field in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
    ):
        _require_finite_real(training, field)
    for field in ("precision", "test_policy"):
        _require_exact_type(training, field, str)
    for field in ("amp", "cache_physics_prior"):
        _require_exact_type(training, field, bool)
    seeds = _require_exact_type(training, "seeds", list)
    if not seeds or any(type(value) is not int for value in seeds):
        raise TypeError("seeds must be a non-empty list of int")


def formal_runtime_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Map the frozen task/training contract to formal-runner argument names."""
    _validate_frozen_config_types(config)
    task = config["task"]
    training = config["training"]
    precision = str(training["precision"])
    amp = training["amp"]
    if precision != "float32" or amp is not False:
        raise ValueError(
            "Frozen formal precision requires training.precision='float32' "
            "and training.amp=False."
        )
    return {
        "seeds": ",".join(str(value) for value in training["seeds"]),
        "epochs": training["epochs_max"],
        "batch_size": training["batch_size"],
        "prediction_length": task["pred_len"],
        "prediction_horizons": ",".join(
            str(value) for value in task["reporting_horizons"]
        ),
        "train_supervision_protocol": task["train_supervision_protocol"],
        "eval_protocol": task["eval_protocol"],
        "eval_ar_seed_mode": task["eval_ar_seed_mode"],
        "label_len": task["label_len"],
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "warmup_epochs": training["warmup_epochs"],
        "patience": training["patience"],
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "workers": training["workers"],
        "amp": amp,
        "cache_physics_prior": training["cache_physics_prior"],
        "physics_prior_cache_batch_size": training[
            "physics_prior_cache_batch_size"
        ],
    }


def validate_formal_runtime(args: Namespace, config: dict[str, Any]) -> None:
    """Reject formal runtime values that drift from the frozen contract."""
    raw_config_path = Path(str(getattr(args, "config", ""))).expanduser()
    if not raw_config_path.is_absolute():
        raw_config_path = PROJECT_ROOT / raw_config_path
    resolved_config_path = raw_config_path.resolve()
    if resolved_config_path != ACTIVE_CONFIG_PATH.resolve():
        raise ValueError(
            f"config must resolve to frozen {ACTIVE_CONFIG_PATH}; "
            f"got {resolved_config_path}"
        )

    selection_only = getattr(args, "selection_only", None)
    if type(selection_only) is not bool:
        raise ValueError("selection_only must be bool")
    subset_ratio = getattr(args, "subset_ratio", None)
    if (
        isinstance(subset_ratio, bool)
        or not isinstance(subset_ratio, (int, float))
        or not math.isfinite(subset_ratio)
        or not 0 < subset_ratio <= 1
    ):
        raise ValueError("subset_ratio must be finite and satisfy 0 < ratio <= 1")
    if not selection_only and subset_ratio != 1.0:
        raise ValueError("subset_ratio must equal frozen value 1.0 for final runs")

    allowed = tuple(int(value) for value in config["training"]["seeds"])
    raw_seeds = str(getattr(args, "seeds", ""))
    parts = raw_seeds.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError(f"seeds must be a non-empty subset of {allowed}")
    try:
        requested = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"seeds must be a non-empty subset of {allowed}"
        ) from exc
    if len(set(requested)) != len(requested) or any(
        seed not in allowed for seed in requested
    ):
        raise ValueError(f"seeds must be a non-empty subset of {allowed}")

    for field, expected in formal_runtime_defaults(config).items():
        if field == "seeds":
            continue
        actual = getattr(args, field, None)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"{field} conflicts with frozen value {expected!r}; got {actual!r}"
            )
