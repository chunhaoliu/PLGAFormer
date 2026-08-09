#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight helpers for cross-experiment metric aggregation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .experiment_io import ensure_dir
from .formal_evidence import (
    DISPLAY_NAMES,
    load_formal_config,
    normalize_run_record,
    resolve_study_artifact_root,
    validate_evidence_bundle,
)

PROTOCOL_COLUMNS = [
    "eval_protocol",
    "eval_ar_seed_mode",
    "train_supervision_protocol",
    "optimizer_profile",
    "strict_repro_mode",
]


UNIFIED_COLUMNS = [
    "experiment",
    "phase",
    "seed",
    *PROTOCOL_COLUMNS,
    "scenario_type",
    "scenario_value",
    "model",
    "horizon",
    "metric",
    "value",
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row(
    experiment: str,
    model: str,
    horizon: str | int | float,
    metric: str,
    value: str | int | float,
    phase: str = "",
    seed: str | int = "",
    protocol: dict[str, Any] | None = None,
    scenario_type: str = "",
    scenario_value: str | int | float = "",
) -> dict[str, str | int | float]:
    base = {
        "experiment": experiment,
        "phase": phase,
        "seed": seed,
        "scenario_type": scenario_type,
        "scenario_value": scenario_value,
        "model": model,
        "horizon": horizon,
        "metric": metric,
        "value": value,
    }
    protocol = protocol or {}
    for k in PROTOCOL_COLUMNS:
        base[k] = protocol.get(k, "")
    return base


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _extract_protocol_fields(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in PROTOCOL_COLUMNS:
        if k in payload:
            out[k] = payload.get(k, "")
    return out


def _load_protocol_context(result_dir: Path, result_json_names: list[str] | None = None) -> dict[str, Any]:
    result_json_names = result_json_names or []
    merged = {k: "" for k in PROTOCOL_COLUMNS}

    meta = _read_json(result_dir / "run_metadata.json")
    for k, v in _extract_protocol_fields(meta).items():
        merged[k] = v

    for name in result_json_names:
        payload = _read_json(result_dir / name)
        cfg = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        for k, v in _extract_protocol_fields(cfg).items():
            if merged.get(k, "") in ("", None):
                merged[k] = v
    return merged


def _load_protocol_from_script(script_path: Path) -> dict[str, str]:
    if not script_path.exists():
        return {}
    key_to_const = {
        "eval_protocol": "EVAL_PROTOCOL",
        "eval_ar_seed_mode": "EVAL_AR_SEED_MODE",
        "train_supervision_protocol": "TRAIN_SUPERVISION_PROTOCOL",
        "optimizer_profile": "OPTIMIZER_PROFILE",
        "strict_repro_mode": "STRICT_REPRO_MODE",
    }
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    for key, const_name in key_to_const.items():
        assign_pattern = rf"^\s*{re.escape(const_name)}\s*=\s*(.+)$"
        assign_match = re.search(assign_pattern, text, flags=re.MULTILINE)
        if not assign_match:
            continue
        expr = assign_match.group(1).strip()

        direct_quoted = re.match(r"""^['"]([^'"]+)['"]$""", expr)
        if direct_quoted:
            out[key] = direct_quoted.group(1).strip()
            continue

        default_match = re.search(r""",\s*(['"][^'"]+['"]|True|False)\s*\)""", expr)
        if default_match:
            raw = default_match.group(1).strip()
            if raw in ("True", "False"):
                out[key] = raw.lower()
            else:
                out[key] = raw.strip("\"'")
    return out


def _load_protocol_context_with_script(
    result_dir: Path,
    script_path: Path,
    result_json_names: list[str] | None = None,
) -> dict[str, Any]:
    merged = _load_protocol_context(result_dir, result_json_names=result_json_names)
    script_vals = _load_protocol_from_script(script_path)
    for k in PROTOCOL_COLUMNS:
        if merged.get(k, "") in ("", None):
            merged[k] = script_vals.get(k, "")
    return merged


def _attach_protocol(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    for r in rows:
        for k in PROTOCOL_COLUMNS:
            r[k] = protocol.get(k, "")
    return rows


def _collect_exp1_rows(project_root: Path) -> list[dict[str, str | int | float]]:
    result_dir = project_root / "experiments" / "exp1_sota" / "results"
    protocol = _load_protocol_context_with_script(
        result_dir,
        project_root / "experiments" / "exp1_sota" / "SOTA_comparison.py",
        result_json_names=["baseline_results.json"],
    )
    long_csv = project_root / "experiments" / "exp1_sota" / "results" / "baseline_results_long.csv"
    if long_csv.exists():
        return _attach_protocol([dict(r) for r in _read_csv_rows(long_csv)], protocol)

    rows: list[dict[str, str | int | float]] = []
    csv_path = project_root / "experiments" / "exp1_sota" / "results" / "baseline_results.csv"
    for r in _read_csv_rows(csv_path):
        model = r.get("Model", "")
        for key, value in r.items():
            if key == "Model" or value in ("", None):
                continue
            parts = key.split("_")
            if len(parts) == 2:
                metric, horizon = parts[0], parts[1]
                rows.append(_row("exp1_sota", model, horizon, metric, value, protocol=protocol))
                continue
            if len(parts) == 3:
                metric, horizon, suffix = parts[0], parts[1], parts[2]
                rows.append(_row("exp1_sota", model, horizon, f"{metric}_{suffix}", value, protocol=protocol))
                continue
    return rows


def _collect_exp2_rows(project_root: Path) -> list[dict[str, str | int | float]]:
    result_dir = project_root / "experiments" / "exp2_ablation" / "results"
    protocol = _load_protocol_context_with_script(
        result_dir,
        project_root / "experiments" / "exp2_ablation" / "ablation_study.py",
        result_json_names=["ablation_results.json"],
    )
    rows: list[dict[str, str | int | float]] = []
    csv_path = result_dir / "latest_runs.csv"
    for r in _read_csv_rows(csv_path):
        rows.append(
            _row(
                experiment="exp2_ablation",
                phase=r.get("phase", ""),
                seed=r.get("seed", ""),
                scenario_type="",
                scenario_value="",
                model=r.get("model", ""),
                horizon=r.get("horizon", ""),
                metric=r.get("metric", ""),
                value=r.get("value", ""),
                protocol=protocol,
            )
        )
    return rows


def _collect_exp3_rows(project_root: Path) -> list[dict[str, str | int | float]]:
    result_dir = project_root / "experiments" / "exp3_robustness" / "results"
    protocol = _load_protocol_context_with_script(
        result_dir,
        project_root / "experiments" / "exp3_robustness" / "robustness_analysis.py",
        result_json_names=["robustness_results.json"],
    )
    long_csv = result_dir / "robustness_results_long.csv"
    if long_csv.exists():
        return _attach_protocol([dict(r) for r in _read_csv_rows(long_csv)], protocol)

    # Fallback for legacy CSV format.
    rows: list[dict[str, str | int | float]] = []
    legacy = result_dir / "robustness_results.csv"
    for r in _read_csv_rows(legacy):
        scenario_type = r.get("test_type", "")
        scenario_value = r.get("test_param", "")
        model = r.get("model", "")
        rows.append(_row("exp3_robustness", model, "", "mse", r.get("mse", ""), protocol=protocol, scenario_type=scenario_type, scenario_value=scenario_value))
        rows.append(_row("exp3_robustness", model, "", "mae", r.get("mae", ""), protocol=protocol, scenario_type=scenario_type, scenario_value=scenario_value))
    return rows


def _collect_exp4_rows(project_root: Path) -> list[dict[str, str | int | float]]:
    result_dir = project_root / "experiments" / "exp4_physics_consistency" / "results"
    protocol = _load_protocol_context_with_script(
        result_dir,
        project_root / "experiments" / "exp4_physics_consistency" / "physics_consistency.py",
        result_json_names=["physics_consistency_results.json"],
    )
    long_csv = result_dir / "physics_consistency_results_long.csv"
    if long_csv.exists():
        return _attach_protocol([dict(r) for r in _read_csv_rows(long_csv)], protocol)

    # Fallback for legacy CSV format.
    rows: list[dict[str, str | int | float]] = []
    legacy = result_dir / "physics_consistency_results.csv"
    for r in _read_csv_rows(legacy):
        model = r.get("Model", "")
        mapping = {
            "mse_scaled": r.get("MSE_scaled", ""),
            "mse_physical": r.get("MSE_physical", ""),
            "height_violation_rate": r.get("height_violation", ""),
            "velocity_violation_rate": r.get("velocity_violation", ""),
            "acceleration_violation_rate": r.get("accel_violation", ""),
            "position_smoothness_ratio": r.get("pos_smooth_ratio", ""),
            "velocity_smoothness_ratio": r.get("vel_smooth_ratio", ""),
        }
        for metric, value in mapping.items():
            rows.append(
                _row(
                    experiment="exp4_physics_consistency",
                    model=model,
                    horizon="",
                    metric=metric,
                    value=value,
                    protocol=protocol,
                    scenario_type="physics",
                    scenario_value="overall",
                )
            )
    return rows


def _collect_formal_bundle_rows(project_root: Path) -> list[dict[str, str | int | float]]:
    """Collect only protocol-validated rows from active formal bundles."""
    config_path = project_root / "configs" / "formal_v3.json"
    if not config_path.is_file():
        return []
    try:
        config = load_formal_config(config_path)
    except (OSError, ValueError):
        return []
    main_path = resolve_study_artifact_root(config, "main") / "main_run_set_manifest.json"
    if not main_path.is_file():
        return []
    main_verification = validate_evidence_bundle(main_path, config, expected_kind="main")
    if not main_verification["passed"]:
        return []
    bundles = [("main", main_verification["bundle"])]
    ablation_path = resolve_study_artifact_root(config, "mechanism") / "ablation_run_set_manifest.json"
    if ablation_path.is_file():
        ablation_verification = validate_evidence_bundle(
            ablation_path,
            config,
            expected_kind="ablation",
            main_bundle=main_verification["bundle"],
        )
        if ablation_verification["passed"]:
            bundles.append(("mechanism", ablation_verification["bundle"]))

    rows: list[dict[str, str | int | float]] = []
    for study_name, bundle in bundles:
        study_id = str(config["studies"][study_name]["study_id"])
        for entry in bundle.get("source_records", []):
            record = normalize_run_record(entry["source_path"])
            protocol = record.get("scientific_protocol_core", {})
            model_key = str(record.get("model_key", ""))
            model_name = DISPLAY_NAMES.get(model_key, str(record.get("model", model_key)))
            for horizon in config["task"]["reporting_horizons"]:
                metrics = record.get("eval_results", {}).get(str(horizon), {})
                for metric, aliases in {
                    "ade": ("ade", "trajectory_window_ade"),
                    "fde": ("fde", "trajectory_window_fde"),
                    "rmse_cart_m": ("rmse_cart_m",),
                }.items():
                    value = next(
                        (metrics.get(alias) for alias in aliases if metrics.get(alias) is not None),
                        None,
                    )
                    if value is None:
                        continue
                    rows.append(
                        _row(
                            study_id,
                            model_name,
                            horizon,
                            metric,
                            value,
                            phase=str(record.get("phase", "")),
                            seed=record.get("seed", ""),
                            protocol=protocol,
                        )
                    )
    return rows


def collect_unified_metric_rows(
    project_root: str | Path,
    *,
    include_legacy: bool = False,
) -> list[dict[str, str | int | float]]:
    """Collect validated formal rows; legacy Exp1--Exp4 rows require opt-in."""
    root = Path(project_root)
    rows = _collect_formal_bundle_rows(root)
    if include_legacy:
        rows.extend(_collect_exp1_rows(root))
        rows.extend(_collect_exp2_rows(root))
        rows.extend(_collect_exp3_rows(root))
        rows.extend(_collect_exp4_rows(root))
    return rows


def collect_protocol_consistency_rows(
    project_root: str | Path,
    *,
    include_legacy: bool = False,
) -> list[dict[str, str]]:
    """Audit formal study protocol values; legacy comparison requires opt-in."""
    root = Path(project_root)
    if not include_legacy:
        config_path = root / "configs" / "formal_v3.json"
        try:
            config = load_formal_config(config_path)
        except (OSError, ValueError):
            return []
        protocol = config["task"]
        study_names = list(config["studies"])
        rows: list[dict[str, str]] = []
        for field in PROTOCOL_COLUMNS:
            key = field if field in {"eval_protocol", "eval_ar_seed_mode", "train_supervision_protocol"} else None
            value = str(protocol.get(key, "")) if key else ""
            for study in study_names:
                rows.append(
                    {
                        "field": field,
                        "experiment": str(config["studies"][study]["study_id"]),
                        "value": value,
                        "is_consistent": "1" if value else "",
                        "consistency_state": "consistent" if value else "all_missing",
                        "non_empty_count": str(len(study_names) if value else 0),
                        "total_count": str(len(study_names)),
                    }
                )
        return rows

    exp_to_dir = {
        "exp1_sota": root / "experiments" / "exp1_sota" / "results",
        "exp2_ablation": root / "experiments" / "exp2_ablation" / "results",
        "exp3_robustness": root / "experiments" / "exp3_robustness" / "results",
        "exp4_physics_consistency": root / "experiments" / "exp4_physics_consistency" / "results",
    }
    exp_to_json = {
        "exp1_sota": ["baseline_results.json"],
        "exp2_ablation": ["ablation_results.json"],
        "exp3_robustness": ["robustness_results.json"],
        "exp4_physics_consistency": ["physics_consistency_results.json"],
    }

    protocol_by_exp = {
        "exp1_sota": _load_protocol_context_with_script(
            exp_to_dir["exp1_sota"],
            root / "experiments" / "exp1_sota" / "SOTA_comparison.py",
            exp_to_json.get("exp1_sota", []),
        ),
        "exp2_ablation": _load_protocol_context_with_script(
            exp_to_dir["exp2_ablation"],
            root / "experiments" / "exp2_ablation" / "ablation_study.py",
            exp_to_json.get("exp2_ablation", []),
        ),
        "exp3_robustness": _load_protocol_context_with_script(
            exp_to_dir["exp3_robustness"],
            root / "experiments" / "exp3_robustness" / "robustness_analysis.py",
            exp_to_json.get("exp3_robustness", []),
        ),
        "exp4_physics_consistency": _load_protocol_context_with_script(
            exp_to_dir["exp4_physics_consistency"],
            root / "experiments" / "exp4_physics_consistency" / "physics_consistency.py",
            exp_to_json.get("exp4_physics_consistency", []),
        ),
    }
    rows: list[dict[str, str]] = []
    for field in PROTOCOL_COLUMNS:
        values = []
        for exp in exp_to_dir:
            val = protocol_by_exp.get(exp, {}).get(field, "")
            if val not in ("", None):
                values.append(str(val))
        all_missing = len(values) == 0
        is_consistent = (len(set(values)) <= 1) if not all_missing else False
        state = "all_missing" if all_missing else ("consistent" if is_consistent else "inconsistent")
        for exp in exp_to_dir:
            rows.append(
                {
                    "field": field,
                    "experiment": exp,
                    "value": str(protocol_by_exp.get(exp, {}).get(field, "")),
                    "is_consistent": "" if all_missing else ("1" if is_consistent else "0"),
                    "consistency_state": state,
                    "non_empty_count": str(len(values)),
                    "total_count": str(len(exp_to_dir)),
                }
            )
    return rows


def save_unified_metrics_csv(
    project_root: str | Path,
    output_path: str | Path | None = None,
    *,
    include_legacy: bool = False,
) -> Path:
    """Save validated formal rows into one unified CSV file.

    Set ``include_legacy=True`` only for an explicit historical diagnostic.
    Also writes `protocol_consistency.csv` next to the unified file.
    """
    root = Path(project_root)
    rows = collect_unified_metric_rows(root, include_legacy=include_legacy)
    out = Path(output_path) if output_path is not None else root / "experiments" / "unified_metrics.csv"
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in UNIFIED_COLUMNS})

    protocol_out = out.parent / "protocol_consistency.csv"
    protocol_rows = collect_protocol_consistency_rows(root, include_legacy=include_legacy)
    with protocol_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "field",
                "experiment",
                "value",
                "is_consistent",
                "consistency_state",
                "non_empty_count",
                "total_count",
            ],
        )
        writer.writeheader()
        writer.writerows(protocol_rows)
    return out
