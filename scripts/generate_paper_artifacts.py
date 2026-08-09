#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper-ready tables and figures for the PLGAFormer HGV study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.trajectory_protocol import dataset_time_metadata


MODEL_ORDER = [
    "Transformer (baseline)",
    "PLGAFormer (A+B+C)",
    "PLGAFormer w/o A",
    "PLGAFormer w/o B",
    "PLGAFormer w/o C",
]

FORMAL_ABLATION_MODEL_NAMES = {
    "baseline": "Transformer (baseline)",
    "full": "PLGAFormer (A+B+C)",
    "wo_a": "PLGAFormer w/o A",
    "wo_b": "PLGAFormer w/o B",
    "wo_c": "PLGAFormer w/o C",
}

SOTA_MODEL_NAMES = {
    "full": "PLGAFormer (proposed)",
    "baseline": "Transformer (baseline)",
    "itransformer": "iTransformer",
    "pit": "PIT",
    "fedformer": "FEDformer",
    "patchtst": "PatchTST",
    "autoformer": "Autoformer",
    "informer": "Informer",
    "kalman": "Kalman",
    "timesnet": "TimesNet",
}

SOTA_MODEL_ORDER = [
    "PLGAFormer (proposed)",
    "iTransformer",
    "PIT",
    "Transformer (baseline)",
    "FEDformer",
    "PatchTST",
    "Autoformer",
    "Informer",
    "TimesNet",
    "Kalman",
]

ROBUSTNESS_SCENARIO_ORDER = ["noise", "missing", "input_length"]
PHYSICS_MODEL_ORDER = ["PLGAFormer (proposed)", "Transformer (baseline)"]

METRICS = ("mse", "mae", "rmse")
SOTA_METRICS = ("mse", "mae", "fde", "ade")
EARTH_RADIUS_M = 6_378_000.0


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _pct_change(reference: float, value: float) -> float:
    if not math.isfinite(reference) or abs(reference) < 1e-12:
        return math.nan
    return (reference - value) / reference * 100.0


def _pct_delta(reference: float, value: float) -> float:
    if not math.isfinite(reference) or abs(reference) < 1e-12:
        return math.nan
    return (value - reference) / reference * 100.0


def _ordered_models(stats: dict[str, Any]) -> list[str]:
    ordered = [model for model in MODEL_ORDER if model in stats]
    ordered.extend(sorted(model for model in stats if model not in set(ordered)))
    return ordered


def _ordered_sota_models(stats: dict[str, Any]) -> list[str]:
    ordered = [model for model in SOTA_MODEL_ORDER if model in stats]
    ordered.extend(sorted(model for model in stats if model not in set(ordered)))
    return ordered


def _ordered_physics_models(stats: dict[str, Any]) -> list[str]:
    ordered = [model for model in PHYSICS_MODEL_ORDER if model in stats]
    ordered.extend(sorted(model for model in stats if model not in set(ordered)))
    return ordered


def _select_horizon(stats: dict[str, Any], horizon: str | int | None = None) -> str:
    if horizon is not None:
        return str(horizon)
    horizons: set[str] = set()
    for by_horizon in stats.values():
        if isinstance(by_horizon, dict):
            horizons.update(str(key) for key in by_horizon.keys())
    if not horizons:
        raise ValueError("No horizons found in ablation statistics")
    return sorted(horizons, key=lambda item: (not str(item).isdigit(), int(item) if str(item).isdigit() else str(item)))[0]


def _all_horizons(stats: dict[str, Any]) -> list[str]:
    horizons: set[str] = set()
    for by_horizon in stats.values():
        if isinstance(by_horizon, dict):
            horizons.update(str(key) for key in by_horizon.keys())
    return sorted(horizons, key=lambda item: (not str(item).isdigit(), int(item) if str(item).isdigit() else str(item)))


def _formal_ablation_summary_to_phases(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Convert summarize_formal_ablation.py JSON payloads into the legacy phase shape."""
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    stats: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    matched = False
    for item in rows:
        if not isinstance(item, dict):
            continue
        model_key = str(item.get("model_key", "")).strip()
        metric = str(item.get("metric", "")).strip()
        horizon = str(item.get("horizon", "")).strip()
        if model_key not in FORMAL_ABLATION_MODEL_NAMES or metric not in METRICS or not horizon:
            continue
        matched = True
        model = FORMAL_ABLATION_MODEL_NAMES[model_key]
        stats.setdefault(model, {}).setdefault(horizon, {})[metric] = {
            "mean": _as_float(item.get("mean")),
            "std": _as_float(item.get("std"), 0.0),
            "n": int(float(item.get("n", 0) or 0)),
            "min": _as_float(item.get("min")),
            "max": _as_float(item.get("max")),
        }
    if not matched:
        return None
    return {"phases": {"formal_ablation": {"statistics": stats}}}


def extract_ablation_rows(
    payload: dict[str, Any],
    phase: str = "best_candidate_ablation",
    horizon: str | int | None = None,
) -> list[dict[str, Any]]:
    """Extract one compact ablation table from exp2 latest_results.json."""
    formal_payload = _formal_ablation_summary_to_phases(payload)
    if formal_payload is not None:
        payload = formal_payload
        phase = "formal_ablation"

    phases = payload.get("phases", {})
    if phase not in phases:
        if not isinstance(phases, dict) or not phases:
            raise ValueError("No phases found in ablation results")
        phase = next(iter(phases.keys()))
    stats = phases[phase].get("statistics", {})
    if not isinstance(stats, dict) or not stats:
        raise ValueError(f"No statistics found for phase {phase}")

    selected_horizons = _all_horizons(stats) if str(horizon).lower().strip() == "all" else [_select_horizon(stats, horizon)]
    rows: list[dict[str, Any]] = []
    for selected_horizon in selected_horizons:
        horizon_rows: list[dict[str, Any]] = []
        for model in _ordered_models(stats):
            model_stats = stats.get(model, {})
            horizon_stats = model_stats.get(selected_horizon)
            if horizon_stats is None and isinstance(model_stats, dict) and model_stats and len(selected_horizons) == 1:
                fallback_horizon = _select_horizon({model: model_stats})
                horizon_stats = model_stats.get(fallback_horizon, {})
            if not isinstance(horizon_stats, dict):
                continue

            row: dict[str, Any] = {
                "phase": phase,
                "model": model,
                "horizon": int(selected_horizon) if str(selected_horizon).isdigit() else selected_horizon,
                "n": 0,
            }
            for metric in METRICS:
                metric_stats = horizon_stats.get(metric, {})
                if not isinstance(metric_stats, dict):
                    metric_stats = {}
                row[metric] = _as_float(metric_stats.get("mean"))
                row[f"{metric}_std"] = _as_float(metric_stats.get("std"))
                row[f"{metric}_ci95_low"] = _as_float(metric_stats.get("ci95_low"))
                row[f"{metric}_ci95_high"] = _as_float(metric_stats.get("ci95_high"))
                row["n"] = max(int(metric_stats.get("n", 0) or 0), int(row["n"]))
            horizon_rows.append(row)

        baseline = next((row for row in horizon_rows if row["model"] == "Transformer (baseline)"), None)
        full = next((row for row in horizon_rows if row["model"] == "PLGAFormer (A+B+C)"), None)
        for row in horizon_rows:
            for metric in METRICS:
                value = _as_float(row.get(metric))
                baseline_value = _as_float(baseline.get(metric)) if baseline else math.nan
                full_value = _as_float(full.get(metric)) if full else math.nan
                row[f"{metric}_improvement_vs_baseline_pct"] = _pct_change(baseline_value, value)
                row[f"{metric}_delta_vs_full_pct"] = _pct_delta(full_value, value)
        rows.extend(horizon_rows)
    return rows


def extract_sota_rows(payload: dict[str, Any], horizon: str | int | None = None) -> list[dict[str, Any]]:
    """Extract paper-ready SOTA rows from formal SOTA summary JSON."""
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("No formal SOTA rows found")
    stats: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        model_key = str(item.get("model_key", "")).strip()
        metric = str(item.get("metric", "")).strip()
        horizon_key = str(item.get("horizon", "")).strip()
        if model_key not in SOTA_MODEL_NAMES or metric not in SOTA_METRICS or not horizon_key:
            continue
        model = SOTA_MODEL_NAMES[model_key]
        stats.setdefault(model, {}).setdefault(horizon_key, {})[metric] = {
            "mean": _as_float(item.get("mean")),
            "std": _as_float(item.get("std"), 0.0),
            "n": int(float(item.get("n", 0) or 0)),
        }
    if not stats:
        raise ValueError("No SOTA statistics found")

    selected_horizons = _all_horizons(stats) if str(horizon).lower().strip() == "all" else [_select_horizon(stats, horizon)]
    rows: list[dict[str, Any]] = []
    for selected_horizon in selected_horizons:
        horizon_rows: list[dict[str, Any]] = []
        for model in _ordered_sota_models(stats):
            horizon_stats = stats.get(model, {}).get(str(selected_horizon), {})
            if not horizon_stats:
                continue
            row: dict[str, Any] = {
                "model": model,
                "horizon": int(selected_horizon) if str(selected_horizon).isdigit() else selected_horizon,
                "n": 0,
            }
            for metric in SOTA_METRICS:
                metric_stats = horizon_stats.get(metric, {})
                row[metric] = _as_float(metric_stats.get("mean"))
                row[f"{metric}_std"] = _as_float(metric_stats.get("std"), 0.0)
                row["n"] = max(int(metric_stats.get("n", 0) or 0), int(row["n"]))
            horizon_rows.append(row)

        baseline = next((row for row in horizon_rows if row["model"] == "Transformer (baseline)"), None)
        for row in horizon_rows:
            for metric in SOTA_METRICS:
                baseline_value = _as_float(baseline.get(metric)) if baseline else math.nan
                row[f"{metric}_improvement_vs_baseline_pct"] = _pct_change(baseline_value, _as_float(row.get(metric)))
        rows.extend(horizon_rows)
    return rows


def _as_int_if_whole(value: float) -> float | int:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def _ordered_scenarios(result_block: dict[str, Any]) -> list[str]:
    ordered = [scenario for scenario in ROBUSTNESS_SCENARIO_ORDER if scenario in result_block]
    ordered.extend(sorted(scenario for scenario in result_block if scenario not in set(ordered)))
    return ordered


def extract_robustness_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten exp3 robustness JSON into paper-ready rows."""
    config = payload.get("config", {})
    horizon = _as_int_if_whole(_as_float(config.get("prediction_length")))
    rows: list[dict[str, Any]] = []
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return rows

    for model, by_scenario in results.items():
        if not isinstance(by_scenario, dict):
            continue
        for scenario_type in _ordered_scenarios(by_scenario):
            scenario_values = by_scenario.get(scenario_type, {})
            if not isinstance(scenario_values, dict):
                continue
            ordered_values = sorted(
                scenario_values.items(),
                key=lambda item: (_as_float(item[0]), str(item[0])),
            )
            for raw_value, metrics in ordered_values:
                if not isinstance(metrics, dict):
                    continue
                scenario_value = _as_float(raw_value)
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "scenario_type": scenario_type,
                        "scenario_value": _as_int_if_whole(scenario_value),
                        "mse": _as_float(metrics.get("mse")),
                        "mae": _as_float(metrics.get("mae")),
                    }
                )
    return rows


def extract_physics_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract exp4 physics-consistency rows with baseline-relative gains."""
    config = payload.get("config", {})
    horizon = _as_int_if_whole(_as_float(config.get("prediction_length")))
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return []

    baseline = results.get("Transformer (baseline)", {})
    baseline_mse_scaled = _as_float(baseline.get("mse_scaled") if isinstance(baseline, dict) else math.nan)
    baseline_mse_physical = _as_float(baseline.get("mse_physical") if isinstance(baseline, dict) else math.nan)

    rows: list[dict[str, Any]] = []
    for model in _ordered_physics_models(results):
        item = results.get(model, {})
        if not isinstance(item, dict):
            continue
        violations = item.get("violations", {})
        smoothness = item.get("smoothness", {})
        if not isinstance(violations, dict):
            violations = {}
        if not isinstance(smoothness, dict):
            smoothness = {}
        mse_scaled = _as_float(item.get("mse_scaled"))
        mse_physical = _as_float(item.get("mse_physical"))
        position_ratio = _as_float(smoothness.get("position_smoothness_ratio"))
        velocity_ratio = _as_float(smoothness.get("velocity_smoothness_ratio"))
        rows.append(
            {
                "model": model,
                "horizon": horizon,
                "mse_scaled": mse_scaled,
                "mse_physical": mse_physical,
                "height_violation_rate": _as_float(violations.get("height_violation_rate")),
                "velocity_violation_rate": _as_float(violations.get("velocity_violation_rate")),
                "acceleration_violation_rate": _as_float(violations.get("acceleration_violation_rate")),
                "velocity_relative_rmse": _as_float(violations.get("velocity_relative_rmse")),
                "acceleration_relative_rmse": _as_float(violations.get("acceleration_relative_rmse")),
                "position_smoothness_ratio": position_ratio,
                "velocity_smoothness_ratio": velocity_ratio,
                "mse_scaled_improvement_vs_baseline_pct": _pct_change(baseline_mse_scaled, mse_scaled),
                "mse_physical_improvement_vs_baseline_pct": _pct_change(baseline_mse_physical, mse_physical),
                "position_smoothness_excess_pct": _pct_delta(1.0, position_ratio),
                "velocity_smoothness_excess_pct": _pct_delta(1.0, velocity_ratio),
            }
        )
    return rows


def extract_candidate_rows(payload: dict[str, Any], top_k: int | None = None) -> list[dict[str, Any]]:
    """Extract baseline and ranked PLGAFormer candidates from candidate search output."""
    rows: list[dict[str, Any]] = []
    baseline = next((row for row in payload.get("rows", []) if row.get("candidate_id") == "baseline"), None)
    if baseline:
        rows.append(
            {
                "rank": "",
                "candidate_id": "baseline",
                "learning_rate": "",
                "dropout": "",
                "alpha": "",
                "mse": _as_float(baseline.get("mse")),
                "mae": _as_float(baseline.get("mae")),
                "rmse": _as_float(baseline.get("rmse")),
                "improvement_pct": _as_float(baseline.get("baseline_improvement_pct"), 0.0),
                "best_val_loss": _as_float(baseline.get("best_val_loss")),
            }
        )

    ranked = [dict(row) for row in payload.get("ranked_candidates", [])]
    ranked.sort(key=lambda row: int(row.get("rank", 10**9)))
    if top_k is not None:
        ranked = ranked[: int(top_k)]
    for row in ranked:
        rows.append(
            {
                "rank": int(row.get("rank", len(rows))),
                "candidate_id": row.get("candidate_id", ""),
                "learning_rate": row.get("learning_rate", ""),
                "dropout": row.get("dropout", ""),
                "alpha": row.get("alpha", ""),
                "mse": _as_float(row.get("mse")),
                "mae": _as_float(row.get("mae")),
                "rmse": _as_float(row.get("rmse")),
                "improvement_pct": _as_float(row.get("baseline_improvement_pct")),
                "best_val_loss": _as_float(row.get("best_val_loss")),
            }
        )
    return rows


def write_csv_table(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return "" if value is None else str(value)


def write_latex_table(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    caption: str,
    label: str,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "c" * max(0, len(fieldnames) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_latex_escape(name) for name in fieldnames) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_cell(row.get(name, ""))) for name in fieldnames) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _metric_pm(mean: Any, std: Any) -> str:
    mean_f = _as_float(mean)
    std_f = _as_float(std)
    if math.isfinite(mean_f) and math.isfinite(std_f):
        return f"{mean_f:.6f} +/- {std_f:.6f}"
    if math.isfinite(mean_f):
        return f"{mean_f:.6f}"
    return ""


def make_ablation_display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        display_rows.append(
            {
                "model": row["model"],
                "horizon": row["horizon"],
                "n": row["n"],
                "mse": _metric_pm(row.get("mse"), row.get("mse_std")),
                "mae": _metric_pm(row.get("mae"), row.get("mae_std")),
                "rmse": _metric_pm(row.get("rmse"), row.get("rmse_std")),
                "mse_gain_pct": _as_float(row.get("mse_improvement_vs_baseline_pct")),
                "mse_delta_full_pct": _as_float(row.get("mse_delta_vs_full_pct")),
            }
        )
    return display_rows


def select_representative_indices(
    labels: Any,
    trajectory_ids: Any,
    window_starts: Any,
    max_samples: int = 4,
) -> list[int]:
    """Pick early windows while covering distinct maneuver labels first."""
    n = min(len(labels), len(trajectory_ids), len(window_starts))
    if n <= 0 or max_samples <= 0:
        return []
    labels_list = [str(labels[i]) for i in range(n)]
    ids = [int(trajectory_ids[i]) for i in range(n)]
    starts = [int(window_starts[i]) for i in range(n)]

    label_order: list[str] = []
    best_by_label: dict[str, int] = {}
    for idx, label in enumerate(labels_list):
        if label not in best_by_label:
            best_by_label[label] = idx
            label_order.append(label)
        else:
            cur = best_by_label[label]
            if (ids[idx], starts[idx], idx) < (ids[cur], starts[cur], cur):
                best_by_label[label] = idx

    selected = [best_by_label[label] for label in label_order[:max_samples]]
    if len(selected) < max_samples:
        for idx in sorted(range(n), key=lambda item: (ids[item], starts[item], item)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= max_samples:
                break
    return selected


def _decode_np_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_np_value(value.item())
        if value.size == 1:
            return _decode_np_value(value.reshape(-1)[0])
        return ",".join(_decode_np_value(v) for v in value.reshape(-1))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _split_labels(data: dict[str, np.ndarray], split: str) -> np.ndarray:
    split_key = f"maneuver_labels_{split}"
    if split_key in data:
        return np.asarray(data[split_key])
    if "maneuver_labels" in data:
        labels = np.asarray(data["maneuver_labels"])
        n_train = len(data.get("X_train", []))
        n_val = len(data.get("X_val", []))
        if split == "train":
            return labels[:n_train]
        if split == "val":
            return labels[n_train:n_train + n_val]
        return labels[n_train + n_val:]
    return np.asarray([])


def load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def build_dataset_summary_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    protocol = _decode_np_value(data.get("dataset_protocol", "unknown"))
    timing = dataset_time_metadata(data)
    has_timing = "sampling_interval_s" in data
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        x_key = f"X_{split}"
        y_key = f"y_{split}"
        ids = np.asarray(data.get(f"trajectory_ids_{split}", []))
        labels = _split_labels(data, split)
        x = np.asarray(data.get(x_key, []))
        y = np.asarray(data.get(y_key, []))
        rows.append(
            {
                "split": split,
                "protocol": protocol,
                "windows": int(len(x)),
                "trajectories": int(len(np.unique(ids))) if ids.size else 0,
                "maneuver_classes": int(len(np.unique(labels))) if labels.size else 0,
                "input_length": int(x.shape[1]) if x.ndim >= 2 else "",
                "prediction_length": int(y.shape[1]) if y.ndim >= 2 else "",
                "sampling_interval_s": timing["sampling_interval_s"] if has_timing else "",
                "input_duration_s": timing["input_duration_s"] if has_timing else "",
                "prediction_duration_s": timing["prediction_duration_s"] if has_timing else "",
                "input_dim": int(x.shape[2]) if x.ndim >= 3 else "",
                "output_dim": int(y.shape[2]) if y.ndim >= 3 else "",
            }
        )
    return rows


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    return plt


def _save_figure(fig: Any, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths


def make_ablation_figure(rows: list[dict[str, Any]], output_stem: str | Path) -> list[Path]:
    plt = _setup_matplotlib()
    models = [row["model"] for row in rows]
    x = np.arange(len(models))
    palette = ["#4d4d4d", "#0072b2", "#d55e00", "#009e73", "#cc79a7"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ("mse", "mae"), ("MSE", "MAE")):
        means = np.array([_as_float(row.get(metric), 0.0) for row in rows])
        stds = np.array([_as_float(row.get(f"{metric}_std"), 0.0) for row in rows])
        ax.bar(x, means, yerr=stds, capsize=4, color=palette[: len(rows)], edgecolor="#222222", linewidth=0.8)
        ax.set_title(f"Ablation {title}")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, Path(output_stem))


def make_sota_figure(rows: list[dict[str, Any]], output_stem: str | Path) -> list[Path]:
    plt = _setup_matplotlib()
    models = _ordered_sota_models({row["model"]: {} for row in rows})
    fig, ax = plt.subplots(figsize=(9.8, 4.6), constrained_layout=True)
    palette = ["#0072b2", "#009e73", "#cc79a7", "#4d4d4d", "#d55e00", "#56b4e9", "#e69f00", "#999999"]
    for idx, model in enumerate(models):
        model_rows = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: int(row["horizon"]) if str(row["horizon"]).isdigit() else 0,
        )
        if not model_rows:
            continue
        horizons = [int(row["horizon"]) for row in model_rows]
        mse = [_as_float(row.get("mse")) for row in model_rows]
        ax.plot(horizons, mse, marker="o", linewidth=2.0, label=model, color=palette[idx % len(palette)])
    ax.set_yscale("log")
    ax.set_title("Formal SOTA Comparison")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("MSE (scaled, log)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, ncol=2)
    return _save_figure(fig, Path(output_stem))


def make_robustness_figure(rows: list[dict[str, Any]], output_stem: str | Path) -> list[Path]:
    plt = _setup_matplotlib()
    scenarios = _ordered_scenarios({row["scenario_type"]: {} for row in rows})
    if not scenarios:
        return []
    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.2 * len(scenarios), 6.2), constrained_layout=True, squeeze=False)
    palette = ["#0072b2", "#d55e00", "#009e73", "#cc79a7"]
    labels = {
        "noise": ("Noise level (%)", lambda value: _as_float(value) * 100.0),
        "missing": ("Missing rate (%)", lambda value: _as_float(value) * 100.0),
        "input_length": ("Input length (steps)", lambda value: _as_float(value)),
    }
    models = list(dict.fromkeys(row["model"] for row in rows))
    for col, scenario in enumerate(scenarios):
        scenario_rows = [row for row in rows if row["scenario_type"] == scenario]
        x_label, x_transform = labels.get(scenario, (scenario.replace("_", " ").title(), lambda value: _as_float(value)))
        for row_idx, metric in enumerate(("mse", "mae")):
            ax = axes[row_idx, col]
            for model_idx, model in enumerate(models):
                model_rows = sorted(
                    [row for row in scenario_rows if row["model"] == model],
                    key=lambda row: _as_float(row["scenario_value"]),
                )
                if not model_rows:
                    continue
                x = [x_transform(row["scenario_value"]) for row in model_rows]
                y = [_as_float(row.get(metric)) for row in model_rows]
                ax.plot(x, y, marker="o", linewidth=2.0, label=model, color=palette[model_idx % len(palette)])
            ax.set_title(f"{scenario.replace('_', ' ').title()} {metric.upper()}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(f"{metric.upper()} (scaled)")
            ax.grid(alpha=0.25)
            if row_idx == 0:
                ax.legend(fontsize=8)
    return _save_figure(fig, Path(output_stem))


def make_physics_figure(rows: list[dict[str, Any]], output_stem: str | Path) -> list[Path]:
    plt = _setup_matplotlib()
    rows = [row for row in rows if row.get("model")]
    if not rows:
        return []
    models = [row["model"] for row in rows]
    x = np.arange(len(models))
    colors = ["#0072b2" if model == "PLGAFormer (proposed)" else "#4d4d4d" for model in models]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)

    axes[0].bar(x, [_as_float(row.get("mse_scaled"), 0.0) for row in rows], color=colors, edgecolor="#222222", linewidth=0.8)
    axes[0].set_title("Scaled Prediction Error")
    axes[0].set_ylabel("MSE (scaled)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    dynamics_metrics = [
        ("velocity_relative_rmse", "Velocity"),
        ("acceleration_relative_rmse", "Acceleration"),
    ]
    width = 0.32
    dynamics_x = np.arange(len(dynamics_metrics))
    for idx, row in enumerate(rows):
        values = [_as_float(row.get(key), 0.0) for key, _ in dynamics_metrics]
        axes[1].bar(dynamics_x + (idx - (len(rows) - 1) / 2) * width, values, width, label=row["model"], color=colors[idx], edgecolor="#222222", linewidth=0.7)
    axes[1].set_title("Dynamic Relative RMSE")
    axes[1].set_ylabel("Relative RMSE")
    axes[1].set_xticks(dynamics_x)
    axes[1].set_xticklabels([label for _, label in dynamics_metrics])
    axes[1].grid(axis="y", alpha=0.25)

    smoothness_metrics = [
        ("position_smoothness_ratio", "Position"),
        ("velocity_smoothness_ratio", "Velocity"),
    ]
    smooth_x = np.arange(len(smoothness_metrics))
    for idx, row in enumerate(rows):
        values = [_as_float(row.get(key), 0.0) for key, _ in smoothness_metrics]
        axes[2].bar(smooth_x + (idx - (len(rows) - 1) / 2) * width, values, width, label=row["model"], color=colors[idx], edgecolor="#222222", linewidth=0.7)
    axes[2].axhline(1.0, color="#d55e00", linestyle="--", linewidth=1.4)
    axes[2].set_title("Trajectory Smoothness")
    axes[2].set_ylabel("Pred/True ratio")
    axes[2].set_xticks(smooth_x)
    axes[2].set_xticklabels([label for _, label in smoothness_metrics])
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    return _save_figure(fig, Path(output_stem))


def make_candidate_figure(rows: list[dict[str, Any]], output_stem: str | Path) -> list[Path]:
    plt = _setup_matplotlib()
    rows = rows[: min(len(rows), 8)]
    labels = [str(row["candidate_id"]) for row in rows]
    x = np.arange(len(rows))
    mse = np.array([_as_float(row.get("mse"), 0.0) for row in rows])
    improvement = np.array([_as_float(row.get("improvement_pct"), 0.0) for row in rows])
    fig, ax1 = plt.subplots(figsize=(9.8, 4.4), constrained_layout=True)
    ax1.bar(x, mse, color="#56b4e9", edgecolor="#222222", linewidth=0.8, label="MSE")
    ax1.set_ylabel("MSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, improvement, color="#d55e00", marker="o", linewidth=2.0, label="Improvement")
    ax2.set_ylabel("Improvement vs baseline (%)")
    ax1.set_title("Candidate Search Ranking")
    return _save_figure(fig, Path(output_stem))


def make_dataset_window_figure(
    data: dict[str, np.ndarray],
    output_stem: str | Path,
    max_samples: int = 4,
    split: str = "test",
    horizon: int | None = 64,
) -> list[Path]:
    plt = _setup_matplotlib()
    X = np.asarray(data[f"X_{split}"])
    y = np.asarray(data[f"y_{split}"])
    labels = _split_labels(data, split)
    ids = np.asarray(data[f"trajectory_ids_{split}"])
    starts = np.asarray(data[f"window_starts_{split}"])
    selected = select_representative_indices(labels, ids, starts, max_samples=max_samples)
    pred_len = min(y.shape[1], int(horizon or y.shape[1]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
    for order, idx in enumerate(selected):
        obs = X[idx, :, :3]
        future = y[idx, :pred_len, :3]
        label = f"{str(labels[idx])} T{int(ids[idx])} W{int(starts[idx])}"
        obs_alt = (obs[:, 0] - EARTH_RADIUS_M) / 1000.0
        fut_alt = (future[:, 0] - EARTH_RADIUS_M) / 1000.0
        steps_obs = np.arange(len(obs))
        steps_future = np.arange(len(obs), len(obs) + len(future))
        color = colors[order % len(colors)]
        axes[0].plot(steps_obs, obs_alt, color=color, linewidth=1.8, alpha=0.9)
        axes[0].plot(steps_future, fut_alt, color=color, linewidth=1.8, linestyle="--", label=label)
        axes[1].plot(np.rad2deg(obs[:, 1]), np.rad2deg(obs[:, 2]), color=color, linewidth=1.8)
        axes[1].plot(np.rad2deg(future[:, 1]), np.rad2deg(future[:, 2]), color=color, linewidth=1.8, linestyle="--")
    axes[0].axvline(X.shape[1] - 0.5, color="#222222", linewidth=1.0, alpha=0.6)
    axes[0].set_title("Trajectory Windows: Observed Solid, Future Dashed")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    axes[1].set_title("Ground Track")
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].set_ylabel("Latitude (deg)")
    axes[1].grid(alpha=0.25)
    return _save_figure(fig, Path(output_stem))


def spherical_to_cartesian_np(pos: np.ndarray) -> np.ndarray:
    r = pos[..., 0]
    lon = pos[..., 1]
    lat = pos[..., 2]
    cos_lat = np.cos(lat)
    return np.stack(
        [
            r * cos_lat * np.cos(lon),
            r * cos_lat * np.sin(lon),
            r * np.sin(lat),
        ],
        axis=-1,
    )


def _fit_or_load_scalers(project_root: Path, data: dict[str, np.ndarray]):
    import joblib
    from sklearn.preprocessing import StandardScaler

    input_scaler_path = project_root / "data_generation" / "data" / "processed" / "input_scaler.pkl"
    output_scaler_path = project_root / "data_generation" / "data" / "processed" / "output_scaler.pkl"
    if input_scaler_path.exists():
        input_scaler = joblib.load(input_scaler_path)
    else:
        input_scaler = StandardScaler().fit(data["X_train"].reshape(-1, data["X_train"].shape[-1]))
    if output_scaler_path.exists():
        output_scaler = joblib.load(output_scaler_path)
    else:
        output_scaler = StandardScaler().fit(data["y_train"].reshape(-1, data["y_train"].shape[-1]))
    return input_scaler, output_scaler


def _build_full_plgaformer_kwargs(candidate: dict[str, Any]) -> dict[str, Any]:
    from models import HGVConfig

    model_config = HGVConfig.get_model_config("plgaformer")
    innovations = candidate.get("innovations", {})
    if not isinstance(innovations, dict):
        innovations = {}
    return {
        "d_model": model_config.get("d_model", 256),
        "nhead": model_config.get("nhead", 8),
        "num_encoder_layers": model_config.get("num_encoder_layers", 3),
        "num_decoder_layers": model_config.get("num_decoder_layers", 2),
        "dim_feedforward": model_config.get("dim_feedforward", 1024),
        "dropout": float(candidate.get("dropout", model_config.get("dropout", 0.1))),
        "output_dim": model_config.get("output_dim", 3),
        "use_sparse_attention": bool(innovations.get("use_sparse_attention", False)),
        "use_physics_corrector": bool(innovations.get("use_physics_corrector", False)),
        "use_multi_head_output": bool(innovations.get("use_multi_head_output", True)),
        "use_adaptive_fusion": True,
    }


def make_prediction_figure(
    project_root: Path,
    data: dict[str, np.ndarray],
    candidate_payload: dict[str, Any],
    output_stem: str | Path,
    sample_index: int = 0,
    horizon: int = 64,
    label_len: int = 48,
) -> list[Path]:
    """Load latest checkpoints and draw a qualitative prediction figure."""
    import torch
    from models import create_registered_model
    from utils.inference_protocol import predict_by_eval_protocol
    from utils.seq2seq_protocol import align_source_position_scale

    plt = _setup_matplotlib()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = candidate_payload.get("best_candidate") or {}
    input_scaler, output_scaler = _fit_or_load_scalers(project_root, data)

    X = np.asarray(data["X_test"])
    y = np.asarray(data["y_test"])
    labels = _split_labels(data, "test")
    ids = np.asarray(data["trajectory_ids_test"])
    starts = np.asarray(data["window_starts_test"])
    selected = select_representative_indices(labels, ids, starts, max_samples=max(1, sample_index + 1))
    idx = selected[min(sample_index, len(selected) - 1)] if selected else 0
    horizon = min(int(horizon), y.shape[1])

    x_raw = X[idx : idx + 1].astype(np.float32)
    y_raw = y[idx : idx + 1, :horizon].astype(np.float32)
    x_scaled = input_scaler.transform(x_raw.reshape(-1, x_raw.shape[-1])).reshape(x_raw.shape).astype(np.float32)
    x_scaled = align_source_position_scale(
        x_scaled,
        input_mean=input_scaler.mean_,
        input_scale=input_scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=y_raw.shape[-1],
    ).astype(np.float32, copy=False)
    y_scaled = output_scaler.transform(y_raw.reshape(-1, y_raw.shape[-1])).reshape(y_raw.shape).astype(np.float32)
    x_tensor = torch.from_numpy(x_scaled).to(device)
    y_tensor = torch.from_numpy(y_scaled).to(device)

    models_dir = project_root / "experiments" / "exp2_ablation" / "trained_models"
    ckpt_paths = {
        "Transformer": models_dir / "best_baseline_best_candidate_ablation_transformer_baseline_ablation.pth",
        "PLGAFormer": models_dir / "best_plgaformer_best_candidate_ablation_plgaformer_a_b_c_ablation.pth",
    }
    baseline = create_registered_model("baseline", input_dim=6, device=device)
    full = create_registered_model(
        "plgaformer",
        input_dim=6,
        device=device,
        plgaformer_kwargs=_build_full_plgaformer_kwargs(candidate),
    )
    baseline.load_state_dict(torch.load(ckpt_paths["Transformer"], map_location=device))
    full.load_state_dict(torch.load(ckpt_paths["PLGAFormer"], map_location=device))
    baseline.eval()
    full.eval()

    predictions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, model in (("Transformer", baseline), ("PLGAFormer", full)):
            pred_scaled = predict_by_eval_protocol(
                model=model,
                x=x_tensor,
                y_true_scaled=y_tensor,
                pred_length=horizon,
                device=device,
                eval_protocol="source_context_decoder",
                eval_ar_seed_mode="zero",
                label_len=label_len,
            )
            pred_np = pred_scaled.detach().cpu().numpy()[0]
            predictions[name] = output_scaler.inverse_transform(pred_np.reshape(-1, pred_np.shape[-1])).reshape(pred_np.shape)

    truth = y_raw[0]
    observed = x_raw[0, :, :3]
    steps_obs = np.arange(len(observed))
    steps_future = np.arange(len(observed), len(observed) + horizon)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    axes[0].plot(steps_obs, (observed[:, 0] - EARTH_RADIUS_M) / 1000.0, color="#4d4d4d", linewidth=2.0, label="Observed")
    axes[0].plot(steps_future, (truth[:, 0] - EARTH_RADIUS_M) / 1000.0, color="#111111", linewidth=2.2, label="True")
    axes[0].plot(steps_future, (predictions["Transformer"][:, 0] - EARTH_RADIUS_M) / 1000.0, color="#d55e00", linestyle="--", label="Transformer")
    axes[0].plot(steps_future, (predictions["PLGAFormer"][:, 0] - EARTH_RADIUS_M) / 1000.0, color="#0072b2", linestyle="--", label="PLGAFormer")
    axes[0].set_title("Altitude Prediction")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(np.rad2deg(observed[:, 1]), np.rad2deg(observed[:, 2]), color="#4d4d4d", linewidth=2.0, label="Observed")
    axes[1].plot(np.rad2deg(truth[:, 1]), np.rad2deg(truth[:, 2]), color="#111111", linewidth=2.2, label="True")
    axes[1].plot(np.rad2deg(predictions["Transformer"][:, 1]), np.rad2deg(predictions["Transformer"][:, 2]), color="#d55e00", linestyle="--", label="Transformer")
    axes[1].plot(np.rad2deg(predictions["PLGAFormer"][:, 1]), np.rad2deg(predictions["PLGAFormer"][:, 2]), color="#0072b2", linestyle="--", label="PLGAFormer")
    axes[1].set_title("Ground Track Prediction")
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].set_ylabel("Latitude (deg)")
    axes[1].grid(alpha=0.25)

    truth_xyz = spherical_to_cartesian_np(truth)
    for name, color in (("Transformer", "#d55e00"), ("PLGAFormer", "#0072b2")):
        err_km = np.linalg.norm(spherical_to_cartesian_np(predictions[name]) - truth_xyz, axis=-1) / 1000.0
        axes[2].plot(np.arange(horizon), err_km, color=color, linewidth=2.0, label=name)
    axes[2].set_title("Cartesian Position Error")
    axes[2].set_xlabel("Prediction Step")
    axes[2].set_ylabel("Error (km)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    return _save_figure(fig, Path(output_stem))


def write_protocol_note(path: str | Path, dataset_rows: list[dict[str, Any]], run_config: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Protocol Summary",
        "",
        "- Dataset split protocol: trajectory-level split before windowing.",
        f"- Training supervision: {run_config.get('train_supervision_protocol', '')}.",
        f"- Evaluation protocol: {run_config.get('eval_protocol', '')}.",
        f"- Decoder label length: {run_config.get('label_len', '')}.",
        f"- Candidate seed mode: {run_config.get('seed_mode', '')}.",
        "",
        "## Split Summary",
        "",
    ]
    for row in dataset_rows:
        lines.append(
            f"- {row['split']}: {row['trajectories']} trajectories, {row['windows']} windows, "
            + (
                f"{row['input_length']} observed steps ({row['input_duration_s']:g} s), "
                f"{row['prediction_length']} future steps ({row['prediction_duration_s']:g} s), "
                f"dt={row['sampling_interval_s']:g} s."
                if row["sampling_interval_s"] != ""
                else f"{row['input_length']} observed steps, {row['prediction_length']} future steps; dt metadata missing."
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def generate_artifacts(args: argparse.Namespace) -> dict[str, str]:
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_payload = read_json(project_root / args.ablation_json)
    candidate_payload = read_json(project_root / args.candidate_json)
    sota_payload = None
    sota_summary_path = project_root / args.sota_summary_json
    if sota_summary_path.exists():
        sota_payload = read_json(sota_summary_path)
    robustness_payload = None
    robustness_json_path = project_root / args.robustness_json
    if robustness_json_path.exists():
        robustness_payload = read_json(robustness_json_path)
    physics_payload = None
    physics_json_path = project_root / args.physics_json
    if physics_json_path.exists():
        physics_payload = read_json(physics_json_path)
    data = load_npz_dict(project_root / args.dataset_npz)

    ablation_rows = extract_ablation_rows(ablation_payload, horizon=args.horizon)
    sota_rows = extract_sota_rows(sota_payload, horizon=args.horizon) if sota_payload is not None else []
    candidate_rows = extract_candidate_rows(candidate_payload, top_k=args.top_k)
    dataset_rows = build_dataset_summary_rows(data)
    robustness_rows = extract_robustness_rows(robustness_payload) if robustness_payload is not None else []
    physics_rows = extract_physics_rows(physics_payload) if physics_payload is not None else []

    artifacts: dict[str, str] = {}
    ablation_fields = [
        "model",
        "horizon",
        "n",
        "mse",
        "mse_std",
        "mae",
        "mae_std",
        "rmse",
        "rmse_std",
        "mse_improvement_vs_baseline_pct",
        "mse_delta_vs_full_pct",
    ]
    candidate_fields = [
        "rank",
        "candidate_id",
        "learning_rate",
        "dropout",
        "alpha",
        "mse",
        "mae",
        "rmse",
        "improvement_pct",
        "best_val_loss",
    ]
    dataset_fields = [
        "split",
        "protocol",
        "trajectories",
        "windows",
        "maneuver_classes",
        "input_length",
        "prediction_length",
        "sampling_interval_s",
        "input_duration_s",
        "prediction_duration_s",
        "input_dim",
        "output_dim",
    ]
    sota_fields = [
        "model",
        "horizon",
        "n",
        "mse",
        "mse_std",
        "mae",
        "mae_std",
        "fde",
        "fde_std",
        "ade",
        "ade_std",
        "mse_improvement_vs_baseline_pct",
    ]
    robustness_fields = [
        "model",
        "horizon",
        "scenario_type",
        "scenario_value",
        "mse",
        "mae",
    ]
    physics_fields = [
        "model",
        "horizon",
        "mse_scaled",
        "mse_physical",
        "height_violation_rate",
        "velocity_violation_rate",
        "acceleration_violation_rate",
        "velocity_relative_rmse",
        "acceleration_relative_rmse",
        "position_smoothness_ratio",
        "velocity_smoothness_ratio",
        "mse_scaled_improvement_vs_baseline_pct",
        "mse_physical_improvement_vs_baseline_pct",
        "position_smoothness_excess_pct",
        "velocity_smoothness_excess_pct",
    ]

    artifacts["table_ablation_csv"] = str(write_csv_table(output_dir / "table_ablation_metrics.csv", ablation_rows, ablation_fields))
    artifacts["table_ablation_tex"] = str(
        write_latex_table(
            output_dir / "table_ablation_metrics.tex",
            make_ablation_display_rows(ablation_rows),
            ["model", "horizon", "n", "mse", "mae", "rmse", "mse_gain_pct", "mse_delta_full_pct"],
            "Ablation study under the source-context long-trajectory protocol.",
            "tab:plgaformer_ablation",
        )
    )
    artifacts["table_candidate_csv"] = str(write_csv_table(output_dir / "table_candidate_search.csv", candidate_rows, candidate_fields))
    artifacts["table_candidate_tex"] = str(
        write_latex_table(
            output_dir / "table_candidate_search.tex",
            candidate_rows,
            candidate_fields,
            "Candidate search summary for the full PLGAFormer.",
            "tab:plgaformer_candidate_search",
        )
    )
    artifacts["table_dataset_csv"] = str(write_csv_table(output_dir / "table_dataset_protocol.csv", dataset_rows, dataset_fields))
    artifacts["table_dataset_tex"] = str(
        write_latex_table(
            output_dir / "table_dataset_protocol.tex",
            dataset_rows,
            dataset_fields,
            "Trajectory-level dataset protocol summary.",
            "tab:hgv_dataset_protocol",
        )
    )

    for path in make_ablation_figure(ablation_rows, output_dir / "fig_ablation_mse_mae"):
        artifacts[f"figure_ablation_{path.suffix[1:]}"] = str(path)
    if sota_rows:
        artifacts["table_sota_csv"] = str(write_csv_table(output_dir / "table_sota_metrics.csv", sota_rows, sota_fields))
        artifacts["table_sota_tex"] = str(
            write_latex_table(
                output_dir / "table_sota_metrics.tex",
                sota_rows,
                sota_fields,
                "Formal SOTA comparison under the source-context long-trajectory protocol.",
                "tab:hgv_sota_comparison",
            )
        )
        for path in make_sota_figure(sota_rows, output_dir / "fig_sota_mse"):
            artifacts[f"figure_sota_{path.suffix[1:]}"] = str(path)
    if robustness_rows:
        artifacts["table_robustness_csv"] = str(write_csv_table(output_dir / "table_robustness_metrics.csv", robustness_rows, robustness_fields))
        artifacts["table_robustness_tex"] = str(
            write_latex_table(
                output_dir / "table_robustness_metrics.tex",
                robustness_rows,
                robustness_fields,
                "Robustness analysis under noise, missing data, and context-length perturbations.",
                "tab:hgv_robustness",
            )
        )
        for path in make_robustness_figure(robustness_rows, output_dir / "fig_robustness_mse_mae"):
            artifacts[f"figure_robustness_{path.suffix[1:]}"] = str(path)
    if physics_rows:
        artifacts["table_physics_csv"] = str(write_csv_table(output_dir / "table_physics_consistency.csv", physics_rows, physics_fields))
        artifacts["table_physics_tex"] = str(
            write_latex_table(
                output_dir / "table_physics_consistency.tex",
                physics_rows,
                physics_fields,
                "Physics-consistency analysis on the held-out trajectory split.",
                "tab:hgv_physics_consistency",
            )
        )
        for path in make_physics_figure(physics_rows, output_dir / "fig_physics_consistency"):
            artifacts[f"figure_physics_{path.suffix[1:]}"] = str(path)
    for path in make_candidate_figure(candidate_rows, output_dir / "fig_candidate_search"):
        artifacts[f"figure_candidate_{path.suffix[1:]}"] = str(path)
    figure_horizon = None if str(args.horizon).lower().strip() == "all" else int(args.horizon)
    for path in make_dataset_window_figure(data, output_dir / "fig_dataset_windows", max_samples=args.max_trajectory_samples, horizon=figure_horizon):
        artifacts[f"figure_dataset_{path.suffix[1:]}"] = str(path)

    if not args.skip_prediction_figure:
        try:
            for path in make_prediction_figure(
                project_root=project_root,
                data=data,
                candidate_payload=candidate_payload,
                output_stem=output_dir / "fig_prediction_example",
                horizon=int(figure_horizon or 64),
                label_len=int(candidate_payload.get("run_config", {}).get("label_len", 48)),
            ):
                artifacts[f"figure_prediction_{path.suffix[1:]}"] = str(path)
        except Exception as exc:
            artifacts["prediction_figure_warning"] = str(exc)

    artifacts["protocol_note"] = str(
        write_protocol_note(
            output_dir / "protocol_summary.md",
            dataset_rows,
            candidate_payload.get("run_config", {}),
        )
    )
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = {
        "ablation_json": str(project_root / args.ablation_json),
        "sota_summary_json": str(sota_summary_path) if sota_summary_path.exists() else "",
        "robustness_json": str(robustness_json_path) if robustness_json_path.exists() else "",
        "physics_json": str(physics_json_path) if physics_json_path.exists() else "",
        "candidate_json": str(project_root / args.candidate_json),
        "dataset_npz": str(project_root / args.dataset_npz),
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts["manifest"] = str(manifest_path)
    return artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--ablation-json", type=str, default="experiments/exp2_ablation/results/latest_results.json")
    parser.add_argument("--sota-summary-json", type=str, default="experiments/exp1_sota/results/formal/formal_ablation_summary.json")
    parser.add_argument("--robustness-json", type=str, default="experiments/exp3_robustness/results/robustness_results.json")
    parser.add_argument("--physics-json", type=str, default="experiments/exp4_physics_consistency/results/physics_consistency_results.json")
    parser.add_argument("--candidate-json", type=str, default="experiments/exp2_ablation/results/latest_candidate_search.json")
    parser.add_argument("--dataset-npz", type=str, default="data_generation/data/processed/hgv_trajectory_dataset.npz")
    parser.add_argument("--output-dir", type=str, default="experiments/paper_artifacts")
    parser.add_argument("--horizon", type=str, default="64")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-trajectory-samples", type=int, default=4)
    parser.add_argument("--skip-prediction-figure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = generate_artifacts(args)
    print("[paper-artifacts] generated:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
