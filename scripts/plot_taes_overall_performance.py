#!/usr/bin/env python3
"""Render the current multi-horizon PLGAFormer performance figure.

Figure contract
---------------
Core conclusion:
    PLGAFormer has the lowest mean ADE, FDE, and Cartesian RMSE among the
    listed learning-based methods at the four evaluated horizons.
Archetype:
    Quantitative grid with three aligned metric panels.
Evidence:
    PublicRelease/evidence/main_results.csv, using all 60 rows required by
    five trainable methods, four horizons, and three metrics.
    The 24 rows for the two analytical models in the same CSV are explicitly
    excluded from this learning-based ranking figure and recorded in the QA
    manifest.
Uncertainty:
    Mean plus sample standard deviation across three seeds, converted from m
    to km. No significance annotation is added.
Transform:
    A log y-axis is used because the verified errors span multiple orders of
    magnitude across the methods and horizons.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "DLinear",
    "Transformer",
    "PatchTST",
    "iTransformer",
    "PLGAFormer",
]
METHOD_SOURCE_NAMES = {
    "DLinear": "DLinear",
    "Transformer": "Transformer (baseline)",
    "PatchTST": "PatchTST",
    "iTransformer": "iTransformer",
    "PLGAFormer": "PLGAFormer (proposed)",
}
ANALYTICAL_SOURCE_NAMES = {"Rotating-Earth 3-DOF", "Spherical kinematics"}
METHOD_COLORS = {
    "DLinear": "#7A8B99",
    "Transformer": "#9C9FA8",
    "PatchTST": "#B18A49",
    "iTransformer": "#4C78A8",
    "PLGAFormer": "#C44E52",
}
METHOD_MARKERS = {
    "DLinear": "o",
    "Transformer": "s",
    "PatchTST": "^",
    "iTransformer": "D",
    "PLGAFormer": "*",
}
METRIC_FIELDS = {
    "ADE": "ade",
    "FDE": "fde",
    "Cartesian RMSE": "rmse_cart_m",
}
HORIZONS = [32, 64, 128, 256]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )


def read_evidence(
    path: Path,
) -> tuple[dict[tuple[str, int, str], tuple[float, float]], dict[str, int]]:
    values: dict[tuple[str, int, str], tuple[float, float]] = {}
    excluded_models: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if model in ANALYTICAL_SOURCE_NAMES:
                excluded_models[model] += 1
                continue
            display_model = next(
                (name for name, source_name in METHOD_SOURCE_NAMES.items() if source_name == model),
                None,
            )
            if display_model is None:
                raise ValueError(f"Unexpected model in evidence table: {model}")
            horizon = int(row["horizon_s"])
            metric = row["metric"]
            mean_km = float(row["mean_m"]) / 1000.0
            std_km = float(row["sample_std_m"]) / 1000.0
            if not np.isfinite(mean_km) or not np.isfinite(std_km) or mean_km <= 0 or std_km < 0:
                raise ValueError(f"Invalid value in evidence table: {row}")
            values[(display_model, horizon, metric)] = (mean_km, std_km)

    expected = {
        (method, horizon, metric)
        for method in METHOD_ORDER
        for horizon in HORIZONS
        for metric in METRIC_FIELDS.values()
    }
    missing = expected.difference(values)
    extra = set(values).difference(expected)
    if missing or extra or len(values) != len(expected):
        raise ValueError(f"Evidence matrix mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    for key, (mean_km, std_km) in values.items():
        if mean_km - std_km <= 0:
            raise ValueError(f"Log-scale uncertainty crosses zero for {key}: {mean_km}, {std_km}")
    return values, dict(sorted(excluded_models.items()))


def read_confirmatory_evidence(
    path: Path,
) -> tuple[dict[tuple[str, int, str], tuple[float, float]], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    key_map = {
        "DLinear": "dlinear",
        "Transformer": "baseline",
        "PatchTST": "patchtst",
        "iTransformer": "itransformer",
        "PLGAFormer": "full",
    }
    values: dict[tuple[str, int, str], tuple[float, float]] = {}
    for method, model_key in key_map.items():
        for horizon in HORIZONS:
            for metric in METRIC_FIELDS.values():
                item = aggregate[model_key][str(horizon)][metric]
                values[(method, horizon, metric)] = (
                    float(item["mean"]) / 1000.0,
                    float(item["std"]) / 1000.0,
                )
    if len(values) != len(METHOD_ORDER) * len(HORIZONS) * len(METRIC_FIELDS):
        raise ValueError("Confirmatory evidence matrix is incomplete.")
    return values, {}


def render(values: dict[tuple[str, int, str], tuple[float, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True)

    for panel_index, (metric_label, metric_field) in enumerate(METRIC_FIELDS.items()):
        ax = axes[panel_index]
        for method in METHOD_ORDER:
            mean = np.array([values[(method, horizon, metric_field)][0] for horizon in HORIZONS])
            std = np.array([values[(method, horizon, metric_field)][1] for horizon in HORIZONS])
            if not np.all(np.isfinite(mean)) or np.any(mean <= 0):
                raise ValueError(f"Log-scale means must be finite and positive for {method}, {metric_label}")
            if not np.all(np.isfinite(std)) or np.any(std < 0) or np.any(mean - std <= 0):
                raise ValueError(f"Log-scale uncertainty must remain positive for {method}, {metric_label}")
            ax.errorbar(
                HORIZONS,
                mean,
                yerr=std,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=5.0 if method != "PLGAFormer" else 6.0,
                linewidth=1.25 if method == "PLGAFormer" else 0.95,
                elinewidth=0.7,
                capsize=2.0,
                capthick=0.7,
                label=method,
                zorder=3 if method == "PLGAFormer" else 2,
            )
        ax.set_yscale("log")
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast horizon (s)")
        ax.set_ylabel(f"{metric_label} (km)")
        ax.grid(axis="y", which="major", color="#D9DDE3", linewidth=0.55)
        ax.grid(axis="y", which="minor", color="#EEF0F3", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.text(
            0.02,
            0.98,
            "abc"[panel_index],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=8,
        )

    fig.legend(
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        columnspacing=1.1,
        handlelength=2.0,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.98, bottom=0.28, wspace=0.38)

    stem = output_dir / "fig_overall_performance"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source.suffix.lower() == ".json":
        values, excluded_models = read_confirmatory_evidence(args.source)
        uncertainty = "mean +/- population standard deviation across three frozen checkpoints"
        transformation = "mean and population standard deviation divided by 1000; logarithmic y axes"
        exclusion_reason = "none; compact confirmatory evidence contains only the five learning methods"
    else:
        values, excluded_models = read_evidence(args.source)
        uncertainty = "mean +/- sample standard deviation across three seeds"
        transformation = "mean_m and sample_std_m divided by 1000; logarithmic y axes"
        exclusion_reason = (
            "analytical models are retained in the source table but are outside the learning-based ranking claim"
        )
    render(values, args.output_dir)
    qa = {
        "figure": "fig_overall_performance",
        "source_file": args.source.name,
        "row_count": len(values),
        "methods": METHOD_ORDER,
        "horizons_s": HORIZONS,
        "metrics": list(METRIC_FIELDS),
        "units": "kilometers",
        "uncertainty": uncertainty,
        "transformation": transformation,
        "excluded_rows": sum(excluded_models.values()),
        "excluded_source_models": excluded_models,
        "exclusion_reason": exclusion_reason,
        "statistical_tests": "none",
    }
    (args.output_dir / "fig_overall_performance_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
