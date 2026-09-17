#!/usr/bin/env python3
"""Render the confirmatory multi-horizon comparison figure.

Figure contract
---------------
Core conclusion:
    After the 64 s prior lock is released, PLGAFormer remains the lowest-error
    learning method in ECEF ADE and FDE, with the gap versus iTransformer
    largest at 256 s.
Archetype:
    Quantitative grid; two aligned grouped-bar panels.
Evidence:
    Frozen confirmatory holdout, three checkpoints. The figure shows DLinear,
    Transformer, iTransformer, and PLGAFormer. PatchTST is omitted because its
    errors lie far above the linear axis; the ranking table retains all five
    methods and Cartesian RMSE.
Uncertainty:
    Mean and population standard deviation across three frozen checkpoints,
    converted from m to km. No significance stars.
Transform:
    Linear y-axis from zero. Hatched PLGAFormer bars at 32 s and 64 s mark the
    paper-facing prior lock. A shaded band covers those two horizons.
Output:
    IEEE subfigure panels (ADE, FDE) plus a patch legend bar and source CSV.
    Panel letters are assigned in LaTeX, not drawn in matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_taes_trajectory_case import (
    BAR_PANEL_SIZE_IEEE,
    configure_ieee_matplotlib,
    render_ieee_patch_legend,
    save_ieee_panel,
    style_ieee_bar_axes,
)


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
    "DLinear": "#8C96A0",
    "Transformer": "#66727D",
    "PatchTST": "#B79A63",
    "iTransformer": "#2F6B9A",
    "PLGAFormer": "#C23B3B",
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
FIGURE_METHODS = [
    "DLinear",
    "Transformer",
    "iTransformer",
    "PLGAFormer",
]
FIGURE_METRICS = {
    "ADE": "ade",
    "FDE": "fde",
}
LOCK_HORIZONS = {32, 64}
LOCK_BAND = "#E8EEF4"


def configure_matplotlib() -> None:
    configure_ieee_matplotlib()


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


def _method_legend_handles() -> list:
    return [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=METHOD_COLORS[method],
            edgecolor="#272727" if method == "PLGAFormer" else "#4D4D4D",
            linewidth=0.6,
        )
        for method in FIGURE_METHODS
    ]


def _bar_style(method: str) -> dict:
    is_proposed = method == "PLGAFormer"
    return {
        "color": METHOD_COLORS[method],
        "edgecolor": "#272727" if is_proposed else "#4D4D4D",
        "linewidth": 0.7 if is_proposed else 0.4,
        "error_kw": {
            "ecolor": "#272727",
            "elinewidth": 0.7,
            "capsize": 1.6,
            "capthick": 0.7,
        },
        "zorder": 3 if is_proposed else 2,
    }


def _reduction_at_256(
    values: dict[tuple[str, int, str], tuple[float, float]], metric_field: str
) -> float:
    proposed = values[("PLGAFormer", 256, metric_field)][0]
    comparator = values[("iTransformer", 256, metric_field)][0]
    return 100.0 * (comparator - proposed) / comparator


def _overall_legend_handles() -> tuple[list, list[str]]:
    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=METHOD_COLORS[method],
            edgecolor="#272727" if method == "PLGAFormer" else "#4D4D4D",
            linewidth=0.6,
        )
        for method in FIGURE_METHODS
    ]
    handles.append(
        Patch(
            facecolor="#E39A96",
            edgecolor="#C23B3B",
            hatch="////",
            label="PLGAFormer, prior lock",
        )
    )
    return handles, [*FIGURE_METHODS, "PLGAFormer, prior lock"]


def _draw_grouped_bars(
    ax: plt.Axes,
    values: dict[tuple[str, int, str], tuple[float, float]],
    metric_label: str,
    metric_field: str,
) -> None:
    x = np.arange(len(HORIZONS))
    n_methods = len(FIGURE_METHODS)
    width = 0.18
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width
    ymax = 0.0

    ax.axvspan(-0.52, 1.52, color=LOCK_BAND, linewidth=0, zorder=0)
    ax.text(
        0.5,
        0.97,
        "64 s prior lock",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        color="#5B6570",
        zorder=5,
    )

    for index, method in enumerate(FIGURE_METHODS):
        means = np.array([values[(method, horizon, metric_field)][0] for horizon in HORIZONS])
        stds = np.array([values[(method, horizon, metric_field)][1] for horizon in HORIZONS])
        if not np.all(np.isfinite(means)) or np.any(means < 0):
            raise ValueError(f"Invalid means for {method}, {metric_label}")
        if not np.all(np.isfinite(stds)) or np.any(stds < 0):
            raise ValueError(f"Invalid standard deviations for {method}, {metric_label}")
        ymax = max(ymax, float(np.max(means + stds)))
        is_proposed = method == "PLGAFormer"
        bars = ax.bar(
            x + offsets[index],
            means,
            width,
            yerr=stds,
            color=METHOD_COLORS[method],
            edgecolor="#272727" if is_proposed else "#4D4D4D",
            linewidth=0.7 if is_proposed else 0.4,
            error_kw={
                "ecolor": "#272727",
                "elinewidth": 0.7,
                "capsize": 1.6,
                "capthick": 0.7,
            },
            zorder=3 if is_proposed else 2,
            label=method,
        )
        for bar, horizon in zip(bars, HORIZONS, strict=True):
            if is_proposed and horizon in LOCK_HORIZONS:
                bar.set_hatch("////")
                bar.set_facecolor("#E39A96")
                bar.set_edgecolor("#C23B3B")

    ax.set_xticks(x, [str(horizon) for horizon in HORIZONS])
    ax.set_xlim(-0.62, len(HORIZONS) - 0.38)
    ax.set_ylim(0.0, ymax * 1.14)
    ax.set_xlabel("Forecast horizon (s)")
    ax.set_ylabel(f"{metric_label} (km)")
    style_ieee_bar_axes(ax, categorical="x")


def _write_source_csv(
    values: dict[tuple[str, int, str], tuple[float, float]], path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "horizon_s",
                "metric",
                "mean_km",
                "std_km",
                "shown_in_figure",
                "prior_lock",
            ],
        )
        writer.writeheader()
        for method in METHOD_ORDER:
            for horizon in HORIZONS:
                for metric_label, metric_field in METRIC_FIELDS.items():
                    mean_km, std_km = values[(method, horizon, metric_field)]
                    writer.writerow(
                        {
                            "method": method,
                            "horizon_s": horizon,
                            "metric": metric_label,
                            "mean_km": f"{mean_km:.6f}",
                            "std_km": f"{std_km:.6f}",
                            "shown_in_figure": (
                                method in FIGURE_METHODS and metric_field in FIGURE_METRICS.values()
                            ),
                            "prior_lock": method == "PLGAFormer" and horizon in LOCK_HORIZONS,
                        }
                    )


def render(
    values: dict[tuple[str, int, str], tuple[float, float]], output_dir: Path
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_ieee_matplotlib()
    reductions = {
        metric_field: _reduction_at_256(values, metric_field)
        for metric_field in FIGURE_METRICS.values()
    }
    for metric_label, metric_field in FIGURE_METRICS.items():
        fig, ax = plt.subplots(figsize=BAR_PANEL_SIZE_IEEE, constrained_layout=True)
        _draw_grouped_bars(ax, values, metric_label, metric_field)
        save_ieee_panel(fig, output_dir / f"fig_overall_performance_{metric_label.lower()}")
        plt.close(fig)
    handles, labels = _overall_legend_handles()
    render_ieee_patch_legend(
        output_dir,
        handles,
        labels,
        stem="fig_overall_performance_legend",
        ncol=5,
    )
    _write_source_csv(values, output_dir / "fig_overall_performance_source.csv")
    return reductions


def _draw_maneuver_panel(
    ax: plt.Axes,
    rows: list[dict],
    metric: str,
    groups: tuple[str, ...],
    group_labels: tuple[str, ...],
) -> None:
    x = np.arange(len(groups))
    n_methods = len(FIGURE_METHODS)
    width = 0.18
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width
    ymax = 0.0
    for index, method in enumerate(FIGURE_METHODS):
        data = [
            next(
                row
                for row in rows
                if row["method"] == method
                and row["maneuver"] == group
                and row["metric"] == metric
            )
            for group in groups
        ]
        means = np.asarray([row["mean_km"] for row in data])
        stds = np.asarray([row["std_km"] for row in data])
        if not np.all(np.isfinite(means)) or np.any(means < 0):
            raise ValueError(f"Invalid maneuver means for {method}, {metric}")
        if not np.all(np.isfinite(stds)) or np.any(stds < 0):
            raise ValueError(f"Invalid maneuver standard deviations for {method}, {metric}")
        ymax = max(ymax, float(np.max(means + stds)))
        ax.bar(x + offsets[index], means, width, yerr=stds, **_bar_style(method))
    ax.set_xticks(x, group_labels)
    ax.set_xlim(-0.62, len(groups) - 0.38)
    ax.set_ylim(0.0, ymax * 1.14)
    ax.set_ylabel(f"{metric.upper()} (km)")
    style_ieee_bar_axes(ax, categorical="x")


def _render_maneuver_panels(rows: list[dict], output_dir: Path) -> None:
    groups = ("longitudinal", "turning", "weaving")
    group_labels = ("Longitudinal", "Turning", "Weaving")
    configure_ieee_matplotlib()
    for metric in ("ade", "fde"):
        fig, ax = plt.subplots(figsize=BAR_PANEL_SIZE_IEEE, constrained_layout=True)
        _draw_maneuver_panel(ax, rows, metric, groups, group_labels)
        save_ieee_panel(fig, output_dir / f"fig_maneuver_performance_{metric}")
        plt.close(fig)
    render_ieee_patch_legend(
        output_dir,
        _method_legend_handles(),
        list(FIGURE_METHODS),
        stem="fig_maneuver_performance_legend",
        ncol=4,
    )


def read_figure_source_csv(
    path: Path,
) -> tuple[dict[tuple[str, int, str], tuple[float, float]], dict[str, int]]:
    values: dict[tuple[str, int, str], tuple[float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric_field = METRIC_FIELDS[row["metric"]]
            values[(row["method"], int(row["horizon_s"]), metric_field)] = (
                float(row["mean_km"]),
                float(row["std_km"]),
            )
    expected = {
        (method, horizon, metric_field)
        for method in METHOD_ORDER
        for horizon in HORIZONS
        for metric_field in METRIC_FIELDS.values()
    }
    if set(values) != expected:
        raise ValueError("Figure source CSV does not match the confirmatory matrix.")
    return values, {}


def read_maneuver_source_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "method": row["method"],
                    "maneuver": row["maneuver"],
                    "metric": row["metric"].lower(),
                    "mean_km": float(row["mean_km"]),
                    "std_km": float(row["std_km"]),
                }
            )
    return rows


def render_maneuver_from_csv(source: Path, output_dir: Path) -> None:
    rows = read_maneuver_source_csv(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    _render_maneuver_panels(rows, output_dir)
    qa = {
        "figure": "fig_maneuver_performance",
        "source_file": source.name,
        "row_count": len(rows),
        "methods_in_figure": FIGURE_METHODS,
        "methods_in_table_only": ["PatchTST"],
        "panel_letters": "LaTeX subcaptions: a ADE; b FDE",
        "output_assets": [
            "fig_maneuver_performance_legend.pdf",
            "fig_maneuver_performance_ade.pdf",
            "fig_maneuver_performance_fde.pdf",
        ],
    }
    (output_dir / "fig_maneuver_performance_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )


def render_maneuver(source: Path, records_dir: Path, output_dir: Path) -> None:
    """Compare 256 s ADE/FDE by maneuver on a linear grouped-bar axis.

    Core conclusion:
        The 256 s ranking versus iTransformer holds for longitudinal, turning,
        and weaving tracks; absolute error rises with commanded lateral change.
    Archetype:
        Quantitative grid; two aligned grouped-bar panels.
    Evidence:
        Frozen confirmatory holdout, 120 source trajectories per maneuver,
        three checkpoints. The figure shows DLinear, Transformer, iTransformer,
        and PLGAFormer. PatchTST is omitted because its errors lie far above
        the linear axis; the ranking table retains all five methods.
    """
    payload = json.loads(source.read_text(encoding="utf-8"))
    groups = ("longitudinal", "turning", "weaving")
    group_labels = ("Longitudinal", "Turning", "Weaving")
    model_keys = ("dlinear", "baseline", "patchtst", "itransformer", "full")
    if payload["seeds"] != [42, 123, 456]:
        raise ValueError("Expected the three frozen confirmatory seeds.")
    rows, sources = [], []
    for method, model_key in zip(METHOD_ORDER, model_keys, strict=True):
        records = []
        for seed in payload["seeds"]:
            path = records_dir / f"{model_key}_seed{seed}.json"
            raw = path.read_bytes()
            record = json.loads(raw)
            if (record["model_key"] != model_key or record["seed"] != seed
                    or record["evidence_tier"] != "confirmatory_final"
                    or record["test_evaluation_performed"] is not True):
                raise ValueError(f"Invalid final record: {path.name}")
            for key in ("dataset_sha256", "protocol_document_sha256", "main_bundle_id"):
                if record[key] != payload[key]:
                    raise ValueError(f"Evidence identity mismatch: {path.name}, {key}")
            records.append(record)
            sources.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                            "checkpoint_sha256": record["checkpoint_sha256"],
                            "formal_config_sha256": record["formal_config_sha256"]})
        for group in groups:
            items = [r["maneuver_metrics"]["256"][group] for r in records]
            if any(item["trajectory_count"] != 120 for item in items):
                raise ValueError("Expected 120 trajectories per maneuver.")
            for metric in ("ade", "fde"):
                values = np.asarray([item[metric] for item in items], dtype=float) / 1000.0
                if not np.all(np.isfinite(values)) or np.any(values <= 0):
                    raise ValueError("Invalid maneuver errors.")
                rows.append({"method": method, "maneuver": group, "metric": metric,
                             "mean_km": float(values.mean()),
                             "std_km": float(values.std(ddof=0)),
                             "seed_values_km": values.tolist()})
        for metric in ("ade", "fde"):
            mean = np.mean([row["mean_km"] for row in rows
                            if row["method"] == method and row["metric"] == metric])
            if not np.isclose(mean, payload["aggregate"][model_key]["256"][metric]["mean"] / 1000):
                raise ValueError("Subgroup means do not reconstruct overall evidence.")
    if len({item["formal_config_sha256"] for item in sources}) != 1:
        raise ValueError("Mixed formal configurations.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _render_maneuver_panels(rows, output_dir)

    source_csv = output_dir / "fig_maneuver_performance_source.csv"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "maneuver",
                "metric",
                "mean_km",
                "std_km",
                "shown_in_figure",
                "n_trajectories",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "maneuver": row["maneuver"],
                    "metric": row["metric"].upper(),
                    "mean_km": f"{row['mean_km']:.6f}",
                    "std_km": f"{row['std_km']:.6f}",
                    "shown_in_figure": row["method"] in FIGURE_METHODS,
                    "n_trajectories": 120,
                }
            )

    qa = {
        "figure": "fig_maneuver_performance",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sources": sources,
        "row_count": len(rows),
        "trajectory_counts": dict.fromkeys(groups, 120),
        "seeds": payload["seeds"],
        "horizon_s": 256,
        "methods_in_figure": FIGURE_METHODS,
        "methods_in_table_only": ["PatchTST"],
        "uncertainty": "population SD across three frozen checkpoints; ddof=0",
        "scope": "256 s ADE/FDE by commanded lateral group on the confirmatory holdout",
        "transformation": (
            "meters to kilometers; linear y-axis from zero; PatchTST omitted from the figure"
        ),
        "statistical_tests": "none",
        "panel_letters": "LaTeX subcaptions: a ADE; b FDE",
        "output_assets": [
            "fig_maneuver_performance_legend.pdf",
            "fig_maneuver_performance_ade.pdf",
            "fig_maneuver_performance_fde.pdf",
            "fig_maneuver_performance_source.csv",
        ],
    }
    (output_dir / "fig_maneuver_performance_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("overall", "maneuver"), default="overall")
    parser.add_argument("--records-dir", type=Path)
    args = parser.parse_args()
    if args.view == "maneuver":
        if args.source.suffix.lower() == ".csv":
            render_maneuver_from_csv(args.source, args.output_dir)
            return
        if args.records_dir is None:
            parser.error("--records-dir is required for the maneuver JSON view")
        render_maneuver(args.source, args.records_dir, args.output_dir)
        return
    if args.source.suffix.lower() == ".json":
        values, excluded_models = read_confirmatory_evidence(args.source)
        uncertainty = "mean +/- population standard deviation across three frozen checkpoints"
        transformation = (
            "meters to kilometers; linear y-axis from zero; hatched PLGAFormer bars "
            "mark the 64 s prior lock; PatchTST omitted from the figure"
        )
        exclusion_reason = "none; compact confirmatory evidence contains only the five learning methods"
    elif args.source.name.endswith("_source.csv") or "horizon_s" in args.source.read_text(encoding="utf-8")[:200]:
        values, excluded_models = read_figure_source_csv(args.source)
        uncertainty = "mean +/- population standard deviation across three frozen checkpoints"
        transformation = (
            "kilometers from frozen figure source CSV; linear y-axis from zero; "
            "hatched PLGAFormer bars mark the 64 s prior lock; PatchTST omitted from the figure"
        )
        exclusion_reason = "none; restyle from frozen figure source"
    else:
        values, excluded_models = read_evidence(args.source)
        uncertainty = "mean +/- sample standard deviation across three seeds"
        transformation = (
            "mean_m and sample_std_m divided by 1000; linear y-axis from zero; "
            "hatched PLGAFormer bars mark the 64 s prior lock; PatchTST omitted from the figure"
        )
        exclusion_reason = (
            "analytical models are retained in the source table but are outside the learning-based ranking claim"
        )
    reductions = render(values, args.output_dir)
    qa = {
        "figure": "fig_overall_performance",
        "source_file": args.source.name,
        "row_count": len(values),
        "methods_in_figure": FIGURE_METHODS,
        "methods_in_table_only": ["PatchTST"],
        "metrics_in_figure": list(FIGURE_METRICS),
        "horizons_s": HORIZONS,
        "metrics": list(METRIC_FIELDS),
        "units": "kilometers",
        "uncertainty": uncertainty,
        "transformation": transformation,
        "excluded_rows": sum(excluded_models.values()),
        "excluded_source_models": excluded_models,
        "exclusion_reason": exclusion_reason,
        "statistical_tests": "none",
        "relative_reduction_vs_itransformer_at_256s_percent": reductions,
        "output_assets": [
            "fig_overall_performance_legend.pdf",
            "fig_overall_performance_ade.pdf",
            "fig_overall_performance_fde.pdf",
            "fig_overall_performance_source.csv",
        ],
        "panel_letters": "LaTeX subcaptions: a ADE; b FDE",
        "n_checkpoints": 3,
    }
    (args.output_dir / "fig_overall_performance_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
