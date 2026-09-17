#!/usr/bin/env python3
"""Render the 256 s dynamics-shift comparison figure.

The figure shows only Transformer and PLGAFormer, matching Table robustness.
Rotating-Earth 3-DOF is omitted because it is the internal prior, not a
ranked learning method. PatchTST and the other public comparators were not
evaluated under the frozen-checkpoint dynamics-shift protocol.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_taes_trajectory_case import (
    BAR_PANEL_SIZE_IEEE,
    configure_ieee_matplotlib,
    render_ieee_patch_legend,
    save_ieee_panel,
    style_ieee_bar_axes,
)

SOURCE_CSV = PROJECT_ROOT / "PublicRelease" / "evidence" / "robustness.csv"
CONDITION_ORDER = ("nominal", "aero_shift", "ballistic_shift")
CONDITION_LABELS = {
    "nominal": "Nominal",
    "aero_shift": r"$C_L{-}10\%$, $C_D{+}10\%$",
    "ballistic_shift": r"$m{+}15\%$, $S{-}10\%$",
}
METHODS = ("transformer", "plgaformer")
DISPLAY_NAMES = {
    "transformer": "Transformer",
    "plgaformer": "PLGAFormer",
}
COLORS = {
    "transformer": "#66727D",
    "plgaformer": "#C23B3B",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "figure_review" / "robustness",
    )
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    return parser.parse_args(argv)


def load_rows(path: Path) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    values: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = str(row["model"])
            if model not in METHODS:
                continue
            condition = str(row["condition"])
            values[(condition, model)] = (
                float(row["ade_mean_m"]) / 1000.0,
                float(row["ade_sample_std_m"]) / 1000.0,
                float(row["fde_mean_m"]) / 1000.0,
                float(row["fde_sample_std_m"]) / 1000.0,
            )
    missing = [
        (condition, model)
        for condition in CONDITION_ORDER
        for model in METHODS
        if (condition, model) not in values
    ]
    if missing:
        raise RuntimeError(f"Robustness CSV is missing rows: {missing}")
    return values


def _legend_handles() -> tuple[list, list[str]]:
    from matplotlib.patches import Rectangle

    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS[method],
            edgecolor="#272727" if method == "plgaformer" else "#4D4D4D",
            linewidth=0.6,
        )
        for method in METHODS
    ]
    return handles, [DISPLAY_NAMES[method] for method in METHODS]


def draw_panel(
    ax: plt.Axes,
    values: dict[tuple[str, str], tuple[float, float, float, float]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    x = np.arange(len(CONDITION_ORDER))
    width = 0.34
    offsets = (-0.18, 0.18)
    ymax = 0.0
    index = 0 if metric == "ade" else 2
    for method, offset in zip(METHODS, offsets, strict=True):
        means = np.array([values[(condition, method)][index] for condition in CONDITION_ORDER])
        stds = np.array([values[(condition, method)][index + 1] for condition in CONDITION_ORDER])
        ymax = max(ymax, float(np.max(means + stds)))
        is_proposed = method == "plgaformer"
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            color=COLORS[method],
            edgecolor="#272727" if is_proposed else "#4D4D4D",
            linewidth=0.7 if is_proposed else 0.4,
            error_kw={
                "ecolor": "#272727",
                "elinewidth": 0.7,
                "capsize": 1.6,
                "capthick": 0.7,
            },
            zorder=3 if is_proposed else 2,
            label=DISPLAY_NAMES[method],
        )
    ax.set_xticks(x, [CONDITION_LABELS[condition] for condition in CONDITION_ORDER])
    ax.set_xlim(-0.55, len(CONDITION_ORDER) - 0.45)
    ax.set_ylim(0.0, ymax * 1.16)
    ax.set_ylabel(ylabel)
    style_ieee_bar_axes(ax, categorical="x")
    ax.tick_params(axis="x", labelsize=6.5)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    values = load_rows(args.source_csv)
    for metric in ("ade", "fde"):
        proposed = [values[(condition, "plgaformer")][0 if metric == "ade" else 2] for condition in CONDITION_ORDER]
        baseline = [values[(condition, "transformer")][0 if metric == "ade" else 2] for condition in CONDITION_ORDER]
        if not all(p < b for p, b in zip(proposed, baseline, strict=True)):
            raise RuntimeError(f"PLGAFormer does not lead Transformer on {metric}.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_ieee_matplotlib()
    for metric, ylabel in (("ade", "ADE (km)"), ("fde", "FDE (km)")):
        fig, ax = plt.subplots(figsize=BAR_PANEL_SIZE_IEEE, constrained_layout=True)
        draw_panel(ax, values, metric=metric, ylabel=ylabel)
        save_ieee_panel(fig, output_dir / f"fig_robustness_{metric}")
        plt.close(fig)
    handles, labels = _legend_handles()
    render_ieee_patch_legend(output_dir, handles, labels, stem="fig_robustness_legend", ncol=2)
    print({"output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
