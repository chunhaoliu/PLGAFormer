#!/usr/bin/env python3
"""Render the 256 s ablation ADE/FDE bars from the frozen manuscript CSV.

Scientific content matches Table ablation: four configurations, means only,
no error bars. Configuration order matches the table. Panel letters are
assigned in LaTeX.
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
    save_ieee_panel,
    style_ieee_bar_axes,
)

BAR_COLOR = "#8BB565"
BAR_EDGE = "#536944"
CONFIG_ORDER = (
    "Learned-only backbone",
    "Spherical prior + adaptive fusion",
    "Rotating-Earth prior + fixed schedule",
    "PLGAFormer (full)",
)
DISPLAY_LABELS = {
    "Learned-only backbone": "Learned-only\nbackbone",
    "Spherical prior + adaptive fusion": "Spherical prior\n+ adaptive fusion",
    "Rotating-Earth prior + fixed schedule": "Rotating-Earth prior\n+ fixed schedule",
    "PLGAFormer (full)": "PLGAFormer (full)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "configuration": row["configuration"],
                    "ade": float(row["ADE_mean_km"]),
                    "fde": float(row["FDE_mean_km"]),
                }
            )
    names = [str(row["configuration"]) for row in rows]
    if names != list(CONFIG_ORDER):
        raise RuntimeError(f"Ablation CSV order changed: {names}")
    if not (rows[-1]["ade"] < rows[-2]["ade"] and rows[-1]["fde"] < rows[-2]["fde"]):
        raise RuntimeError("PLGAFormer (full) is not lowest among the four configurations.")
    return rows


def draw_panel(ax: plt.Axes, rows: list[dict[str, float | str]], metric: str, xlabel: str) -> None:
    y = np.arange(len(rows))
    means = np.array([float(row[metric]) for row in rows])
    labels = [DISPLAY_LABELS[str(row["configuration"])] for row in rows]
    bars = ax.barh(
        y,
        means,
        height=0.62,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.5,
        zorder=3,
    )
    xmax = float(np.max(means))
    ax.set_xlim(0.0, xmax * 1.28)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    style_ieee_bar_axes(ax, categorical="y")
    for bar, value in zip(bars, means, strict=True):
        ax.text(
            bar.get_width() + 0.04 * xmax,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            clip_on=False,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.source_csv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_ieee_matplotlib()
    for metric, xlabel, stem in (
        ("ade", "ADE (km)", "fig_ablation_summary_ade"),
        ("fde", "FDE (km)", "fig_ablation_summary_fde"),
    ):
        fig, ax = plt.subplots(figsize=BAR_PANEL_SIZE_IEEE, constrained_layout=True)
        draw_panel(ax, rows, metric, xlabel)
        save_ieee_panel(fig, output_dir / stem)
        plt.close(fig)
    print({"output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
