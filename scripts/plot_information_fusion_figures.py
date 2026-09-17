#!/usr/bin/env python3
"""Render the Information Fusion manuscript figures from frozen evidence.

This module changes only visual encoding and panel composition. It never runs
training, changes a metric, selects a new qualitative case, or edits evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Okabe--Ito: a fixed, color-vision-safe publication palette. Method colors
# are invariant across every figure in the manuscript.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}
COLORS = {
    "dlinear": OKABE_ITO["sky_blue"], "baseline": OKABE_ITO["blue"],
    "itransformer": OKABE_ITO["bluish_green"], "full": OKABE_ITO["vermillion"],
    "truth": OKABE_ITO["black"], "lock": "#F0F0F0",
    "lock_edge": "#6B6B6B", "grid": "#D9D9D9",
    "axis": "#333333", "text": "#222222",
    "scenario_longitudinal": OKABE_ITO["blue"],
    "scenario_turning": OKABE_ITO["bluish_green"],
    "scenario_weaving": OKABE_ITO["vermillion"],
    "start": OKABE_ITO["black"], "end": OKABE_ITO["reddish_purple"],
}
DISPLAY_NAMES = {
    "dlinear": "DLinear", "baseline": "Transformer",
    "itransformer": "iTransformer", "full": "PLGAFormer",
}
METHOD_ORDER = ("dlinear", "baseline", "itransformer", "full")
METHOD_CSV_NAMES = {
    "dlinear": "DLinear", "baseline": "Transformer",
    "itransformer": "iTransformer", "full": "PLGAFormer",
}
LINESTYLES = {
    "dlinear": (0, (1.2, 1.4)), "baseline": (0, (4.0, 1.8)),
    "itransformer": (0, (5.0, 1.8, 1.2, 1.8)), "full": "-",
}
LINEWIDTHS = {"dlinear": 1.15, "baseline": 1.25, "itransformer": 1.35, "full": 2.0}
MARKERS = {"dlinear": "o", "baseline": "s", "itransformer": "^", "full": "D"}
PANEL_LABELS = "abcdefghijklmnopqrstuvwxyz"
require_matplotlib_panel_alignment = None


def load_alignment_auditor(scripts_dir: Path | None) -> None:
    """Load the optional publication-layout gate without hard-coding a user path."""
    global require_matplotlib_panel_alignment
    if scripts_dir is None:
        return
    scripts_dir = scripts_dir.resolve()
    if not (scripts_dir / "audit_panel_alignment.py").is_file():
        raise FileNotFoundError(f"Missing alignment auditor in {scripts_dir}")
    sys.path.insert(0, str(scripts_dir))
    import audit_panel_alignment as alignment

    def engineering_panel_label_anchor(ax):
        """Recognize both ``a`` and engineering-style ``(a)`` panel labels."""
        for artist in ax.texts:
            label = artist.get_text().strip()
            match = re.fullmatch(r"\(([a-z])\)(?:\s+.+)?", label)
            if match is None:
                continue
            display = artist.get_transform().transform(artist.get_position())
            axes_xy = ax.transAxes.inverted().transform(display)
            if not (0.3 <= axes_xy[0] <= 0.7 and -0.45 <= axes_xy[1] <= 0.1):
                continue
            inches = ax.figure.dpi_scale_trans.inverted().transform(display)
            return match.group(1), [
                float(inches[0] * 72), float(inches[1] * 72)
            ]
        return None

    alignment._matplotlib_panel_label_anchor = engineering_panel_label_anchor
    require_matplotlib_panel_alignment = alignment.require_matplotlib_panel_alignment


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.5,
        "axes.titleweight": "semibold", "axes.labelcolor": COLORS["text"],
        "axes.edgecolor": COLORS["axis"], "axes.linewidth": 0.7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.color": COLORS["text"], "ytick.color": COLORS["text"],
        "legend.fontsize": 7.0, "legend.frameon": True,
        "legend.facecolor": "white", "legend.edgecolor": "#BDBDBD",
        "legend.framealpha": 0.94, "legend.fancybox": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 600, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def finite_nonnegative(values: Iterable[float], label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"Invalid nonnegative values for {label}: {array}")
    return array


def style_axes(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(direction="out", length=2.6, width=0.65, pad=2.2)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.55, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, index: int, descriptor: str, *,
                y_pt: float = -31.0) -> None:
    transform = ax.transAxes + ScaledTranslation(
        0.0, y_pt / 72.0, ax.figure.dpi_scale_trans)
    ax.text(0.5, 0.0, f"({PANEL_LABELS[index]}) {descriptor}", transform=transform,
            ha="center", va="top", fontsize=8.2, fontweight="normal",
            color=COLORS["text"], clip_on=False)


def method_handles(include_truth: bool = False) -> list[Line2D]:
    handles: list[Line2D] = []
    if include_truth:
        handles.append(Line2D([0], [0], color=COLORS["truth"], linewidth=1.8,
                              label="Truth"))
    for key in METHOD_ORDER:
        handles.append(Line2D([0], [0], color=COLORS[key],
                              linestyle=LINESTYLES[key], linewidth=LINEWIDTHS[key],
                              label=DISPLAY_NAMES[key]))
    return handles


def inside_legend(ax: plt.Axes, handles: list, *, loc: str = "upper left",
                  ncol: int = 2, fontsize: float = 6.8,
                  handlelength: float = 1.75) -> None:
    """Place a compact engineering-style legend inside the plotting area."""
    legend = ax.legend(
        handles=handles, loc=loc, ncol=ncol, fontsize=fontsize,
        handlelength=handlelength, handletextpad=0.38, columnspacing=0.72,
        borderpad=0.34, labelspacing=0.28,
    )
    legend.get_frame().set_linewidth(0.55)


def save_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    tiff_path = output_dir / f"{stem}.tiff"
    if require_matplotlib_panel_alignment is not None:
        comparable_axes = [
            ax for ax in fig.axes
            if getattr(ax, "get_subplotspec", lambda: None)() is not None
        ]
        inset_axes_list = [ax for ax in fig.axes if ax not in comparable_axes]
        require_matplotlib_panel_alignment(
            fig,
            axes=comparable_axes,
            exclude_axes=inset_axes_list,
            json_out=output_dir / f"{stem}.alignment.json",
            overlay_svg=output_dir / f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return {
        "pdf": pdf_path.name,
        "svg": svg_path.name,
        "png": png_path.name,
        "tiff": tiff_path.name,
    }


def render_overall(source: Path, output_dir: Path) -> dict:
    rows = read_csv(source)
    horizons, metrics = (32, 64, 128, 256), ("ADE", "FDE")
    reverse = {v: k for k, v in METHOD_CSV_NAMES.items()}
    values: dict[tuple[str, int, str], tuple[float, float]] = {}
    for row in rows:
        if row["method"] in reverse and row["metric"] in metrics:
            values[(reverse[row["method"]], int(row["horizon_s"]), row["metric"])] = (
                float(row["mean_km"]), float(row["std_km"]))
    expected = {(k, h, m) for k in METHOD_ORDER for h in horizons for m in metrics}
    if set(values) != expected:
        raise ValueError(f"Overall matrix mismatch: missing={sorted(expected-set(values))}")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.30))
    x = np.arange(len(horizons), dtype=float)
    for p, (ax, metric) in enumerate(zip(axes, metrics, strict=True)):
        ymax = 0.0
        ax.axvspan(-0.25, 1.25, color=COLORS["lock"], alpha=0.24,
                   linewidth=0, zorder=0)
        for key in METHOD_ORDER:
            means = finite_nonnegative((values[(key, h, metric)][0] for h in horizons),
                                       f"overall {key} {metric}")
            stds = finite_nonnegative((values[(key, h, metric)][1] for h in horizons),
                                      f"overall std {key} {metric}")
            ymax = max(ymax, float(np.max(means + stds)))
            ax.fill_between(
                x, np.maximum(0.0, means-stds), means+stds,
                color=COLORS[key], alpha=0.15 if key == "full" else 0.10,
                linewidth=0, zorder=1,
            )
            ax.plot(
                x, means, color=COLORS[key],
                linestyle=LINESTYLES[key], linewidth=LINEWIDTHS[key],
                solid_capstyle="round", zorder=5 if key == "full" else 3,
            )
        ax.set_xticks(x, [str(h) for h in horizons])
        ax.set_xlim(-0.35, 3.18); ax.set_ylim(0, ymax * 1.27)
        ax.set_xlabel("Forecast horizon (s)"); ax.set_ylabel(f"{metric} (km)")
        style_axes(ax, grid=None); panel_label(ax, p, metric)
        inside_legend(ax, method_handles(), loc="upper right", ncol=2)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.27, top=0.94, wspace=0.31)
    return {"outputs": save_bundle(fig, output_dir, "fig_overall_performance"),
            "rows": len(rows), "source_sha256": sha256_file(source)}


def render_maneuver(source: Path, output_dir: Path) -> dict:
    rows = read_csv(source)
    maneuvers = ("longitudinal", "turning", "weaving")
    group_labels, metrics = ("Longitudinal", "Turning", "Weaving"), ("ADE", "FDE")
    reverse = {v: k for k, v in METHOD_CSV_NAMES.items()}
    values: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in rows:
        metric = row["metric"].upper()
        if row["method"] in reverse and metric in metrics:
            values[(reverse[row["method"]], row["maneuver"], metric)] = (
                float(row["mean_km"]), float(row["std_km"]))
    expected = {(k, g, m) for k in METHOD_ORDER for g in maneuvers for m in metrics}
    if set(values) != expected:
        raise ValueError(f"Maneuver matrix mismatch: missing={sorted(expected-set(values))}")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35))
    x = np.arange(len(maneuvers), dtype=float)
    for p, (ax, metric) in enumerate(zip(axes, metrics, strict=True)):
        ymax = 0.0
        for key in METHOD_ORDER:
            means = finite_nonnegative((values[(key, g, metric)][0] for g in maneuvers),
                                       f"maneuver {key} {metric}")
            stds = finite_nonnegative((values[(key, g, metric)][1] for g in maneuvers),
                                      f"maneuver std {key} {metric}")
            ymax = max(ymax, float(np.max(means + stds)))
            ax.fill_between(
                x, np.maximum(0.0, means-stds), means+stds,
                color=COLORS[key], alpha=0.13 if key == "full" else 0.075,
                linewidth=0, zorder=1,
            )
            ax.plot(
                x, means, color=COLORS[key], linestyle=LINESTYLES[key],
                linewidth=2.05 if key == "full" else 1.35,
                solid_capstyle="round", zorder=4 if key == "full" else 3,
            )
        ax.set_xticks(x, group_labels)
        ax.set_xlim(-0.06, 2.06); ax.set_ylim(0, ymax * 1.27)
        ax.set_ylabel(f"{metric} (km)")
        style_axes(ax, grid="y"); panel_label(ax, p, metric, y_pt=-21.0)
        inside_legend(ax, method_handles(), loc="upper left", ncol=2)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.94, wspace=0.31)
    return {"outputs": save_bundle(fig, output_dir, "fig_maneuver_performance"),
            "rows": len(rows), "source_sha256": sha256_file(source)}


def render_scenario(source: Path, output_dir: Path) -> dict:
    rows = read_csv(source)
    strata = (
        "quasi_equilibrium__longitudinal", "quasi_equilibrium__turning",
        "quasi_equilibrium__weaving", "skip_glide__longitudinal",
        "skip_glide__turning", "skip_glide__weaving",
    )
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in strata}
    for row in rows:
        if row["joint_stratum"] not in grouped:
            raise ValueError(f"Unexpected scenario stratum: {row['joint_stratum']}")
        grouped[row["joint_stratum"]].append(row)
    if any(len(grouped[name]) != 1000 for name in strata):
        raise ValueError("Expected 1000 source samples in every scenario panel")

    longitude = np.asarray([float(r["delta_longitude_deg"]) for r in rows])
    latitude = np.asarray([float(r["delta_latitude_deg"]) for r in rows])
    altitude = np.asarray([float(r["altitude_km"]) for r in rows])
    if not all(np.isfinite(x).all() for x in (longitude, latitude, altitude)):
        raise ValueError("Non-finite scenario coordinates")
    lon_lim = max(abs(float(longitude.min())), abs(float(longitude.max()))) * 1.06
    lat_lim = max(abs(float(latitude.min())), abs(float(latitude.max()))) * 1.06
    alt_min, alt_max = float(altitude.min()), float(altitude.max())
    alt_pad = max(1.0, 0.04 * (alt_max - alt_min))

    fig = plt.figure(figsize=(7.15, 5.15))
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    row_titles = ("Quasi-equilibrium", "Skip-glide")
    maneuver_colors = {
        "longitudinal": COLORS["scenario_longitudinal"],
        "turning": COLORS["scenario_turning"],
        "weaving": COLORS["scenario_weaving"],
    }
    for i, (ax, stratum) in enumerate(zip(axes, strata, strict=True)):
        _, maneuver = stratum.split("__")
        panel = grouped[stratum]
        lon = np.asarray([float(r["delta_longitude_deg"]) for r in panel])
        lat = np.asarray([float(r["delta_latitude_deg"]) for r in panel])
        alt = np.asarray([float(r["altitude_km"]) for r in panel])
        ax.plot(lon, lat, alt, color=maneuver_colors[maneuver], linewidth=1.55)
        ax.plot(lon, lat, np.full_like(alt, alt_min), color="#BDBDBD",
                linewidth=0.8, alpha=0.95)
        ax.scatter(lon[0], lat[0], alt[0], s=18, color=COLORS["start"],
                   edgecolor="white", linewidth=0.45, depthshade=False, zorder=5)
        ax.scatter(lon[-1], lat[-1], alt[-1], s=18, marker="s",
                   color=COLORS["end"], edgecolor="white", linewidth=0.45,
                   depthshade=False, zorder=5)
        ax.set_xlim(-lon_lim, lon_lim); ax.set_ylim(-lat_lim, lat_lim)
        ax.set_zlim(alt_min - alt_pad, alt_max + alt_pad)
        ax.set_yticks([])
        ax.view_init(elev=24, azim=-61); ax.set_proj_type("ortho")
        ax.set_box_aspect((1.25, 1.0, 0.82))
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("#B8B8B8")
        ax.yaxis.line.set_linewidth(0.0)
        ax.tick_params(labelsize=6.4, pad=6.0, colors=COLORS["text"])
        # Repeating long labels in each compact 3-D panel obscures projected
        # tick labels, so the common axis definitions are given once below.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        if i % 3 == 0:
            ax.text2D(-0.20, 0.50, row_titles[i // 3], transform=ax.transAxes,
                      rotation=90, rotation_mode="anchor", ha="center", va="center",
                      fontsize=7.2, fontweight="semibold", color=COLORS["text"])
        label_transform = ax.transAxes + ScaledTranslation(
            0.0, -12.0 / 72.0, fig.dpi_scale_trans)
        ax.text2D(0.5, 0.0, f"({PANEL_LABELS[i]}) {maneuver.title()}",
                  transform=label_transform, ha="center", va="top",
                  fontsize=8.0, fontweight="normal", color=COLORS["text"])
    fig.text(0.985, 0.48, "h (km)", rotation=90, rotation_mode="anchor",
             ha="center", va="center", fontsize=6.8, color=COLORS["text"])
    fig.text(0.51, 0.015,
             rf"Horizontal axes: $\Delta\lambda$ and $\Delta\phi$ (deg); shared limits $\pm${lon_lim:.1f} and $\pm${lat_lim:.1f}",
             ha="center", va="bottom", fontsize=6.8, color=COLORS["text"])
    fig.subplots_adjust(left=0.04, right=0.965, bottom=0.13, top=0.97,
                        wspace=0.02, hspace=0.27)
    return {"outputs": save_bundle(fig, output_dir, "fig_scenario_evidence"),
            "rows": len(rows), "source_sha256": sha256_file(source)}


def _spatial_limits(points: np.ndarray, pad_fraction: float = 0.14):
    xy = np.asarray(points, dtype=float)[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = np.maximum(span * pad_fraction, 3.0)
    return (float(lo[0]-pad[0]), float(hi[0]+pad[0])), (
        float(lo[1]-pad[1]), float(hi[1]+pad[1]))


def render_typical(source: Path, output_dir: Path) -> dict:
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key], dtype=np.float64) for key in data.files}
    maneuvers = ("longitudinal", "turning")
    for maneuver in maneuvers:
        if arrays[f"{maneuver}_truth_enu"].shape != (256, 3):
            raise ValueError(f"Unexpected truth array for {maneuver}")
        for key in METHOD_ORDER:
            if arrays[f"{maneuver}_{key}_enu"].shape != (3, 256, 3):
                raise ValueError(f"Unexpected prediction array for {maneuver}, {key}")
            if arrays[f"{maneuver}_{key}_err"].shape != (3, 256):
                raise ValueError(f"Unexpected error array for {maneuver}, {key}")

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.85))
    time_s, terminal = np.arange(1, 257), 64
    error_ceiling = max(
        float(np.max(arrays[f"{m}_{k}_err"].mean(axis=0) +
                     arrays[f"{m}_{k}_err"].std(axis=0, ddof=0)))
        for m in maneuvers for k in METHOD_ORDER)
    for col, maneuver in enumerate(maneuvers):
        ax = axes[0, col]
        truth = arrays[f"{maneuver}_truth_enu"][-terminal:]
        ax.plot(truth[:, 0], truth[:, 1], color=COLORS["truth"],
                linewidth=1.8, zorder=5, rasterized=True)
        for key in METHOD_ORDER:
            seeds = arrays[f"{maneuver}_{key}_enu"][:, -terminal:, :]
            for seed in seeds:
                ax.plot(seed[:, 0], seed[:, 1], color=COLORS[key],
                        linewidth=0.6 if key == "full" else 0.45,
                        alpha=0.20 if key == "full" else 0.10,
                        rasterized=True)
            mean = seeds.mean(axis=0)
            ax.plot(mean[:, 0], mean[:, 1], color=COLORS[key],
                    linestyle=LINESTYLES[key], linewidth=LINEWIDTHS[key],
                    zorder=4 if key == "full" else 3, rasterized=True)
            ax.plot(mean[-1, 0], mean[-1, 1], marker="o", markersize=3.5,
                    color=COLORS[key], linestyle="none", zorder=6,
                    rasterized=True)
        ax.plot(truth[-1, 0], truth[-1, 1], marker="s", markersize=4.0,
                color=COLORS["truth"], linestyle="none", zorder=6,
                rasterized=True)
        (x0, x1), (y0, y1) = _spatial_limits(truth)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xlabel("East displacement (km)", labelpad=6.0)
        ax.set_ylabel("North displacement (km)")
        style_axes(ax)
        ax.tick_params(axis="x", length=2.0, pad=4.5)
        panel_label(ax, col, f"{maneuver.title()} trajectory")
    for col, maneuver in enumerate(maneuvers):
        ax = axes[1, col]
        for key in METHOD_ORDER:
            values = arrays[f"{maneuver}_{key}_err"]
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
            ax.fill_between(time_s, np.maximum(0.0, mean-std), mean+std,
                            color=COLORS[key],
                            alpha=0.16 if key == "full" else 0.075,
                            linewidth=0, zorder=1)
            ax.plot(time_s, mean, color=COLORS[key],
                    linestyle=LINESTYLES[key], linewidth=LINEWIDTHS[key],
                    zorder=4 if key == "full" else 3, rasterized=True)
        ax.axvspan(1, 64, color=COLORS["lock"], alpha=0.24,
                   linewidth=0, zorder=0)
        ax.axvline(64, color=COLORS["lock_edge"], linewidth=0.8,
                   linestyle="--", zorder=2)
        ax.set_xlim(1, 256); ax.set_xticks([1, 64, 128, 192, 256])
        ax.set_ylim(0, error_ceiling * 1.26)
        ax.set_xlabel("Forecast time (s)")
        ax.set_ylabel("3-D position error (km)")
        style_axes(ax); panel_label(ax, col+2, f"{maneuver.title()} error")
    inside_legend(axes[1, 0], method_handles(include_truth=True),
                  loc="upper left", ncol=3, fontsize=6.3)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.15, top=0.97,
                        wspace=0.27, hspace=0.62)
    return {"outputs": save_bundle(fig, output_dir, "fig_typical_maneuver_cases"),
            "array_count": len(arrays), "source_sha256": sha256_file(source)}


def render_altitude(source: Path, selection_path: Path, output_dir: Path) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    maneuvers = ("longitudinal", "turning")
    for maneuver in maneuvers:
        if arrays[f"{maneuver}_truth_alt_km"].shape != (256,):
            raise ValueError(f"Unexpected altitude truth for {maneuver}")
        for key in METHOD_ORDER:
            if arrays[f"{maneuver}_{key}_alt_km"].shape != (3, 256):
                raise ValueError(f"Unexpected altitude prediction for {maneuver}, {key}")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.85))
    time_s = np.arange(1, 257)
    for col, maneuver in enumerate(maneuvers):
        truth = np.asarray(arrays[f"{maneuver}_truth_alt_km"], dtype=float)
        full_values = [truth]
        z0, z1 = [int(v) for v in selection[maneuver]["zoom_s"]]
        slice_obj = slice(z0-1, z1)

        ax = axes[col]
        ax.axvspan(1, 64, color=COLORS["lock"], alpha=0.24,
                   linewidth=0, zorder=0)
        ax.plot(time_s, truth, color=COLORS["truth"], linewidth=1.8, zorder=5)
        zoom_values = [truth[slice_obj]]
        for key in METHOD_ORDER:
            seeds = np.asarray(arrays[f"{maneuver}_{key}_alt_km"], dtype=float)
            full_values.append(seeds.ravel())
            zoom_values.append(seeds[:, slice_obj].ravel())
            for seed in seeds:
                ax.plot(time_s, seed, color=COLORS[key],
                        linewidth=0.6 if key == "full" else 0.45,
                        alpha=0.20 if key == "full" else 0.09)
            ax.plot(time_s, seeds.mean(axis=0), color=COLORS[key],
                    linestyle=LINESTYLES[key], linewidth=LINEWIDTHS[key],
                    zorder=4 if key == "full" else 3)
        ax.axvline(64, color=COLORS["lock_edge"], linewidth=0.8, linestyle="--")
        combined = np.concatenate(zoom_values)
        pad = max(0.30, 0.07 * float(combined.max()-combined.min()))
        zoom_lo, zoom_hi = float(combined.min()-pad), float(combined.max()+pad)
        ax.add_patch(Rectangle((z0, zoom_lo), z1-z0, zoom_hi-zoom_lo, fill=False,
                               edgecolor=COLORS["end"], linewidth=1.0,
                               linestyle=(0, (3, 2)), zorder=6))
        ax.set_xlim(1, 256)
        all_altitude = np.concatenate(full_values)
        altitude_span = max(1.0, float(all_altitude.max() - all_altitude.min()))
        ax.set_ylim(float(all_altitude.min() - 0.04 * altitude_span),
                    float(all_altitude.max() + 0.24 * altitude_span))
        ax.set_xlabel("Forecast time (s)"); ax.set_ylabel("Altitude (km)")
        style_axes(ax)
        # The inset occupies the upper-right quadrant; suppressing the parent
        # grid prevents hidden parent-grid strokes from passing beneath inset
        # tick labels in the exported vector PDF.
        ax.grid(False)
        panel_label(ax, col, maneuver.title())

        zoom_ax = inset_axes(ax, width="43%", height="43%", loc="upper right",
                             borderpad=0.9)
        zoom_time = time_s[slice_obj]
        zoom_ax.plot(zoom_time, truth[slice_obj], color=COLORS["truth"],
                     linewidth=1.45, zorder=5)
        for key in METHOD_ORDER:
            seeds = np.asarray(arrays[f"{maneuver}_{key}_alt_km"], dtype=float)[:, slice_obj]
            for seed in seeds:
                zoom_ax.plot(zoom_time, seed, color=COLORS[key],
                             linewidth=0.5, alpha=0.16 if key == "full" else 0.08)
            zoom_ax.plot(zoom_time, seeds.mean(axis=0), color=COLORS[key],
                         linestyle=LINESTYLES[key],
                         linewidth=1.55 if key == "full" else 1.0,
                         zorder=4 if key == "full" else 3)
        zoom_ax.set_xlim(float(zoom_time[0]), float(zoom_time[-1]))
        zoom_ax.set_ylim(zoom_lo, zoom_hi)
        # The zoom interval is stated in the figure caption; omitting repeated
        # interior time labels keeps the compact inset legible at journal size.
        zoom_ax.set_xticks([])
        zoom_ax.tick_params(direction="out", length=0, width=0.55,
                            labelsize=6.2, pad=4.0, colors=COLORS["text"])
        zoom_ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45,
                     alpha=0.65, zorder=0)
        for spine in zoom_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS["axis"])
            spine.set_linewidth(0.6)

    inside_legend(axes[0], method_handles(include_truth=True),
                  loc="upper left", ncol=1, fontsize=6.0, handlelength=1.2)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.27, top=0.95,
                        wspace=0.27)
    return {"outputs": save_bundle(fig, output_dir, "fig_altitude_skip"),
            "array_count": len(arrays), "source_sha256": sha256_file(source),
            "selection_sha256": sha256_file(selection_path)}


def render_ablation(source: Path, output_dir: Path) -> dict:
    rows = read_csv(source)
    configurations = (
        "Learned-only backbone",
        "Spherical prior + adaptive fusion",
        "Rotating-Earth prior + fixed schedule",
        "PLGAFormer (full)",
    )
    display = (
        "Learned-only\nbackbone", "Spherical prior\n+ adaptive fusion",
        "Rotating-Earth prior\n+ fixed schedule", "PLGAFormer\n(full)",
    )
    metrics = ("ade", "fde")
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        configuration = row["configuration"]
        aliases = {
            "PLGAFormer learned-only backbone": "Learned-only backbone",
            "PLGAFormer": "PLGAFormer (full)",
        }
        configuration = aliases.get(configuration, configuration)
        if "ADE_mean_km" in row:
            for metric in metrics:
                prefix = metric.upper()
                lookup[(configuration, metric)] = (
                    float(row[f"{prefix}_mean_km"]),
                    float(row[f"{prefix}_sample_SD_km"]),
                )
        elif "metric" in row:
            metric = row["metric"].lower()
            if metric in metrics and configuration in configurations:
                lookup[(configuration, metric)] = (
                    float(row["mean_m"])/1000.0,
                    float(row["sample_std_m"])/1000.0,
                )
    expected = {(c, m) for c in configurations for m in metrics}
    if set(lookup) != expected:
        raise ValueError(f"Ablation matrix mismatch: missing={sorted(expected-set(lookup))}")
    palette = (
        OKABE_ITO["blue"], OKABE_ITO["orange"],
        OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.70), sharey=True)
    y = np.arange(len(configurations))
    for p, (ax, metric) in enumerate(zip(axes, metrics, strict=True)):
        means = finite_nonnegative((lookup[(c, metric)][0] for c in configurations), metric)
        ax.barh(
            y, means, height=0.56, color=palette,
            edgecolor="none", zorder=3,
        )
        ax.set_yticks(y)
        if p == 0:
            ax.set_yticklabels(display)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(0, float(np.max(means))*1.08)
        ax.set_xlabel(f"{metric.upper()} (km)"); style_axes(ax, grid="x")
        panel_label(ax, p, metric.upper())
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.22, right=0.995, bottom=0.25, top=0.94, wspace=0.18)
    return {"outputs": save_bundle(fig, output_dir, "fig_ablation_summary"),
            "rows": len(rows), "source_sha256": sha256_file(source)}


def render_robustness(source: Path, output_dir: Path) -> dict:
    rows = read_csv(source)
    conditions = ("nominal", "aero_shift", "ballistic_shift")
    labels = ("Nominal", "CL -10%, CD +10%", "m +15%, S -10%")
    models = ("transformer", "plgaformer")
    lookup = {(row["condition"], row["model"]): row for row in rows}
    expected = {(c, m) for c in conditions for m in models}
    if not expected.issubset(lookup):
        raise ValueError(f"Robustness matrix mismatch: missing={sorted(expected-set(lookup))}")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.50))
    x = np.arange(len(conditions), dtype=float)
    width = 0.30
    for p, (ax, metric) in enumerate(zip(axes, ("ade", "fde"), strict=True)):
        ymax = 0.0
        for model_index, model in enumerate(models):
            means = finite_nonnegative(
                (float(lookup[(c, model)][f"{metric}_mean_m"])/1000.0 for c in conditions),
                f"robustness {model} {metric}")
            ymax = max(ymax, float(np.max(means)))
            color = COLORS["baseline"] if model == "transformer" else COLORS["full"]
            offset = (-0.5 if model_index == 0 else 0.5) * width
            ax.bar(
                x+offset, means, width=width,
                color=color, edgecolor="none", zorder=3,
            )
        ax.set_xticks(x, ("Nominal", "CL -10%\nCD +10%", "m +15%\nS -10%"))
        ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, ymax*1.25)
        ax.set_ylabel(f"{metric.upper()} (km)")
        style_axes(ax, grid="y"); panel_label(ax, p, metric.upper(), y_pt=-28.0)
        inside_legend(ax, handles=[
            Patch(facecolor=COLORS["baseline"], edgecolor="none", label="Transformer"),
            Patch(facecolor=COLORS["full"], edgecolor="none", label="PLGAFormer"),
        ], loc="upper left", ncol=2)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.31, top=0.94, wspace=0.27)
    return {"outputs": save_bundle(fig, output_dir, "fig_robustness"),
            "rows": len(rows), "source_sha256": sha256_file(source)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overall-csv", type=Path, required=True)
    parser.add_argument("--maneuver-csv", type=Path, required=True)
    parser.add_argument("--scenario-csv", type=Path, required=True)
    parser.add_argument("--typical-npz", type=Path, required=True)
    parser.add_argument("--altitude-npz", type=Path, required=True)
    parser.add_argument("--altitude-selection-json", type=Path, required=True)
    parser.add_argument("--ablation-csv", type=Path, required=True)
    parser.add_argument("--robustness-csv", type=Path,
                        default=PROJECT_ROOT/"PublicRelease"/"evidence"/"robustness.csv")
    parser.add_argument("--qa-scripts-dir", type=Path,
                        help="Optional directory containing audit_panel_alignment.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); configure_style(); load_alignment_auditor(args.qa_scripts_dir)
    output_dir = args.output_dir.resolve()
    sources = {
        "overall": args.overall_csv.resolve(),
        "maneuver": args.maneuver_csv.resolve(),
        "scenario": args.scenario_csv.resolve(),
        "typical": args.typical_npz.resolve(),
        "altitude": args.altitude_npz.resolve(),
        "altitude_selection": args.altitude_selection_json.resolve(),
        "ablation": args.ablation_csv.resolve(),
        "robustness": args.robustness_csv.resolve(),
    }
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} source: {path}")
    results = {
        "fig_overall_performance": render_overall(sources["overall"], output_dir),
        "fig_maneuver_performance": render_maneuver(sources["maneuver"], output_dir),
        "fig_scenario_evidence": render_scenario(sources["scenario"], output_dir),
        "fig_typical_maneuver_cases": render_typical(sources["typical"], output_dir),
        "fig_altitude_skip": render_altitude(
            sources["altitude"], sources["altitude_selection"], output_dir),
        "fig_ablation_summary": render_ablation(sources["ablation"], output_dir),
        "fig_robustness": render_robustness(sources["robustness"], output_dir),
    }
    manifest = {
        "schema_version": 1, "backend": "python_matplotlib",
        "visual_only": True, "numerical_values_changed": False,
        "palette": COLORS,
        "sources": {name: {"file": path.name, "sha256": sha256_file(path)}
                    for name, path in sources.items()},
        "figures": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"information_fusion_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir),
                      "figures": list(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
