#!/usr/bin/env python3
"""Generate publication figures directly from the multi-regime data artifact.

Figure contract
---------------
Conclusion:
    The formal dataset contains distinct quasi-equilibrium and skip-glide
    vertical dynamics across all three primary lateral maneuver families,
    while randomized controls remain inside the declared physical envelope.
Archetype:
    Quantitative grid, double-column.
Evidence:
    A six-panel representative-trajectory grid shows the factorial maneuver
    coverage; a four-panel characterization figure shows control diversity,
    operating-envelope coverage, and physical gates.
Review risks:
    Representatives are selected by a deterministic medoid rule, all 3-D
    panels share camera/altitude/speed scales, and global quantitative panels
    consume every complete trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_multiregime_dataset_v2_1.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "taes_submission_artifacts"
    / "generated"
    / "multiregime_dataset"
)
EARTH_RADIUS_M = 6_378_000.0
STRATA = (
    ("quasi_equilibrium", "longitudinal"),
    ("quasi_equilibrium", "turning"),
    ("quasi_equilibrium", "weaving"),
    ("skip_glide", "longitudinal"),
    ("skip_glide", "turning"),
    ("skip_glide", "weaving"),
)
MANEUVER_COLORS = {
    "longitudinal": "#315F9E",
    "turning": "#D47B2A",
    "weaving": "#2C8C7B",
}
VERTICAL_COLORS = {
    "quasi_equilibrium": "#315F9E",
    "skip_glide": "#C65D35",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _representatives(
    states: np.ndarray,
    joint: np.ndarray,
    extrema: np.ndarray,
    controls: np.ndarray,
) -> dict[str, int]:
    altitude = (states[:, :, 0] - EARTH_RADIUS_M) / 1000.0
    longitude = np.unwrap(states[:, :, 1], axis=1)
    latitude = states[:, :, 2]
    descriptors = np.column_stack(
        [
            altitude[:, 0],
            states[:, 0, 3] / 1000.0,
            np.rad2deg(states[:, 0, 4]),
            altitude[:, -1],
            states[:, -1, 3] / 1000.0,
            np.rad2deg(longitude[:, -1] - longitude[:, 0]),
            np.rad2deg(latitude[:, -1] - latitude[:, 0]),
            extrema,
            controls[:, 5],
            controls[:, 8],
        ]
    )
    selected: dict[str, int] = {}
    for vertical, maneuver in STRATA:
        name = f"{vertical}__{maneuver}"
        ids = np.flatnonzero(joint == name)
        values = descriptors[ids]
        median = np.median(values, axis=0)
        scale = np.subtract(*np.percentile(values, [75, 25], axis=0))
        scale[scale < 1e-9] = 1.0
        distance = np.sum(((values - median) / scale) ** 2, axis=1)
        selected[name] = int(ids[int(np.argmin(distance))])
    return selected


def _colored_3d_line(
    axis: plt.Axes,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    altitude_km: np.ndarray,
    speed_km_s: np.ndarray,
    norm: Normalize,
) -> None:
    points = np.column_stack([longitude_deg, latitude_deg, altitude_km])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = Line3DCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidth=1.35,
    )
    collection.set_array(speed_km_s[:-1])
    axis.add_collection3d(collection)
    floor = 30.0
    axis.plot(
        longitude_deg,
        latitude_deg,
        np.full_like(altitude_km, floor),
        color="#9AA0A6",
        linewidth=0.65,
        alpha=0.75,
    )
    axis.scatter(
        longitude_deg[0],
        latitude_deg[0],
        altitude_km[0],
        s=13,
        color="#2E8B57",
        edgecolor="white",
        linewidth=0.35,
        zorder=5,
    )
    axis.scatter(
        longitude_deg[-1],
        latitude_deg[-1],
        altitude_km[-1],
        s=13,
        marker="s",
        color="#C84A42",
        edgecolor="white",
        linewidth=0.35,
        zorder=5,
    )


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    outputs: list[str] = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def _trajectory_grid(
    output_dir: Path,
    states: np.ndarray,
    selected: dict[str, int],
) -> list[str]:
    width_in = 183.0 / 25.4
    fig = plt.figure(figsize=(width_in, 4.25))
    norm = Normalize(vmin=3.0, vmax=7.2)
    for index, (vertical, maneuver) in enumerate(STRATA):
        axis = fig.add_subplot(2, 3, index + 1, projection="3d")
        trajectory = states[selected[f"{vertical}__{maneuver}"]]
        longitude = np.rad2deg(np.unwrap(trajectory[:, 1]))
        latitude = np.rad2deg(trajectory[:, 2])
        altitude = (trajectory[:, 0] - EARTH_RADIUS_M) / 1000.0
        speed = trajectory[:, 3] / 1000.0
        _colored_3d_line(axis, longitude, latitude, altitude, speed, norm)
        axis.set_xlim(float(longitude.min()), float(longitude.max()))
        latitude_pad = max(float(np.ptp(latitude)) * 0.05, 0.2)
        axis.set_ylim(
            float(latitude.min() - latitude_pad),
            float(latitude.max() + latitude_pad),
        )
        axis.set_zlim(30.0, 80.0)
        axis.set_xlabel(r"$\lambda$ (deg)", labelpad=-5)
        axis.set_ylabel(r"$\phi$ (deg)", labelpad=-5)
        axis.set_zlabel(r"$h$ (km)", labelpad=-5)
        axis.tick_params(pad=-1, length=2)
        axis.view_init(elev=25, azim=-63)
        axis.set_box_aspect((1.35, 1.0, 0.8))
        if index < 3:
            axis.set_title(
                maneuver.capitalize(),
                color=MANEUVER_COLORS[maneuver],
                fontweight="bold",
                pad=2,
            )
        row_label = "Quasi-equilibrium" if vertical == "quasi_equilibrium" else "Skip glide"
        axis.text2D(
            0.02,
            0.95,
            f"{chr(97 + index)}  {row_label}",
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
        axis.grid(True, linewidth=0.35, alpha=0.5)
        axis.xaxis.pane.set_facecolor((1, 1, 1, 1))
        axis.yaxis.pane.set_facecolor((1, 1, 1, 1))
        axis.zaxis.pane.set_facecolor((1, 1, 1, 1))

    color_axis = fig.add_axes([0.24, 0.035, 0.52, 0.018])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Earth-relative speed (km/s)", labelpad=1)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2E8B57", markeredgecolor="white", label="Start"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#C84A42", markeredgecolor="white", label="End"),
        Line2D([0], [0], color="#9AA0A6", linewidth=0.8, label="Ground projection"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.96, bottom=0.17, wspace=0.02, hspace=0.02)
    return _save_figure(fig, output_dir / "fig_multiregime_trajectory_grid")


def _characterization(
    output_dir: Path,
    states: np.ndarray,
    labels: np.ndarray,
    vertical: np.ndarray,
    alpha: np.ndarray,
    bank_command: np.ndarray,
    bank_achieved: np.ndarray,
    pressure: np.ndarray,
    load_factor: np.ndarray,
    selected: dict[str, int],
) -> tuple[list[str], list[dict[str, float | str]]]:
    width_in = 183.0 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width_in, 4.45))
    time_s = np.arange(states.shape[1])

    axis = axes[0, 0]
    source_rows: list[dict[str, float | str]] = []
    for regime in ("quasi_equilibrium", "skip_glide"):
        values = np.rad2deg(alpha[vertical == regime])
        median = np.median(values, axis=0)
        low, high = np.percentile(values, [10, 90], axis=0)
        color = VERTICAL_COLORS[regime]
        label = "Quasi-equilibrium" if regime == "quasi_equilibrium" else "Skip glide"
        axis.fill_between(time_s, low, high, color=color, alpha=0.14, linewidth=0)
        axis.plot(time_s, median, color=color, linewidth=1.25, label=label)
        for index in range(0, len(time_s), 10):
            source_rows.append(
                {
                    "panel": "attack_angle",
                    "group": regime,
                    "time_s": float(time_s[index]),
                    "median": float(median[index]),
                    "p10": float(low[index]),
                    "p90": float(high[index]),
                    "achieved_deg": "",
                    "command_deg": "",
                    "trajectory_id": "",
                }
            )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Attack angle (deg)")
    axis.set_xlim(0, 999)
    axis.legend(loc="best")
    axis.text(-0.11, 1.04, "a", transform=axis.transAxes, fontweight="bold", fontsize=8)
    axis.set_title("Randomized longitudinal controls")

    axis = axes[0, 1]
    for maneuver in ("longitudinal", "turning", "weaving"):
        selected_id = selected[f"skip_glide__{maneuver}"]
        axis.plot(
            time_s,
            np.rad2deg(bank_command[selected_id]),
            color=MANEUVER_COLORS[maneuver],
            linewidth=0.75,
            linestyle="--",
            alpha=0.45,
        )
        axis.plot(
            time_s,
            np.rad2deg(bank_achieved[selected_id]),
            color=MANEUVER_COLORS[maneuver],
            linewidth=1.25,
            label=maneuver.capitalize(),
        )
        for index in range(0, len(time_s), 10):
            source_rows.append(
                {
                    "panel": "bank_angle",
                    "group": maneuver,
                    "time_s": float(time_s[index]),
                    "median": "",
                    "p10": "",
                    "p90": "",
                    "achieved_deg": float(
                        np.rad2deg(bank_achieved[selected_id, index])
                    ),
                    "command_deg": float(
                        np.rad2deg(bank_command[selected_id, index])
                    ),
                    "trajectory_id": int(selected_id),
                }
            )
    axis.axhline(0.0, color="#A0A5AA", linewidth=0.55)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Bank angle (deg)")
    axis.set_xlim(0, 999)
    axis.set_ylim(-38, 38)
    axis.legend(loc="best")
    axis.text(-0.11, 1.04, "b", transform=axis.transAxes, fontweight="bold", fontsize=8)
    axis.set_title("Command and rate-limited bank response")
    style_legend = [
        Line2D([0], [0], color="#5F6670", linewidth=1.25, label="Achieved"),
        Line2D(
            [0],
            [0],
            color="#5F6670",
            linewidth=0.75,
            linestyle="--",
            alpha=0.55,
            label="Ideal command",
        ),
    ]
    first_legend = axis.legend(loc="upper right")
    axis.add_artist(first_legend)
    axis.legend(handles=style_legend, loc="lower right")

    axis = axes[1, 0]
    altitude = (states[:, :, 0] - EARTH_RADIUS_M) / 1000.0
    speed = states[:, :, 3] / 1000.0
    sample = np.s_[::5]
    hist = axis.hexbin(
        speed[:, sample].ravel(),
        altitude[:, sample].ravel(),
        gridsize=(52, 40),
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0,
    )
    colorbar = fig.colorbar(hist, ax=axis, pad=0.015)
    colorbar.set_label("log sample count")
    axis.set_xlabel("Earth-relative speed (km/s)")
    axis.set_ylabel("Altitude (km)")
    axis.set_xlim(3.0, 7.2)
    axis.set_ylim(30.0, 80.0)
    axis.text(-0.11, 1.04, "c", transform=axis.transAxes, fontweight="bold", fontsize=8)
    axis.set_title("Complete-dataset operating envelope")

    axis = axes[1, 1]
    positions = np.arange(len(STRATA))
    q_values = [np.max(pressure[(vertical == v) & (labels == m)], axis=1) / 1000.0 for v, m in STRATA]
    n_values = [np.max(load_factor[(vertical == v) & (labels == m)], axis=1) for v, m in STRATA]
    q_box = axis.boxplot(
        q_values,
        positions=positions - 0.16,
        widths=0.27,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#315F9E", "linewidth": 1.0},
        boxprops={"facecolor": "#B8CCE6", "edgecolor": "#315F9E", "linewidth": 0.7},
        whiskerprops={"color": "#315F9E", "linewidth": 0.7},
        capprops={"color": "#315F9E", "linewidth": 0.7},
    )
    del q_box
    twin = axis.twinx()
    twin.spines["top"].set_visible(False)
    twin.boxplot(
        n_values,
        positions=positions + 0.16,
        widths=0.27,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#C65D35", "linewidth": 1.0},
        boxprops={"facecolor": "#EDC1AF", "edgecolor": "#C65D35", "linewidth": 0.7},
        whiskerprops={"color": "#C65D35", "linewidth": 0.7},
        capprops={"color": "#C65D35", "linewidth": 0.7},
    )
    axis.axhline(100.0, color="#315F9E", linestyle="--", linewidth=0.8)
    twin.axhline(5.0, color="#C65D35", linestyle=":", linewidth=0.8)
    axis.set_ylabel("Maximum dynamic pressure (kPa)", color="#315F9E")
    twin.set_ylabel("Maximum load factor (g)", color="#C65D35")
    axis.tick_params(axis="y", colors="#315F9E")
    twin.tick_params(axis="y", colors="#C65D35")
    axis.set_xticks(
        positions,
        ["Q-L", "Q-T", "Q-W", "S-L", "S-T", "S-W"],
    )
    axis.text(-0.11, 1.04, "d", transform=axis.transAxes, fontweight="bold", fontsize=8)
    axis.set_title("Trajectory-level physical gates")
    legend_handles = [
        mpl.patches.Patch(facecolor="#B8CCE6", edgecolor="#315F9E", label="Dynamic pressure"),
        mpl.patches.Patch(facecolor="#EDC1AF", edgecolor="#C65D35", label="Load factor"),
    ]
    axis.legend(handles=legend_handles, loc="upper left")

    for axis_i in axes.ravel():
        axis_i.grid(True, color="#D7DADF", linewidth=0.45, alpha=0.65)
    fig.subplots_adjust(left=0.085, right=0.915, bottom=0.12, top=0.94, wspace=0.32, hspace=0.36)
    outputs = _save_figure(fig, output_dir / "fig_multiregime_characterization")
    return outputs, source_rows


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    with np.load(dataset, allow_pickle=False) as loaded:
        states = loaded["clean_trajectories"].astype(np.float64)
        labels = loaded["trajectory_labels"].astype(str)
        vertical = loaded["vertical_regimes"].astype(str)
        joint = loaded["joint_strata"].astype(str)
        alpha = loaded["alpha_commands"].astype(np.float64)
        bank_command = loaded["bank_commands"].astype(np.float64)
        bank_achieved = loaded["bank_achieved"].astype(np.float64)
        pressure = loaded["dynamic_pressure_pa"].astype(np.float64)
        load_factor = loaded["load_factor_g"].astype(np.float64)
        extrema = loaded["vertical_extrema_counts"].astype(np.float64)
        controls = loaded["control_parameters"].astype(np.float64)
        protocol = str(loaded["dataset_protocol"].item())

    selected = _representatives(states, joint, extrema, controls)
    trajectory_outputs = _trajectory_grid(output_dir, states, selected)
    characterization_outputs, source_rows = _characterization(
        output_dir,
        states,
        labels,
        vertical,
        alpha,
        bank_command,
        bank_achieved,
        pressure,
        load_factor,
        selected,
    )
    source_path = output_dir / "fig_multiregime_characterization_source.csv"
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "panel",
                "group",
                "time_s",
                "median",
                "p10",
                "p90",
                "achieved_deg",
                "command_deg",
                "trajectory_id",
            ),
        )
        writer.writeheader()
        writer.writerows(source_rows)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "dataset_protocol": protocol,
        "representative_rule": "minimum robust-scaled distance to the joint-stratum median descriptor",
        "representative_trajectory_ids": selected,
        "trajectory_grid_outputs": trajectory_outputs,
        "characterization_outputs": characterization_outputs,
        "source_data": str(source_path),
        "figure_contract": {
            "archetype": "quantitative grid",
            "width_mm": 183,
            "formats": ["PDF", "SVG", "PNG 300 dpi", "TIFF 600 dpi"],
            "editable_vector_text": True,
        },
    }
    manifest_path = output_dir / "multiregime_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
