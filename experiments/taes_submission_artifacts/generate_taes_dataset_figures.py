"""Generate TAES dataset-characterization figures from the formal raw artifact.

The legacy AST manuscript contains useful figure concepts, but its exported
figures predate the current 1 Hz, 1000-point, trajectory-level protocol.  This
script recreates only the reusable scientific views from the canonical raw
trajectory artifact so the TAES manuscript does not mix protocols.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "raw_hgv_trajectories.npz"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated"
EARTH_RADIUS_M = 6_378_137.0
MANEUVERS = ("longitudinal", "turning", "weaving")
COLORS = {
    "longitudinal": "#0072B2",
    "turning": "#D55E00",
    "weaving": "#009E73",
}
LINESTYLES = {
    "longitudinal": "-",
    "turning": "--",
    "weaving": "-.",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate protocol-matched TAES dataset figures."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def load_formal_raw(path: Path) -> dict[str, np.ndarray | float | int | str]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    protocol = str(np.asarray(data["raw_protocol"]).item())
    sampling_interval = float(np.asarray(data["sampling_interval_s"]).item())
    points = int(np.asarray(data["points_per_trajectory"]).item())
    trajectories = np.asarray(data["trajectories"])
    labels = np.asarray(data["trajectory_labels"]).astype(str)

    if protocol != "raw_trajectory_v1":
        raise ValueError(f"Unexpected raw protocol: {protocol}")
    if sampling_interval != 1.0 or points != 1000:
        raise ValueError(
            "TAES figures require the formal 1 Hz, 1000-point raw protocol; "
            f"received {sampling_interval:g} Hz interval metadata and {points} points."
        )
    if trajectories.shape != (1000, 1000, 6):
        raise ValueError(f"Unexpected trajectory array shape: {trajectories.shape}")

    counts = {label: int(np.sum(labels == label)) for label in MANEUVERS}
    if counts != {"longitudinal": 333, "turning": 333, "weaving": 334}:
        raise ValueError(f"Unexpected maneuver counts: {counts}")
    return data


def relative_geodetic_track(
    trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unwrapped longitude and latitude changes in degrees.

    The complete trajectories span thousands of kilometres, so a small-area
    east--north tangent-plane approximation would distort the displayed
    geometry. Relative geodetic angles retain the spherical coordinates used
    by the simulator while removing arbitrary launch-location offsets.
    """

    longitude = np.unwrap(trajectory[:, 1].astype(np.float64))
    latitude = trajectory[:, 2].astype(np.float64)
    delta_longitude = np.rad2deg(longitude - longitude[0])
    delta_latitude = np.rad2deg(latitude - latitude[0])
    return delta_longitude, delta_latitude


def representative_indices(
    trajectories: np.ndarray, labels: np.ndarray
) -> dict[str, int]:
    """Select a deterministic medoid-like full trajectory per maneuver."""

    representatives: dict[str, int] = {}
    for label in MANEUVERS:
        indices = np.flatnonzero(labels == label)
        features = []
        for index in indices:
            trajectory = trajectories[index]
            delta_longitude, delta_latitude = relative_geodetic_track(
                trajectory
            )
            altitude_km = (trajectory[:, 0] - EARTH_RADIUS_M) / 1000.0
            features.append(
                [
                    altitude_km[0],
                    trajectory[0, 3] / 1000.0,
                    np.rad2deg(trajectory[0, 4]),
                    np.rad2deg(trajectory[0, 5]),
                    altitude_km[-1],
                    trajectory[-1, 3] / 1000.0,
                    delta_longitude[-1],
                    delta_latitude[-1],
                ]
            )
        matrix = np.asarray(features, dtype=np.float64)
        center = np.median(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-12] = 1.0
        distance = np.sum(((matrix - center) / scale) ** 2, axis=1)
        representatives[label] = int(indices[int(np.argmin(distance))])
    return representatives


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_maneuver_tracks(
    trajectories: np.ndarray,
    representatives: dict[str, int],
    output_dir: Path,
) -> None:
    """Plot one velocity-coded 3-D complete trajectory per maneuver."""

    speed_km_s = np.concatenate(
        [trajectories[index, :, 3] / 1000.0 for index in representatives.values()]
    )
    speed_norm = Normalize(
        vmin=float(np.min(speed_km_s)),
        vmax=float(np.max(speed_km_s)),
    )
    altitude_km = (trajectories[:, :, 0] - EARTH_RADIUS_M) / 1000.0
    altitude_limits = expanded_limits(
        np.concatenate(
            [altitude_km[index] for index in representatives.values()]
        ),
        minimum_span=10.0,
        padding_fraction=0.04,
    )

    for label in MANEUVERS:
        trajectory = trajectories[representatives[label]]
        delta_longitude, delta_latitude = relative_geodetic_track(trajectory)
        height_km = (trajectory[:, 0] - EARTH_RADIUS_M) / 1000.0
        speed = trajectory[:, 3] / 1000.0

        points = np.column_stack(
            [delta_longitude, delta_latitude, height_km]
        ).reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        collection = Line3DCollection(
            segments,
            cmap="viridis",
            norm=speed_norm,
            linewidth=2.25,
        )
        collection.set_array(0.5 * (speed[:-1] + speed[1:]))

        fig = plt.figure(figsize=(2.38, 2.22))
        axis = fig.add_subplot(111, projection="3d")
        axis.add_collection3d(collection)
        axis.plot(
            delta_longitude,
            delta_latitude,
            np.full_like(height_km, altitude_limits[0]),
            color="#6B7280",
            linewidth=0.9,
            alpha=0.50,
        )
        axis.scatter(
            delta_longitude[0],
            delta_latitude[0],
            height_km[0],
            s=24,
            marker="o",
            color="#2A9D63",
            edgecolor="#173B2D",
            linewidth=0.7,
            depthshade=False,
        )
        axis.scatter(
            delta_longitude[-1],
            delta_latitude[-1],
            height_km[-1],
            s=27,
            marker="s",
            color="#D95F59",
            edgecolor="#5B2420",
            linewidth=0.7,
            depthshade=False,
        )
        axis.set_xlim(expanded_limits(delta_longitude, minimum_span=0.12))
        axis.set_ylim(expanded_limits(delta_latitude, minimum_span=1.0))
        axis.set_zlim(altitude_limits)
        axis.set_xlabel(r"$\Delta\lambda$ (deg)", labelpad=-1)
        axis.set_ylabel(r"$\Delta\phi$ (deg)", labelpad=-1)
        axis.set_zlabel("")
        axis.view_init(elev=24.0, azim=-58.0)
        axis.set_box_aspect((1.05, 1.10, 0.82))
        axis.grid(True, color="#D8DDE3", linewidth=0.5, alpha=0.65)
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            pane.set_edgecolor("#C8CDD3")
        axis.tick_params(pad=-1, labelsize=6.2)
        if label == "weaving":
            fig.text(
                0.985,
                0.54,
                "Altitude (km)",
                rotation=90,
                va="center",
                ha="right",
                fontsize=7.5,
            )
        fig.subplots_adjust(left=0.00, right=0.91, bottom=0.02, top=0.99)
        save_figure(fig, output_dir / f"fig_maneuver_{label}")

    plot_trajectory_key(speed_norm, output_dir)


def plot_trajectory_key(speed_norm: Normalize, output_dir: Path) -> None:
    """Create the shared projection/endpoint legend and speed color scale."""

    fig = plt.figure(figsize=(7.05, 0.48))
    legend_axis = fig.add_axes([0.01, 0.03, 0.50, 0.94])
    legend_axis.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color="#6B7280",
            linewidth=1.1,
            alpha=0.65,
            label="Horizontal projection",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.2,
            markerfacecolor="#2A9D63",
            markeredgecolor="#173B2D",
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=5.2,
            markerfacecolor="#D95F59",
            markeredgecolor="#5B2420",
            label="End",
        ),
    ]
    legend_axis.legend(
        handles=handles,
        loc="center",
        ncol=3,
        handlelength=1.8,
        columnspacing=1.5,
    )
    colorbar_axis = fig.add_axes([0.61, 0.42, 0.34, 0.23])
    colorbar = fig.colorbar(
        ScalarMappable(norm=speed_norm, cmap="viridis"),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Earth-relative speed (km/s)", fontsize=7.0, labelpad=1)
    colorbar.ax.tick_params(labelsize=6.4, length=2.2, pad=1)
    save_figure(fig, output_dir / "fig_maneuver_key")


def expanded_limits(
    values: np.ndarray,
    minimum_span: float,
    padding_fraction: float = 0.08,
) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = max(upper - lower, minimum_span)
    center = 0.5 * (lower + upper)
    padding = padding_fraction * span
    return (
        center - 0.5 * span - padding,
        center + 0.5 * span + padding,
    )


def plot_maneuver_responses(
    data: dict[str, np.ndarray | float | int | str],
    representatives: dict[str, int],
    output_dir: Path,
) -> None:
    time = np.asarray(data["time"], dtype=np.float64)
    bank = np.rad2deg(np.asarray(data["bank"], dtype=np.float64))
    speed = np.asarray(data["velocity"], dtype=np.float64) / 1000.0

    fig, axis = plt.subplots(figsize=(2.38, 1.68))
    axis.axhline(0.0, color="#9AA1A8", linewidth=0.65, zorder=0)
    for label in MANEUVERS:
        index = representatives[label]
        axis.plot(
            time[index],
            bank[index],
            color=COLORS[label],
            linewidth=1.35,
            linestyle=LINESTYLES[label],
        )
    axis.set_xlim(0.0, 999.0)
    axis.set_ylim(-32.0, 32.0)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(r"Bank angle $\sigma$ (deg)")
    axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.72)
    axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.97)
    save_figure(fig, output_dir / "fig_bank_commands")

    fig, axis = plt.subplots(figsize=(2.38, 1.68))
    representative_speeds = []
    for label in MANEUVERS:
        index = representatives[label]
        representative_speeds.append(speed[index])
        axis.plot(
            time[index],
            speed[index],
            color=COLORS[label],
            linewidth=1.35,
            linestyle=LINESTYLES[label],
        )
    speed_values = np.concatenate(representative_speeds)
    axis.set_xlim(0.0, 999.0)
    axis.set_ylim(expanded_limits(speed_values, minimum_span=2.4, padding_fraction=0.04))
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Speed (km/s)")
    axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.72)
    axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.97)
    save_figure(fig, output_dir / "fig_speed_responses")

    plot_maneuver_class_key(output_dir)


def plot_maneuver_class_key(output_dir: Path) -> None:
    """Create one shared maneuver legend for bank and speed panels."""

    fig, axis = plt.subplots(figsize=(7.05, 0.25))
    axis.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=1.5,
            label=label.capitalize(),
        )
        for label in MANEUVERS
    ]
    axis.legend(
        handles=handles,
        loc="center",
        ncol=3,
        handlelength=2.4,
        columnspacing=2.0,
    )
    save_figure(fig, output_dir / "fig_maneuver_class_key")


def plot_operational_envelope(
    trajectories: np.ndarray, output_dir: Path
) -> None:
    altitude_km = (trajectories[:, :, 0] - EARTH_RADIUS_M) / 1000.0
    speed_km_s = trajectories[:, :, 3] / 1000.0
    sampled_altitude = altitude_km[:, ::20].reshape(-1)
    sampled_speed = speed_km_s[:, ::20].reshape(-1)
    fig, axis = plt.subplots(figsize=(2.38, 1.68))
    density = axis.hexbin(
        sampled_speed,
        sampled_altitude,
        gridsize=(45, 30),
        bins="log",
        mincnt=1,
        cmap="Blues",
        linewidths=0.0,
        rasterized=True,
    )

    x_min, x_max = float(speed_km_s.min()), float(speed_km_s.max())
    y_min, y_max = float(altitude_km.min()), float(altitude_km.max())
    axis.set_xlim(x_min - 0.08, x_max + 0.08)
    axis.set_ylim(y_min - 0.8, y_max + 0.8)
    axis.set_xlabel("Earth-relative speed (km/s)")
    axis.set_ylabel("Altitude (km)")
    axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.60)
    axis.set_axisbelow(True)
    axis.text(
        0.03,
        0.94,
        r"$n=1{,}000$ complete trajectories",
        transform=axis.transAxes,
        va="top",
        fontsize=6.4,
        color="#2F3740",
    )
    colorbar = fig.colorbar(density, ax=axis, fraction=0.055, pad=0.025)
    colorbar.set_label("Point density (log scale)", fontsize=6.6, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.0, length=2.0, pad=1)
    fig.subplots_adjust(left=0.18, right=0.89, bottom=0.25, top=0.97)
    save_figure(fig, output_dir / "fig_operational_envelope")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_manifest(
    data_path: Path,
    data: dict[str, np.ndarray | float | int | str],
    representatives: dict[str, int],
    output_dir: Path,
) -> None:
    output_stems = [
        *(f"fig_maneuver_{label}" for label in MANEUVERS),
        "fig_maneuver_key",
        "fig_bank_commands",
        "fig_speed_responses",
        "fig_maneuver_class_key",
        "fig_operational_envelope",
    ]
    manifest = {
        "artifact_contract": "taes_dataset_figures_v3",
        "source_data": str(data_path),
        "source_sha256": sha256_file(data_path),
        "raw_protocol": str(np.asarray(data["raw_protocol"]).item()),
        "sampling_interval_s": float(
            np.asarray(data["sampling_interval_s"]).item()
        ),
        "points_per_trajectory": int(
            np.asarray(data["points_per_trajectory"]).item()
        ),
        "trajectory_duration_s": float(
            np.asarray(data["trajectory_duration_s"]).item()
        ),
        "display_coordinate_system": "relative_geodetic_angles_degrees",
        "operational_envelope_trajectory_count": int(
            np.asarray(data["trajectories"]).shape[0]
        ),
        "operational_envelope_time_stride": 20,
        "representatives": {
            label: {
                "array_index": int(index),
                "trajectory_id": int(
                    np.asarray(data["trajectory_ids"])[index]
                ),
            }
            for label, index in representatives.items()
        },
        "outputs": {
            stem: {
                extension: sha256_file(output_dir / f"{stem}.{extension}")
                for extension in ("svg", "pdf", "png")
            }
            for stem in output_stems
        },
    }
    manifest_path = output_dir / "dataset_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_formal_raw(args.data.resolve())
    trajectories = np.asarray(data["trajectories"], dtype=np.float64)
    labels = np.asarray(data["trajectory_labels"]).astype(str)
    representatives = representative_indices(trajectories, labels)

    configure_style()
    plot_maneuver_tracks(trajectories, representatives, args.output_dir)
    plot_maneuver_responses(data, representatives, args.output_dir)
    plot_operational_envelope(trajectories, args.output_dir)
    write_manifest(args.data.resolve(), data, representatives, args.output_dir)

    print(
        "Generated protocol-matched TAES dataset figures from "
        f"{args.data.resolve()}"
    )
    print(
        "Representative trajectory IDs: "
        + ", ".join(
            f"{label}={int(np.asarray(data['trajectory_ids'])[index])}"
            for label, index in representatives.items()
        )
    )


if __name__ == "__main__":
    main()
