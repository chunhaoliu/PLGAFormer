#!/usr/bin/env python3
"""Render six representative motion-regime panels from the frozen HGV dataset.

The figure shows one deterministic medoid from each of six joint motion strata.
Selection uses trajectory descriptors only and never prediction error. Every
selected trajectory contributes all 1,000 samples, and the output QA manifest
records the frozen dataset hash, representative IDs, and panel provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


EARTH_RADIUS_M = 6_378_000.0
EXPECTED_PROTOCOL = "hgv_multiregime_state_v2_1"
EXPECTED_DATASET_SHA256 = "526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7"
STRATA = (
    ("quasi_equilibrium", "longitudinal"),
    ("quasi_equilibrium", "turning"),
    ("quasi_equilibrium", "weaving"),
    ("skip_glide", "longitudinal"),
    ("skip_glide", "turning"),
    ("skip_glide", "weaving"),
)
MANEUVER_COLORS = {
    "longitudinal": "#3B6AA0",
    "turning": "#C77C32",
    "weaving": "#39836F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def select_representatives(
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
        if ids.size != 300:
            raise ValueError(f"Expected 300 trajectories in {name}, found {ids.size}")
        values = descriptors[ids]
        median = np.median(values, axis=0)
        scale = np.subtract(*np.percentile(values, [75, 25], axis=0))
        scale[scale < 1e-9] = 1.0
        distance = np.sum(((values - median) / scale) ** 2, axis=1)
        selected[name] = int(ids[int(np.argmin(distance))])
    return selected


def relative_coordinates(trajectory: np.ndarray) -> tuple[np.ndarray, ...]:
    longitude = np.rad2deg(np.unwrap(trajectory[:, 1]))
    latitude = np.rad2deg(trajectory[:, 2])
    return (
        longitude - longitude[0],
        latitude - latitude[0],
        (trajectory[:, 0] - EARTH_RADIUS_M) / 1000.0,
        trajectory[:, 3] / 1000.0,
    )


def common_limits(panel_data: dict[str, tuple[np.ndarray, ...]]) -> dict[str, tuple[float, float]]:
    delta_longitude = np.concatenate([values[0] for values in panel_data.values()])
    delta_latitude = np.concatenate([values[1] for values in panel_data.values()])
    longitude_limit = max(float(np.max(np.abs(delta_longitude))) * 1.04, 1.0)
    latitude_limit = max(float(np.max(np.abs(delta_latitude))) * 1.08, 1.0)
    return {
        "longitude": (-longitude_limit, longitude_limit),
        "latitude": (-latitude_limit, latitude_limit),
        "altitude": (30.0, 80.0),
    }


def save_panel(fig: plt.Figure, stem: Path) -> dict[str, str]:
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    tiff_path = stem.with_suffix(".tiff")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    return {
        "pdf": pdf_path.name,
        "svg": svg_path.name,
        "png": png_path.name,
        "tiff": tiff_path.name,
    }


def render_panel(
    output_dir: Path,
    vertical: str,
    maneuver: str,
    values: tuple[np.ndarray, ...],
    limits: dict[str, tuple[float, float]],
) -> dict[str, str]:
    delta_longitude, delta_latitude, altitude, _speed = values
    fig = plt.figure(figsize=(2.32, 1.92))
    axis = fig.add_subplot(111, projection="3d")
    color = MANEUVER_COLORS[maneuver]
    axis.plot(delta_longitude, delta_latitude, altitude, color=color, linewidth=1.45)
    axis.plot(
        delta_longitude,
        delta_latitude,
        np.full_like(altitude, 30.0),
        color="#9AA0A6",
        linewidth=0.65,
        alpha=0.8,
    )
    axis.scatter(
        delta_longitude[0], delta_latitude[0], altitude[0],
        s=16, color="#2E8B57", edgecolor="white", linewidth=0.35, zorder=5,
    )
    axis.scatter(
        delta_longitude[-1], delta_latitude[-1], altitude[-1],
        s=16, marker="s", color="#C84A42", edgecolor="white", linewidth=0.35, zorder=5,
    )
    axis.set_xlim(*limits["longitude"])
    axis.set_ylim(*limits["latitude"])
    axis.set_zlim(*limits["altitude"])
    axis.set_xlabel(r"$\Delta\lambda$ (deg)", labelpad=-4)
    axis.set_ylabel(r"$\Delta\phi$ (deg)", labelpad=-4)
    axis.set_zlabel("")
    axis.text2D(
        1.10,
        0.58,
        r"$h$ (km)",
        transform=axis.transAxes,
        rotation=90,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )
    axis.tick_params(pad=-1, length=2)
    axis.view_init(elev=25, azim=-63)
    axis.set_box_aspect((1.25, 1.0, 0.82))
    axis.grid(True, linewidth=0.35, alpha=0.5)
    axis.xaxis.pane.set_facecolor((1, 1, 1, 1))
    axis.yaxis.pane.set_facecolor((1, 1, 1, 1))
    axis.zaxis.pane.set_facecolor((1, 1, 1, 1))
    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.0, top=1.0)
    stem = output_dir / f"fig_scenario_evidence_{vertical}_{maneuver}"
    outputs = save_panel(fig, stem)
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = sha256(dataset)
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(f"Frozen dataset hash mismatch: {dataset_hash}")

    configure_matplotlib()
    with np.load(dataset, allow_pickle=False) as loaded:
        states = loaded["clean_trajectories"].astype(np.float64)
        joint = loaded["joint_strata"].astype(str)
        extrema = loaded["vertical_extrema_counts"].astype(np.float64)
        controls = loaded["control_parameters"].astype(np.float64)
        protocol = str(loaded["dataset_protocol"].item())
    if protocol != EXPECTED_PROTOCOL:
        raise ValueError(f"Protocol mismatch: {protocol}")
    if states.shape != (1800, 1000, 6):
        raise ValueError(f"Unexpected state tensor shape: {states.shape}")

    selected = select_representatives(states, joint, extrema, controls)
    panel_data = {
        name: relative_coordinates(states[trajectory_id])
        for name, trajectory_id in selected.items()
    }
    limits = common_limits(panel_data)
    outputs: dict[str, dict[str, str]] = {}
    source_path = output_dir / "fig_scenario_evidence_source.csv"
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "joint_stratum", "trajectory_id", "time_s",
                "delta_longitude_deg", "delta_latitude_deg", "altitude_km", "speed_km_s",
            ),
        )
        writer.writeheader()
        for vertical, maneuver in STRATA:
            name = f"{vertical}__{maneuver}"
            trajectory_id = selected[name]
            values = panel_data[name]
            outputs[name] = render_panel(output_dir, vertical, maneuver, values, limits)
            for time_s, sample in enumerate(zip(*values)):
                writer.writerow(
                    {
                        "joint_stratum": name,
                        "trajectory_id": trajectory_id,
                        "time_s": time_s,
                        "delta_longitude_deg": sample[0],
                        "delta_latitude_deg": sample[1],
                        "altitude_km": sample[2],
                        "speed_km_s": sample[3],
                    }
                )

    qa = {
        "figure": "fig_scenario_evidence",
        "dataset_file": dataset.name,
        "dataset_protocol": protocol,
        "dataset_sha256": dataset_hash,
        "state_shape": list(states.shape),
        "joint_stratum_counts": {
            f"{vertical}__{maneuver}": int(np.sum(joint == f"{vertical}__{maneuver}"))
            for vertical, maneuver in STRATA
        },
        "selection_rule": "minimum robust-scaled distance to the joint-stratum median descriptor; prediction error is not used",
        "representative_trajectory_ids": selected,
        "samples_per_panel": 1000,
        "excluded_samples": 0,
        "coordinate_transform": "longitude and latitude offsets from each trajectory start; absolute altitude",
        "common_axis_limits": limits,
        "panel_outputs": outputs,
        "source_data_file": source_path.name,
        "statistical_tests": "none",
    }
    qa_path = output_dir / "fig_scenario_evidence_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
