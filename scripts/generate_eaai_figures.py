"""Generate EAAI manuscript figures from the formal PLGAFormer artifacts.

The script intentionally reads the preserved long trajectories and signed
experiment summaries. It does not reconstruct paper numbers from prose.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
OUTPUT = PROJECT / "Init_Submit_EAAI" / "EAAI_Manuscript" / "generated"
RAW_DATA = REPO / "data_generation" / "data" / "processed" / "raw_hgv_trajectories.npz"
MAIN_RESULTS = (
    REPO
    / "experiments"
    / "taes_submission_artifacts"
    / "generated"
    / "main_results_summary.json"
)
ROBUSTNESS_RESULTS = (
    REPO / "experiments" / "exp3_robustness" / "results" / "robustness_results.json"
)
EFFICIENCY_RESULTS = (
    REPO / "experiments" / "exp6_efficiency" / "results" / "efficiency_results.json"
)

COLORS = {
    "PLGAFormer": "#1768AC",
    "AF-CILN": "#D97706",
    "Transformer": "#6B7280",
    "3-DOF": "#23856D",
    "longitudinal": "#1768AC",
    "turning": "#D1495B",
    "weaving": "#23856D",
}
MODEL_KEYS = {
    "PLGAFormer": "PLGAFormer (proposed)",
    "AF-CILN": "AF-CILN",
    "Transformer": "Transformer (baseline)",
    "3-DOF": "Rotating-Earth 3-DOF",
}
MANEUVERS = ("longitudinal", "turning", "weaving")
MANEUVER_LABELS = {
    "longitudinal": "Longitudinal",
    "turning": "Turning",
    "weaving": "Weaving",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.025)
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finish_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#D5D9DE", linewidth=0.45, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def dataset_figures() -> tuple[list[str], dict]:
    raw = np.load(RAW_DATA, allow_pickle=False)
    states = raw["trajectories"]
    times = raw["time"]
    labels = raw["trajectory_labels"]
    radius_earth = 6_378_000.0
    generated: list[str] = []

    # Complete local ground tracks, with every trajectory translated to its own
    # initial location so maneuver geometry is directly comparable.
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    for maneuver in MANEUVERS:
        indices = np.flatnonzero(labels == maneuver)
        chosen = indices[np.linspace(0, len(indices) - 1, 12, dtype=int)]
        for j, idx in enumerate(chosen):
            lon = states[idx, :, 1].astype(float)
            lat = states[idx, :, 2].astype(float)
            east = radius_earth * np.cos(lat[0]) * (lon - lon[0]) / 1000.0
            north = radius_earth * (lat - lat[0]) / 1000.0
            ax.plot(
                east,
                north,
                color=COLORS[maneuver],
                alpha=0.42,
                linewidth=0.85,
                label=MANEUVER_LABELS[maneuver] if j == 0 else None,
            )
    ax.scatter([0], [0], s=14, marker="o", facecolor="white", edgecolor="#111827", zorder=5)
    ax.set_xlabel("Relative east displacement (km)")
    ax.set_ylabel("Relative north displacement (km)")
    ax.set_aspect("equal", adjustable="datalim")
    finish_axes(ax)
    ax.legend(loc="best", ncol=1)
    generated += save_figure(fig, "fig_dataset_ground_tracks")

    # Time-resolved altitude envelope from all complete trajectories.
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    envelope_rows: list[dict] = []
    for maneuver in MANEUVERS:
        group = states[labels == maneuver, :, 0].astype(float) - radius_earth
        altitude = group / 1000.0
        q10, median, q90 = np.percentile(altitude, [10, 50, 90], axis=0)
        t = times[labels == maneuver][0]
        ax.fill_between(t, q10, q90, color=COLORS[maneuver], alpha=0.14, linewidth=0)
        ax.plot(t, median, color=COLORS[maneuver], label=MANEUVER_LABELS[maneuver])
        for step in range(len(t)):
            envelope_rows.append(
                {
                    "maneuver": maneuver,
                    "time_s": float(t[step]),
                    "altitude_q10_km": float(q10[step]),
                    "altitude_median_km": float(median[step]),
                    "altitude_q90_km": float(q90[step]),
                }
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    finish_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    generated += save_figure(fig, "fig_dataset_altitude_envelope")
    write_csv(
        "dataset_altitude_envelope.csv",
        ["maneuver", "time_s", "altitude_q10_km", "altitude_median_km", "altitude_q90_km"],
        envelope_rows,
    )

    # Operating envelope. A deterministic subsample avoids a million-point
    # vector graphic while still drawing from all trajectory families.
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    operating_rows: list[dict] = []
    for maneuver in MANEUVERS:
        group = states[labels == maneuver]
        flat_altitude = (group[:, :, 0].reshape(-1).astype(float) - radius_earth) / 1000.0
        flat_velocity = group[:, :, 3].reshape(-1).astype(float) / 1000.0
        take = np.linspace(0, len(flat_altitude) - 1, 3500, dtype=int)
        ax.scatter(
            flat_velocity[take],
            flat_altitude[take],
            s=2.0,
            alpha=0.16,
            linewidths=0,
            color=COLORS[maneuver],
            label=MANEUVER_LABELS[maneuver],
            rasterized=True,
        )
        for idx in take:
            operating_rows.append(
                {
                    "maneuver": maneuver,
                    "velocity_km_s": float(flat_velocity[idx]),
                    "altitude_km": float(flat_altitude[idx]),
                }
            )
    ax.set_xlabel(r"Earth-relative speed (km s$^{-1}$)")
    ax.set_ylabel("Altitude (km)")
    finish_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    generated += save_figure(fig, "fig_dataset_operating_envelope")
    write_csv(
        "dataset_operating_envelope.csv",
        ["maneuver", "velocity_km_s", "altitude_km"],
        operating_rows,
    )

    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    initial_rows: list[dict] = []
    initial = raw["initial_states"].astype(float)
    for maneuver in MANEUVERS:
        mask = labels == maneuver
        h0 = (initial[mask, 0] - radius_earth) / 1000.0
        v0 = initial[mask, 3] / 1000.0
        ax.scatter(
            v0,
            h0,
            s=8,
            alpha=0.55,
            linewidths=0,
            color=COLORS[maneuver],
            label=f"{MANEUVER_LABELS[maneuver]} ({mask.sum()})",
        )
        source_indices = np.flatnonzero(mask)
        for idx, altitude, velocity in zip(source_indices, h0, v0):
            initial_rows.append(
                {
                    "trajectory_id": int(raw["trajectory_ids"][idx]),
                    "maneuver": maneuver,
                    "initial_altitude_km": float(altitude),
                    "initial_velocity_km_s": float(velocity),
                }
            )
    ax.set_xlabel(r"Initial speed (km s$^{-1}$)")
    ax.set_ylabel("Initial altitude (km)")
    finish_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    generated += save_figure(fig, "fig_dataset_initial_conditions")
    write_csv(
        "dataset_initial_conditions.csv",
        ["trajectory_id", "maneuver", "initial_altitude_km", "initial_velocity_km_s"],
        initial_rows,
    )

    summary = {
        "trajectory_count": int(states.shape[0]),
        "points_per_trajectory": int(states.shape[1]),
        "state_channels": int(states.shape[2]),
        "sampling_interval_s": float(raw["sampling_interval_s"]),
        "trajectory_duration_s": float(raw["trajectory_duration_s"]),
        "generation_seed": int(raw["generation_seed"]),
        "maneuver_counts": {m: int(np.sum(labels == m)) for m in MANEUVERS},
    }
    return generated, summary


def performance_figures(main: dict) -> list[str]:
    generated: list[str] = []
    horizons = np.asarray(main["horizons"], dtype=float)
    rows: list[dict] = []
    for short_name, formal_name in MODEL_KEYS.items():
        for horizon in main["horizons"]:
            stats = main["seed_statistics"][formal_name][str(horizon)]
            rows.append(
                {
                    "model": short_name,
                    "horizon_s": horizon,
                    "ade_mean_km": stats["ade"]["mean_m"] / 1000.0,
                    "ade_sd_km": stats["ade"]["std_m"] / 1000.0,
                    "fde_mean_km": stats["fde"]["mean_m"] / 1000.0,
                    "fde_sd_km": stats["fde"]["std_m"] / 1000.0,
                }
            )
    write_csv(
        "multihorizon_performance.csv",
        ["model", "horizon_s", "ade_mean_km", "ade_sd_km", "fde_mean_km", "fde_sd_km"],
        rows,
    )

    for metric, ylabel in (("ade", "ADE (km)"), ("fde", "FDE (km)")):
        fig, ax = plt.subplots(figsize=(3.48, 2.65))
        for short_name, formal_name in MODEL_KEYS.items():
            means = np.asarray(
                [
                    main["seed_statistics"][formal_name][str(int(h))][metric]["mean_m"] / 1000.0
                    for h in horizons
                ]
            )
            stds = np.asarray(
                [
                    main["seed_statistics"][formal_name][str(int(h))][metric]["std_m"] / 1000.0
                    for h in horizons
                ]
            )
            ax.errorbar(
                horizons,
                means,
                yerr=stds,
                color=COLORS[short_name],
                marker="o",
                capsize=2.2,
                elinewidth=0.8,
                label=short_name,
            )
        ax.set_yscale("log")
        ax.set_xticks(horizons)
        ax.set_xlabel("Forecast horizon (s)")
        ax.set_ylabel(ylabel)
        finish_axes(ax)
        ax.legend(loc="best", ncol=2)
        generated += save_figure(fig, f"fig_performance_{metric}_horizon")

    maneuver_rows: list[dict] = []
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    x = np.arange(len(MANEUVERS))
    width = 0.19
    for offset, (short_name, formal_name) in enumerate(MODEL_KEYS.items()):
        means = []
        for maneuver in MANEUVERS:
            stats = main["maneuver_statistics"][formal_name][maneuver]["ade"]
            means.append(stats["mean_m"] / 1000.0)
            maneuver_rows.append(
                {
                    "model": short_name,
                    "maneuver": maneuver,
                    "ade_mean_km": stats["mean_m"] / 1000.0,
                    "ade_sd_km": stats["std_m"] / 1000.0,
                    "trajectory_count": stats["trajectory_count"],
                }
            )
        ax.bar(
            x + (offset - 1.5) * width,
            means,
            width,
            color=COLORS[short_name],
            label=short_name,
        )
    ax.set_xticks(x, [MANEUVER_LABELS[m] for m in MANEUVERS])
    ax.set_yscale("log")
    ax.set_ylabel("256 s ADE (km)")
    finish_axes(ax, grid_axis="y")
    ax.legend(loc="upper left", ncol=2)
    generated += save_figure(fig, "fig_performance_maneuver_ade")
    write_csv(
        "maneuver_performance.csv",
        ["model", "maneuver", "ade_mean_km", "ade_sd_km", "trajectory_count"],
        maneuver_rows,
    )
    return generated


def robustness_figures(robustness: dict) -> list[str]:
    generated: list[str] = []
    model_names = {
        "PLGAFormer": "PLGAFormer (proposed)",
        "Transformer": "Transformer (baseline)",
    }
    configs = [
        ("noise", robustness["config"]["noise_levels"], "Sensor-noise multiplier", "fig_robustness_noise"),
        ("missing", robustness["config"]["missing_rates"], "Missing-observation rate", "fig_robustness_missing"),
        ("input_length", robustness["config"]["input_lengths"], "Available history (s)", "fig_robustness_context"),
    ]
    rows: list[dict] = []
    for mode, levels, xlabel, stem in configs:
        fig, ax = plt.subplots(figsize=(3.48, 2.65))
        for short_name, formal_name in model_names.items():
            means = []
            stds = []
            for level in levels:
                entry = robustness["results"][formal_name][mode][str(level)]
                means.append(entry["ade_m"] / 1000.0)
                stds.append(entry["ade_m_std"] / 1000.0)
                rows.append(
                    {
                        "condition": mode,
                        "level": level,
                        "model": short_name,
                        "ade_mean_km": entry["ade_m"] / 1000.0,
                        "ade_sd_km": entry["ade_m_std"] / 1000.0,
                    }
                )
            ax.errorbar(
                levels,
                means,
                yerr=stds,
                color=COLORS[short_name],
                marker="o",
                capsize=2.2,
                elinewidth=0.8,
                label=short_name,
            )
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("256 s ADE (km)")
        if mode == "missing":
            ax.set_xticks(levels, [f"{100 * x:.0f}%" for x in levels])
        else:
            ax.set_xticks(levels)
        finish_axes(ax)
        ax.legend(loc="best")
        generated += save_figure(fig, stem)
    write_csv(
        "robustness_profiles.csv",
        ["condition", "level", "model", "ade_mean_km", "ade_sd_km"],
        rows,
    )
    return generated


def efficiency_figure(efficiency: dict) -> list[str]:
    rows = []
    fig, ax = plt.subplots(figsize=(3.48, 2.75))
    aliases = {
        "PLGAFormer": "PLGAFormer",
        "Transformer": "Transformer",
        "Transformer (baseline)": "Transformer",
        "Rotating-Earth 3-DOF": "3-DOF",
    }
    label_offsets = {
        "PLGAFormer": (4, 5),
        "Transformer": (4, -9),
        "AF-CILN": (4, 5),
        "3-DOF": (4, -9),
        "DLinear": (4, 5),
        "PIT": (4, 5),
        "PatchTST": (4, -9),
        "iTransformer": (4, 5),
        "Spherical kinematics": (4, 5),
    }
    for entry in efficiency["results"]:
        name = aliases.get(entry["model"], entry["model"])
        latency = float(entry["latency_batch1_ms"])
        ade = float(entry["ADE_256_m"]) / 1000.0
        params_m = float(entry["params"]) / 1e6
        color = COLORS.get(name, "#9CA3AF")
        ax.scatter(
            latency,
            ade,
            s=18 + 8 * np.sqrt(max(params_m, 0.05)),
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            zorder=3,
        )
        dx, dy = label_offsets.get(name, (4, 4))
        ax.annotate(name, (latency, ade), xytext=(dx, dy), textcoords="offset points", fontsize=5.8)
        rows.append(
            {
                "model": name,
                "latency_batch1_ms": latency,
                "ade_256_km": ade,
                "parameters_million": params_m,
                "peak_inference_memory_mb": entry.get("peak_inference_memory_mb"),
                "throughput_trajectories_s": entry.get("throughput_trajectories_s"),
            }
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch-one latency (ms, log scale)")
    ax.set_ylabel("256 s ADE (km, log scale)")
    finish_axes(ax)
    generated = save_figure(fig, "fig_efficiency_tradeoff")
    write_csv(
        "efficiency_tradeoff.csv",
        [
            "model",
            "latency_batch1_ms",
            "ade_256_km",
            "parameters_million",
            "peak_inference_memory_mb",
            "throughput_trajectories_s",
        ],
        rows,
    )
    return generated


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    main_results = load_json(MAIN_RESULTS)
    robustness = load_json(ROBUSTNESS_RESULTS)
    efficiency = load_json(EFFICIENCY_RESULTS)

    generated, dataset_summary = dataset_figures()
    generated += performance_figures(main_results)
    generated += robustness_figures(robustness)
    generated += efficiency_figure(efficiency)

    manifest = {
        "generator": str(Path(__file__).resolve()),
        "inputs": {
            str(RAW_DATA): sha256(RAW_DATA),
            str(MAIN_RESULTS): sha256(MAIN_RESULTS),
            str(ROBUSTNESS_RESULTS): sha256(ROBUSTNESS_RESULTS),
            str(EFFICIENCY_RESULTS): sha256(EFFICIENCY_RESULTS),
        },
        "dataset_summary": dataset_summary,
        "generated_figures": generated,
        "source_data_csv": sorted(path.name for path in OUTPUT.glob("*.csv")),
        "notes": [
            "All dataset panels derive from the preserved 1,000 complete long trajectories.",
            "All performance, robustness, and efficiency panels derive from formal JSON artifacts.",
            "Logarithmic error axes are used where model errors span more than one order of magnitude.",
        ],
    }
    with (OUTPUT / "eaai_figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
