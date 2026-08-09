#!/usr/bin/env python3
"""PIT-aligned radar-observation protocol for long-horizon HGV prediction.

This module deliberately keeps four artifact layers separate:

1. clean simulated HGV states;
2. noisy radar range/azimuth/elevation measurements;
3. causally filtered tracking states used as model inputs;
4. clean future states used as prediction targets.

The tracking filter never consumes a future measurement.  Complete trajectory
IDs are split before any supervised windows are constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from utils.trajectory_protocol import split_trajectory_ids


PIT_RADAR_PROTOCOL = "pit_aligned_radar_v1"
RADAR_OBSERVATION_PROTOCOL = "radar_razel_gaussian_v1"
TRACKING_FILTER_PROTOCOL = "causal_ecef_cv_kf_v1"
STATE_FEATURE_NAMES = ("r", "lambda", "phi", "V", "gamma", "psi")
TARGET_FEATURE_NAMES = ("r", "lambda", "phi")
DEFAULT_EARTH_RADIUS_M = 6_378_000.0


@dataclass(frozen=True)
class RadarConfig:
    """Radar site and independent measurement-error standard deviations."""

    longitude_deg: float = 12.0
    latitude_deg: float = 0.5
    altitude_m: float = 1_000.0
    range_std_m: float = 100.0
    azimuth_std_rad: float = 0.5e-3
    elevation_std_rad: float = 0.5e-3


@dataclass(frozen=True)
class TrackingConfig:
    """Causal constant-velocity Kalman-filter configuration."""

    process_acceleration_std_mps2: float = 8.0
    initial_velocity_std_mps: float = 2_000.0


def spherical_to_ecef(states: np.ndarray) -> np.ndarray:
    """Convert ``[..., r, longitude, latitude, ...]`` to ECEF metres."""
    arr = np.asarray(states, dtype=np.float64)
    if arr.shape[-1] < 3:
        raise ValueError("Spherical states must have at least three features.")
    radius = arr[..., 0]
    longitude = arr[..., 1]
    latitude = arr[..., 2]
    cos_latitude = np.cos(latitude)
    return np.stack(
        [
            radius * cos_latitude * np.cos(longitude),
            radius * cos_latitude * np.sin(longitude),
            radius * np.sin(latitude),
        ],
        axis=-1,
    )


def ecef_to_spherical(positions_ecef: np.ndarray) -> np.ndarray:
    """Convert ECEF positions in metres to ``[..., r, longitude, latitude]``."""
    positions = np.asarray(positions_ecef, dtype=np.float64)
    if positions.shape[-1] != 3:
        raise ValueError("ECEF positions must have final dimension 3.")
    x, y, z = np.moveaxis(positions, -1, 0)
    horizontal = np.hypot(x, y)
    radius = np.sqrt(x * x + y * y + z * z)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(z, horizontal)
    return np.stack([radius, longitude, latitude], axis=-1)


def _enu_basis(longitude_rad: float, latitude_rad: float) -> np.ndarray:
    """Return a matrix whose columns are local east, north, and up in ECEF."""
    sin_lon, cos_lon = np.sin(longitude_rad), np.cos(longitude_rad)
    sin_lat, cos_lat = np.sin(latitude_rad), np.cos(latitude_rad)
    east = np.asarray([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.asarray(
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        dtype=np.float64,
    )
    up = np.asarray(
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        dtype=np.float64,
    )
    return np.column_stack([east, north, up])


def radar_site_ecef(
    radar: RadarConfig,
    *,
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radar-site ECEF position and its ENU-to-ECEF basis."""
    longitude = np.deg2rad(radar.longitude_deg)
    latitude = np.deg2rad(radar.latitude_deg)
    site_state = np.asarray(
        [earth_radius_m + radar.altitude_m, longitude, latitude],
        dtype=np.float64,
    )
    return spherical_to_ecef(site_state), _enu_basis(longitude, latitude)


def ecef_to_radar_razel(
    positions_ecef: np.ndarray,
    radar: RadarConfig,
    *,
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
) -> np.ndarray:
    """Convert ECEF target positions to radar range, azimuth, and elevation."""
    positions = np.asarray(positions_ecef, dtype=np.float64)
    site, basis = radar_site_ecef(radar, earth_radius_m=earth_radius_m)
    local_enu = (positions - site) @ basis
    east, north, up = np.moveaxis(local_enu, -1, 0)
    ranges = np.linalg.norm(local_enu, axis=-1)
    if np.any(ranges <= 0.0):
        raise ValueError("Radar range must be positive.")
    azimuth = np.arctan2(east, north)
    elevation = np.arctan2(up, np.hypot(east, north))
    return np.stack([ranges, azimuth, elevation], axis=-1)


def radar_razel_to_ecef(
    measurements: np.ndarray,
    radar: RadarConfig,
    *,
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
) -> np.ndarray:
    """Convert radar range, azimuth, and elevation measurements to ECEF."""
    measurements = np.asarray(measurements, dtype=np.float64)
    if measurements.shape[-1] != 3:
        raise ValueError("Radar measurements must have final dimension 3.")
    ranges, azimuth, elevation = np.moveaxis(measurements, -1, 0)
    horizontal = ranges * np.cos(elevation)
    local_enu = np.stack(
        [
            horizontal * np.sin(azimuth),
            horizontal * np.cos(azimuth),
            ranges * np.sin(elevation),
        ],
        axis=-1,
    )
    site, basis = radar_site_ecef(radar, earth_radius_m=earth_radius_m)
    return site + local_enu @ basis.T


def _measurement_covariances_ecef(
    ideal_measurements: np.ndarray,
    radar: RadarConfig,
    *,
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
) -> np.ndarray:
    """Propagate radar-coordinate variances into an ECEF position covariance."""
    ideal = np.asarray(ideal_measurements, dtype=np.float64)
    _, basis = radar_site_ecef(radar, earth_radius_m=earth_radius_m)
    variances = np.diag(
        [
            radar.range_std_m**2,
            radar.azimuth_std_rad**2,
            radar.elevation_std_rad**2,
        ]
    )
    covariances: list[np.ndarray] = []
    for range_m, azimuth, elevation in ideal:
        sin_az, cos_az = np.sin(azimuth), np.cos(azimuth)
        sin_el, cos_el = np.sin(elevation), np.cos(elevation)
        jacobian = np.asarray(
            [
                [cos_el * sin_az, range_m * cos_el * cos_az, -range_m * sin_el * sin_az],
                [cos_el * cos_az, -range_m * cos_el * sin_az, -range_m * sin_el * cos_az],
                [sin_el, 0.0, range_m * cos_el],
            ],
            dtype=np.float64,
        )
        local_covariance = jacobian @ variances @ jacobian.T
        ecef_covariance = basis @ local_covariance @ basis.T
        covariances.append(ecef_covariance + np.eye(3) * 1e-6)
    return np.stack(covariances, axis=0)


def simulate_radar_observations(
    true_states: np.ndarray,
    *,
    rng: np.random.Generator,
    radar: RadarConfig = RadarConfig(),
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
) -> dict[str, np.ndarray]:
    """Generate deterministic-seed noisy radar measurements from clean truth."""
    true_ecef = spherical_to_ecef(true_states)
    ideal = ecef_to_radar_razel(true_ecef, radar, earth_radius_m=earth_radius_m)
    noise = np.column_stack(
        [
            rng.normal(0.0, radar.range_std_m, size=len(ideal)),
            rng.normal(0.0, radar.azimuth_std_rad, size=len(ideal)),
            rng.normal(0.0, radar.elevation_std_rad, size=len(ideal)),
        ]
    )
    noisy = ideal + noise
    noisy[:, 0] = np.maximum(noisy[:, 0], 1.0)
    noisy_ecef = radar_razel_to_ecef(noisy, radar, earth_radius_m=earth_radius_m)
    covariance_ecef = _measurement_covariances_ecef(
        ideal,
        radar,
        earth_radius_m=earth_radius_m,
    )
    return {
        "ideal_razel": ideal,
        "noisy_razel": noisy,
        "noisy_ecef": noisy_ecef,
        "measurement_covariance_ecef": covariance_ecef,
    }


def causal_constant_velocity_filter(
    measurements_ecef: np.ndarray,
    measurement_covariances_ecef: np.ndarray,
    *,
    sampling_interval_s: float,
    tracking: TrackingConfig = TrackingConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Filter ECEF positions using only measurements available up to each step."""
    measurements = np.asarray(measurements_ecef, dtype=np.float64)
    covariances = np.asarray(measurement_covariances_ecef, dtype=np.float64)
    if measurements.ndim != 2 or measurements.shape[1] != 3:
        raise ValueError("measurements_ecef must have shape [time, 3].")
    if covariances.shape != (len(measurements), 3, 3):
        raise ValueError("measurement covariance shape must be [time, 3, 3].")
    dt = float(sampling_interval_s)
    if dt <= 0.0:
        raise ValueError("sampling_interval_s must be positive.")

    identity3 = np.eye(3, dtype=np.float64)
    transition = np.block(
        [[identity3, dt * identity3], [np.zeros((3, 3)), identity3]]
    )
    observation = np.block([identity3, np.zeros((3, 3))])
    accel_variance = tracking.process_acceleration_std_mps2**2
    process_block = np.asarray(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
        dtype=np.float64,
    )
    process_covariance = accel_variance * np.kron(process_block, identity3)
    # np.kron orders the state as [position xyz, velocity xyz] for this block.

    state = np.zeros(6, dtype=np.float64)
    state[:3] = measurements[0]
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[:3, :3] = covariances[0]
    covariance[3:, 3:] = identity3 * tracking.initial_velocity_std_mps**2
    filtered_positions = np.empty_like(measurements)
    filtered_velocities = np.empty_like(measurements)
    identity6 = np.eye(6, dtype=np.float64)

    for index, measurement in enumerate(measurements):
        if index > 0:
            state = transition @ state
            covariance = (
                transition @ covariance @ transition.T + process_covariance
            )
        innovation_covariance = (
            observation @ covariance @ observation.T + covariances[index]
        )
        gain = np.linalg.solve(
            innovation_covariance,
            observation @ covariance,
        ).T
        innovation = measurement - observation @ state
        state = state + gain @ innovation
        correction = identity6 - gain @ observation
        covariance = (
            correction @ covariance @ correction.T
            + gain @ covariances[index] @ gain.T
        )
        filtered_positions[index] = state[:3]
        filtered_velocities[index] = state[3:]

    return filtered_positions, filtered_velocities


def ecef_track_to_spherical_states(
    positions_ecef: np.ndarray,
    velocities_ecef: np.ndarray,
) -> np.ndarray:
    """Convert a filtered ECEF position/velocity track to six HGV features."""
    positions = np.asarray(positions_ecef, dtype=np.float64)
    velocities = np.asarray(velocities_ecef, dtype=np.float64)
    if positions.shape != velocities.shape or positions.shape[-1] != 3:
        raise ValueError("ECEF position and velocity arrays must share shape [time, 3].")
    spherical = ecef_to_spherical(positions)
    states = np.zeros((len(spherical), 6), dtype=np.float64)
    states[:, :3] = spherical
    for index, ((_, longitude, latitude), velocity) in enumerate(
        zip(spherical, velocities)
    ):
        basis = _enu_basis(float(longitude), float(latitude))
        east, north, up = basis.T @ velocity
        speed = float(np.linalg.norm(velocity))
        states[index, 3] = speed
        if speed > 1e-9:
            states[index, 4] = np.arcsin(np.clip(up / speed, -1.0, 1.0))
            states[index, 5] = np.arctan2(east, north)
    return states


def track_noisy_radar_states(
    true_states: np.ndarray,
    *,
    sampling_interval_s: float,
    rng: np.random.Generator,
    radar: RadarConfig = RadarConfig(),
    tracking: TrackingConfig = TrackingConfig(),
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
) -> dict[str, np.ndarray]:
    """Generate noisy radar observations and their strictly causal state track."""
    observations = simulate_radar_observations(
        true_states,
        rng=rng,
        radar=radar,
        earth_radius_m=earth_radius_m,
    )
    positions, velocities = causal_constant_velocity_filter(
        observations["noisy_ecef"],
        observations["measurement_covariance_ecef"],
        sampling_interval_s=sampling_interval_s,
        tracking=tracking,
    )
    return {
        **observations,
        "tracked_ecef_position": positions,
        "tracked_ecef_velocity": velocities,
        "tracked_states": ecef_track_to_spherical_states(positions, velocities),
    }


def uniform_window_starts(
    *,
    points_per_trajectory: int,
    seq_len: int,
    pred_len: int,
    burn_in_steps: int,
    origins_per_trajectory: int,
) -> np.ndarray:
    """Select a fixed, trajectory-balanced set of training forecast origins."""
    max_start = int(points_per_trajectory) - int(seq_len) - int(pred_len)
    first_start = max(0, int(burn_in_steps))
    count = int(origins_per_trajectory)
    if max_start < first_start:
        raise ValueError("Trajectory is too short after the requested tracking burn-in.")
    available = max_start - first_start + 1
    if count <= 0 or count > available:
        raise ValueError(
            f"origins_per_trajectory must be in [1, {available}], got {count}."
        )
    if count == 1:
        return np.asarray([first_start], dtype=np.int64)
    starts = np.rint(np.linspace(first_start, max_start, count)).astype(np.int64)
    if len(np.unique(starts)) != count:
        raise RuntimeError("Uniform origin selection produced duplicate starts.")
    return starts


def strided_window_starts(
    *,
    points_per_trajectory: int,
    seq_len: int,
    pred_len: int,
    burn_in_steps: int,
    stride: int,
) -> np.ndarray:
    """Return deterministic validation/test starts after causal-filter burn-in."""
    max_start = int(points_per_trajectory) - int(seq_len) - int(pred_len)
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if max_start < burn_in_steps:
        raise ValueError("Trajectory is too short after the requested tracking burn-in.")
    return np.arange(int(burn_in_steps), max_start + 1, int(stride), dtype=np.int64)


def build_paired_windows(
    input_trajectories: np.ndarray,
    target_trajectories: np.ndarray,
    labels: Sequence[Any],
    trajectory_ids: Sequence[int],
    *,
    seq_len: int,
    pred_len: int,
    starts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build input windows from tracked states and targets from clean truth."""
    inputs = np.asarray(input_trajectories)
    targets = np.asarray(target_trajectories)
    labels_arr = np.asarray(labels)
    ids = np.asarray(trajectory_ids, dtype=np.int64)
    starts_arr = np.asarray(starts, dtype=np.int64)
    if inputs.shape != targets.shape or inputs.ndim != 3 or inputs.shape[-1] != 6:
        raise ValueError("Input and target trajectories must share shape [N, T, 6].")
    if len(inputs) != len(labels_arr) or len(inputs) != len(ids):
        raise ValueError("Labels and trajectory IDs must match the trajectory count.")
    if starts_arr.ndim != 1 or len(starts_arr) == 0:
        raise ValueError("At least one 1D window-start index is required.")

    windows_x: list[np.ndarray] = []
    windows_y: list[np.ndarray] = []
    window_labels: list[Any] = []
    source_ids: list[int] = []
    window_starts: list[int] = []
    for input_track, target_track, label, trajectory_id in zip(
        inputs,
        targets,
        labels_arr,
        ids,
    ):
        for start in starts_arr:
            split = int(start) + int(seq_len)
            end = split + int(pred_len)
            if int(start) < 0 or end > len(input_track):
                raise ValueError(f"Invalid window start {start} for trajectory length {len(input_track)}.")
            windows_x.append(input_track[int(start):split, :6])
            windows_y.append(target_track[split:end, :3])
            window_labels.append(label)
            source_ids.append(int(trajectory_id))
            window_starts.append(int(start))

    return {
        "X": np.asarray(windows_x, dtype=np.float32),
        "y": np.asarray(windows_y, dtype=np.float32),
        "maneuver_labels": np.asarray(window_labels),
        "trajectory_ids": np.asarray(source_ids, dtype=np.int64),
        "window_starts": np.asarray(window_starts, dtype=np.int64),
    }


def assemble_pit_radar_dataset(
    clean_trajectories: np.ndarray,
    tracked_trajectories: np.ndarray,
    noisy_radar_measurements: np.ndarray,
    labels: Sequence[Any],
    *,
    sampling_interval_s: float = 2.0,
    seq_len: int = 256,
    pred_len: int = 256,
    tracking_burn_in_steps: int = 32,
    train_origins_per_trajectory: int = 16,
    eval_window_stride: int = 5,
    split_seed: int = 42,
    generation_seed: int = 42,
    radar: RadarConfig = RadarConfig(),
    tracking: TrackingConfig = TrackingConfig(),
) -> dict[str, np.ndarray]:
    """Create a leakage-safe supervised payload from complete trajectories."""
    clean = np.asarray(clean_trajectories, dtype=np.float64)
    tracked = np.asarray(tracked_trajectories, dtype=np.float64)
    radar_measurements = np.asarray(noisy_radar_measurements, dtype=np.float64)
    labels_arr = np.asarray(labels)
    if clean.shape != tracked.shape or clean.ndim != 3 or clean.shape[-1] != 6:
        raise ValueError("Clean and tracked trajectories must share shape [N, T, 6].")
    if radar_measurements.shape != (clean.shape[0], clean.shape[1], 3):
        raise ValueError("Noisy radar measurements must have shape [N, T, 3].")
    if len(labels_arr) != len(clean):
        raise ValueError("One maneuver label is required per complete trajectory.")

    trajectory_ids = np.arange(len(clean), dtype=np.int64)
    splits = split_trajectory_ids(labels_arr, seed=split_seed)
    train_starts = uniform_window_starts(
        points_per_trajectory=clean.shape[1],
        seq_len=seq_len,
        pred_len=pred_len,
        burn_in_steps=tracking_burn_in_steps,
        origins_per_trajectory=train_origins_per_trajectory,
    )
    eval_starts = strided_window_starts(
        points_per_trajectory=clean.shape[1],
        seq_len=seq_len,
        pred_len=pred_len,
        burn_in_steps=tracking_burn_in_steps,
        stride=eval_window_stride,
    )

    payload: dict[str, np.ndarray] = {
        "dataset_protocol": np.asarray(PIT_RADAR_PROTOCOL),
        "observation_protocol": np.asarray(RADAR_OBSERVATION_PROTOCOL),
        "tracking_filter_protocol": np.asarray(TRACKING_FILTER_PROTOCOL),
        "input_source": np.asarray("causally_filtered_noisy_radar_tracking"),
        "target_source": np.asarray("clean_simulator_truth"),
        "state_feature_names": np.asarray(STATE_FEATURE_NAMES),
        "target_feature_names": np.asarray(TARGET_FEATURE_NAMES),
        "sampling_interval_s": np.asarray(sampling_interval_s, dtype=np.float64),
        "points_per_trajectory": np.asarray(clean.shape[1], dtype=np.int64),
        "trajectory_duration_s": np.asarray(
            (clean.shape[1] - 1) * sampling_interval_s,
            dtype=np.float64,
        ),
        "seq_len": np.asarray(seq_len, dtype=np.int64),
        "pred_len": np.asarray(pred_len, dtype=np.int64),
        # Legacy readers use window_stride; it is explicitly the eval stride.
        "window_stride": np.asarray(eval_window_stride, dtype=np.int64),
        "eval_window_stride": np.asarray(eval_window_stride, dtype=np.int64),
        "train_origin_policy": np.asarray("uniform_per_trajectory"),
        "train_origins_per_trajectory": np.asarray(
            train_origins_per_trajectory,
            dtype=np.int64,
        ),
        "tracking_burn_in_steps": np.asarray(tracking_burn_in_steps, dtype=np.int64),
        "generation_seed": np.asarray(generation_seed, dtype=np.int64),
        "split_seed": np.asarray(split_seed, dtype=np.int64),
        "physical_horizon_steps": np.asarray([32, 64, 128, 256], dtype=np.int64),
        "physical_horizon_s": np.asarray([64, 128, 256, 512], dtype=np.int64),
        "trajectory_ids": trajectory_ids,
        "trajectory_labels": labels_arr,
        "clean_trajectories": clean.astype(np.float32),
        "tracked_trajectories": tracked.astype(np.float32),
        "noisy_radar_razel": radar_measurements.astype(np.float32),
        "radar_config_keys": np.asarray(tuple(asdict(radar).keys())),
        "radar_config_values": np.asarray(tuple(asdict(radar).values()), dtype=np.float64),
        "tracking_config_keys": np.asarray(tuple(asdict(tracking).keys())),
        "tracking_config_values": np.asarray(
            tuple(asdict(tracking).values()),
            dtype=np.float64,
        ),
    }

    for split_name in ("train", "val", "test"):
        selected_ids = splits[split_name]
        selected_starts = train_starts if split_name == "train" else eval_starts
        windows = build_paired_windows(
            tracked[selected_ids],
            clean[selected_ids],
            labels_arr[selected_ids],
            selected_ids,
            seq_len=seq_len,
            pred_len=pred_len,
            starts=selected_starts,
        )
        payload[f"complete_trajectory_ids_{split_name}"] = selected_ids
        payload[f"X_{split_name}"] = windows["X"]
        payload[f"y_{split_name}"] = windows["y"]
        payload[f"maneuver_labels_{split_name}"] = windows["maneuver_labels"]
        payload[f"trajectory_ids_{split_name}"] = windows["trajectory_ids"]
        payload[f"window_starts_{split_name}"] = windows["window_starts"]
    return payload
