#!/usr/bin/env python3
"""Multi-regime HGV simulation protocol for the formal v2.1 dataset.

The protocol keeps the paper's three primary lateral labels while adding a
second, independently recorded longitudinal factor:

* ``quasi_equilibrium``: angle of attack is chosen online to approximately
  stabilize the flight-path angle;
* ``skip_glide``: angle of attack follows a randomized speed-dependent law.

Bank schedules and their timing are randomized once per complete trajectory.
The ideal command and a smooth rate-limited achieved bank history are stored
separately as audit evidence but are never predictor inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from data_generation.data_generator import DCBNN_HGV_Simulator


MULTIREGIME_DATASET_PROTOCOL = "hgv_multiregime_state_v2_1"
LEGACY_MULTIREGIME_DATASET_PROTOCOL = "hgv_multiregime_state_v2"
VERTICAL_REGIMES = ("quasi_equilibrium", "skip_glide")
MANEUVER_TYPES = ("longitudinal", "turning", "weaving")
CONTROL_PARAMETER_NAMES = (
    "alpha_ld_deg",
    "alpha_max_deg",
    "velocity_threshold_low_mps",
    "velocity_threshold_high_mps",
    "qeg_gamma_reference_deg",
    "bank_amplitude_deg",
    "bank_initial_sign",
    "bank_segment_count",
    "bank_start_s",
    "bank_duration_1_s",
    "bank_duration_2_s",
    "bank_duration_3_s",
    "bank_gap_1_s",
    "bank_gap_2_s",
    "bank_rate_limit_deg_s",
)


@dataclass(frozen=True)
class ManeuverProfile:
    """One trajectory's disclosed longitudinal and lateral control settings."""

    vertical_regime: str
    maneuver_type: str
    alpha_ld_deg: float
    alpha_max_deg: float
    velocity_threshold_low_mps: float
    velocity_threshold_high_mps: float
    qeg_gamma_reference_deg: float
    bank_amplitude_deg: float
    bank_initial_sign: int
    bank_segment_count: int
    bank_start_s: float
    bank_durations_s: tuple[float, float, float]
    bank_gaps_s: tuple[float, float]
    bank_rate_limit_deg_s: float

    def __post_init__(self) -> None:
        if self.vertical_regime not in VERTICAL_REGIMES:
            raise ValueError(f"Unknown vertical regime: {self.vertical_regime}")
        if self.maneuver_type not in MANEUVER_TYPES:
            raise ValueError(f"Unknown maneuver type: {self.maneuver_type}")
        if self.bank_initial_sign not in (-1, 1):
            raise ValueError("bank_initial_sign must be -1 or 1.")
        if not 0 <= self.bank_segment_count <= 3:
            raise ValueError("bank_segment_count must be in [0, 3].")
        if self.maneuver_type == "longitudinal" and self.bank_segment_count != 0:
            raise ValueError("Longitudinal profiles must have zero bank segments.")
        if self.maneuver_type == "turning" and self.bank_segment_count != 1:
            raise ValueError("Turning profiles must have exactly one bank segment.")
        if self.maneuver_type == "weaving" and self.bank_segment_count not in (2, 3):
            raise ValueError("Weaving profiles must have two or three bank segments.")
        if self.bank_segment_count > 0 and not 1.5 <= self.bank_rate_limit_deg_s <= 3.0:
            raise ValueError("Nonzero-bank profiles require a 1.5--3.0 deg/s rate limit.")

    def parameter_vector(self) -> np.ndarray:
        """Return a fixed-width numeric representation for NPZ storage."""
        return np.asarray(
            [
                self.alpha_ld_deg,
                self.alpha_max_deg,
                self.velocity_threshold_low_mps,
                self.velocity_threshold_high_mps,
                self.qeg_gamma_reference_deg,
                self.bank_amplitude_deg,
                float(self.bank_initial_sign),
                float(self.bank_segment_count),
                self.bank_start_s,
                *self.bank_durations_s,
                *self.bank_gaps_s,
                self.bank_rate_limit_deg_s,
            ],
            dtype=np.float64,
        )

    def as_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bank_durations_s"] = list(self.bank_durations_s)
        payload["bank_gaps_s"] = list(self.bank_gaps_s)
        return payload


def sample_maneuver_profile(
    rng: np.random.Generator,
    *,
    vertical_regime: str,
    maneuver_type: str,
    unit_sample: np.ndarray | None = None,
) -> ManeuverProfile:
    """Map a space-filling unit sample to one disclosed control profile."""
    if unit_sample is None:
        unit = rng.random(15)
    else:
        unit = np.asarray(unit_sample, dtype=np.float64)
        if unit.shape != (15,) or np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("unit_sample must contain 15 values in [0, 1].")

    def scale(index: int, low: float, high: float) -> float:
        return float(low + unit[index] * (high - low))

    alpha_ld_deg = scale(0, 9.0, 12.0)
    alpha_max_deg = scale(1, 20.0, 25.0)
    # These thresholds are matched to the 4--7 km/s speed envelope so that
    # low-, transition-, and high-speed branches are all exercised.
    threshold_low = scale(2, 5_100.0, 5_500.0)
    threshold_high = scale(3, 5_900.0, 6_400.0)
    gamma_reference = scale(4, -0.20, -0.02)
    initial_sign = -1 if unit[5] < 0.5 else 1
    rate_limit = scale(14, 1.5, 3.0)

    if maneuver_type == "longitudinal":
        count = 0
        amplitude = 0.0
        start = 0.0
        durations = (0.0, 0.0, 0.0)
        gaps = (0.0, 0.0)
        rate_limit = 0.0
    elif maneuver_type == "turning":
        count = 1
        amplitude = scale(6, 20.0, 35.0)
        start = scale(7, 80.0, 180.0)
        durations = (scale(8, 320.0, 560.0), 0.0, 0.0)
        gaps = (0.0, 0.0)
    else:
        count = 2 if unit[6] < 0.5 else 3
        amplitude = scale(7, 20.0, 35.0)
        start = scale(8, 70.0, 130.0)
        sampled_durations = [scale(9 + index, 140.0, 220.0) for index in range(count)]
        sampled_gaps = [scale(12 + index, 45.0, 90.0) for index in range(count - 1)]
        durations = tuple(
            float(sampled_durations[index]) if index < count else 0.0
            for index in range(3)
        )
        gaps = tuple(
            float(sampled_gaps[index]) if index < count - 1 else 0.0
            for index in range(2)
        )

    return ManeuverProfile(
        vertical_regime=vertical_regime,
        maneuver_type=maneuver_type,
        alpha_ld_deg=alpha_ld_deg,
        alpha_max_deg=alpha_max_deg,
        velocity_threshold_low_mps=threshold_low,
        velocity_threshold_high_mps=threshold_high,
        qeg_gamma_reference_deg=gamma_reference,
        bank_amplitude_deg=amplitude,
        bank_initial_sign=initial_sign,
        bank_segment_count=count,
        bank_start_s=start,
        bank_durations_s=durations,
        bank_gaps_s=gaps,
        bank_rate_limit_deg_s=rate_limit,
    )


def maneuver_profile_from_vector(
    values: np.ndarray,
    *,
    vertical_regime: str,
    maneuver_type: str,
) -> ManeuverProfile:
    """Reconstruct a profile from the immutable NPZ parameter vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(CONTROL_PARAMETER_NAMES),):
        raise ValueError(
            f"Expected {len(CONTROL_PARAMETER_NAMES)} control parameters, "
            f"received {vector.shape}."
        )
    by_name = dict(zip(CONTROL_PARAMETER_NAMES, vector.tolist()))
    return ManeuverProfile(
        vertical_regime=vertical_regime,
        maneuver_type=maneuver_type,
        alpha_ld_deg=by_name["alpha_ld_deg"],
        alpha_max_deg=by_name["alpha_max_deg"],
        velocity_threshold_low_mps=by_name["velocity_threshold_low_mps"],
        velocity_threshold_high_mps=by_name["velocity_threshold_high_mps"],
        qeg_gamma_reference_deg=by_name["qeg_gamma_reference_deg"],
        bank_amplitude_deg=by_name["bank_amplitude_deg"],
        bank_initial_sign=int(round(by_name["bank_initial_sign"])),
        bank_segment_count=int(round(by_name["bank_segment_count"])),
        bank_start_s=by_name["bank_start_s"],
        bank_durations_s=(
            by_name["bank_duration_1_s"],
            by_name["bank_duration_2_s"],
            by_name["bank_duration_3_s"],
        ),
        bank_gaps_s=(
            by_name["bank_gap_1_s"],
            by_name["bank_gap_2_s"],
        ),
        bank_rate_limit_deg_s=by_name["bank_rate_limit_deg_s"],
    )


class MultiregimeHGVSimulator(DCBNN_HGV_Simulator):
    """Rotating-Earth 3-DOF simulator driven by a per-trajectory profile."""

    def bank_command(self, time_s: float, profile: ManeuverProfile) -> float:
        """Return the ideal piecewise-constant bank command."""
        if profile.bank_segment_count == 0:
            return 0.0

        cursor = profile.bank_start_s
        for index in range(profile.bank_segment_count):
            end = cursor + profile.bank_durations_s[index]
            if cursor <= time_s < end:
                sign = profile.bank_initial_sign * ((-1) ** index)
                return float(np.deg2rad(sign * profile.bank_amplitude_deg))
            if index < profile.bank_segment_count - 1:
                cursor = end + profile.bank_gaps_s[index]
        return 0.0

    @staticmethod
    def _raised_cosine_progress(value: float) -> float:
        clipped = float(np.clip(value, 0.0, 1.0))
        return 0.5 - 0.5 * float(np.cos(np.pi * clipped))

    def bank_angle(self, time_s: float, profile: ManeuverProfile) -> float:
        """Return the smooth achieved bank angle used by the dynamics.

        Each nonzero interval has raised-cosine entry and exit ramps.  The ramp
        duration is derived from the sampled rate limit, so the maximum slope
        never exceeds ``bank_rate_limit_deg_s``.
        """
        if profile.bank_segment_count == 0:
            return 0.0

        cursor = profile.bank_start_s
        amplitude = profile.bank_amplitude_deg
        ramp_s = amplitude * np.pi / (2.0 * profile.bank_rate_limit_deg_s)
        for index in range(profile.bank_segment_count):
            duration = profile.bank_durations_s[index]
            end = cursor + duration
            if cursor <= time_s < end:
                local = time_s - cursor
                if local < ramp_s:
                    envelope = self._raised_cosine_progress(local / ramp_s)
                elif local > duration - ramp_s:
                    envelope = self._raised_cosine_progress((duration - local) / ramp_s)
                else:
                    envelope = 1.0
                sign = profile.bank_initial_sign * ((-1) ** index)
                return float(np.deg2rad(sign * amplitude * envelope))
            if index < profile.bank_segment_count - 1:
                cursor = end + profile.bank_gaps_s[index]
        return 0.0

    @staticmethod
    def _speed_law_alpha(velocity_mps: float, profile: ManeuverProfile) -> float:
        low = profile.velocity_threshold_low_mps
        high = profile.velocity_threshold_high_mps
        if velocity_mps <= low:
            alpha_deg = profile.alpha_ld_deg
        elif velocity_mps >= high:
            alpha_deg = profile.alpha_max_deg
        else:
            midpoint = 0.5 * (low + high)
            alpha_mid = 0.5 * (profile.alpha_max_deg + profile.alpha_ld_deg)
            alpha_balance = 0.5 * (profile.alpha_max_deg - profile.alpha_ld_deg)
            phase = np.pi * (velocity_mps - midpoint) / (high - low)
            alpha_deg = alpha_mid + alpha_balance * np.sin(phase)
        return float(np.deg2rad(np.clip(alpha_deg, 0.0, 25.0)))

    def _equilibrium_alpha(
        self,
        state: np.ndarray,
        bank: float,
        profile: ManeuverProfile,
    ) -> float:
        r, _, latitude, velocity, gamma, psi = state
        altitude = r - self.R_earth
        density, sound_speed = self.atmospheric_model(altitude)
        mach = velocity / sound_speed
        gravity = self.standard_gravity_model(altitude)
        omega = self.earth_rotation_rate
        sin_gamma, cos_gamma = np.sin(gamma), np.cos(gamma)
        sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
        cos_psi = np.cos(psi)

        non_lift = (
            -(gravity / velocity - velocity / r) * cos_gamma
            + 2.0 * omega * cos_lat * np.sin(psi)
            + omega**2
            * r
            * cos_lat
            / velocity
            * (cos_gamma * cos_lat + sin_gamma * sin_lat * cos_psi)
        )
        reference_gamma = np.deg2rad(profile.qeg_gamma_reference_deg)
        target_gamma_rate = -0.0025 * (gamma - reference_gamma)
        safe_cos_bank = max(float(np.cos(bank)), 0.25)
        required_lift = (
            (target_gamma_rate - non_lift)
            * self.m
            * velocity
            / safe_cos_bank
        )
        dynamic_pressure = max(0.5 * density * velocity**2, 1e-9)
        required_cl = required_lift / (dynamic_pressure * self.S)

        mach_eval = float(np.clip(mach, 5.0, 25.0))
        quadratic = 0.00037
        linear = 0.05 - 0.00083 * mach_eval
        constant = (
            -0.0561
            - 0.00443 * mach_eval
            + 0.00032 * mach_eval**2
            - required_cl
        )
        discriminant = max(linear**2 - 4.0 * quadratic * constant, 0.0)
        alpha_deg = float(
            np.clip(
                (-linear + np.sqrt(discriminant)) / (2.0 * quadratic),
                0.0,
                25.0,
            )
        )
        return float(np.deg2rad(alpha_deg))

    def control_inputs_profile(
        self,
        time_s: float,
        state: np.ndarray,
        profile: ManeuverProfile,
    ) -> tuple[float, float]:
        bank = self.bank_angle(time_s, profile)
        if profile.vertical_regime == "quasi_equilibrium":
            alpha = self._equilibrium_alpha(state, bank, profile)
        else:
            alpha = self._speed_law_alpha(float(state[3]), profile)
        return alpha, bank

    def hgv_dynamics_profile(
        self,
        time_s: float,
        state: np.ndarray,
        profile: ManeuverProfile,
    ) -> np.ndarray:
        """Evaluate the formal rotating-spherical-Earth 3-DOF equations."""
        state_arr = np.asarray(state, dtype=np.float64)
        if state_arr.shape != (6,) or not np.all(np.isfinite(state_arr)):
            raise ValueError("HGV state must contain six finite values.")
        r, _, latitude, velocity, gamma, psi = state_arr
        if r <= self.R_earth or velocity <= 0.0:
            raise ValueError("HGV dynamics require positive altitude and speed.")

        altitude = r - self.R_earth
        density, sound_speed = self.atmospheric_model(altitude)
        mach = velocity / sound_speed
        alpha, bank = self.control_inputs_profile(time_s, state_arr, profile)
        cl, cd, _ = self.aerodynamic_coefficients(alpha, bank, mach)
        dynamic_pressure = 0.5 * density * velocity**2
        lift = cl * dynamic_pressure * self.S
        drag = cd * dynamic_pressure * self.S
        gravity = self.standard_gravity_model(altitude)
        omega = self.earth_rotation_rate

        sin_gamma, cos_gamma = np.sin(gamma), np.cos(gamma)
        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
        safe_cos_gamma = np.copysign(max(abs(cos_gamma), 1e-8), cos_gamma)
        safe_cos_lat = np.copysign(max(abs(cos_lat), 1e-8), cos_lat)

        derivatives = np.asarray(
            [
                velocity * sin_gamma,
                velocity * cos_gamma * sin_psi / (r * safe_cos_lat),
                velocity * cos_gamma * cos_psi / r,
                -drag / self.m
                - gravity * sin_gamma
                + omega**2
                * r
                * cos_lat
                * (sin_gamma * cos_lat - cos_gamma * sin_lat * cos_psi),
                lift * np.cos(bank) / (self.m * velocity)
                - (gravity / velocity - velocity / r) * cos_gamma
                + 2.0 * omega * cos_lat * sin_psi
                + omega**2
                * r
                * cos_lat
                / velocity
                * (cos_gamma * cos_lat + sin_gamma * sin_lat * cos_psi),
                lift * np.sin(bank) / (self.m * velocity * safe_cos_gamma)
                + velocity * cos_gamma * sin_psi * np.tan(latitude) / r
                + 2.0 * omega * sin_lat
                + omega**2
                * r
                * sin_psi
                * sin_lat
                * cos_lat
                / (velocity * safe_cos_gamma)
                - 2.0 * omega * cos_lat * cos_psi * np.tan(gamma),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(derivatives)):
            raise FloatingPointError("Non-finite derivative in HGV dynamics.")
        return derivatives

    def trajectory_diagnostics(
        self,
        times_s: np.ndarray,
        states: np.ndarray,
        profile: ManeuverProfile,
    ) -> dict[str, np.ndarray]:
        """Reconstruct commands and basic loads along one accepted trajectory."""
        controls = np.asarray(
            [
                self.control_inputs_profile(float(time_s), state, profile)
                for time_s, state in zip(times_s, states)
            ],
            dtype=np.float64,
        )
        alpha, bank = controls[:, 0], controls[:, 1]
        bank_command = np.asarray(
            [self.bank_command(float(time_s), profile) for time_s in times_s],
            dtype=np.float64,
        )
        dynamic_pressure: list[float] = []
        load_factor: list[float] = []
        heat_rate_proxy: list[float] = []
        for state, alpha_i, bank_i in zip(states, alpha, bank):
            altitude = float(state[0] - self.R_earth)
            velocity = float(state[3])
            density, sound_speed = self.atmospheric_model(altitude)
            mach = velocity / sound_speed
            cl, cd, _ = self.aerodynamic_coefficients(alpha_i, bank_i, mach)
            q = 0.5 * density * velocity**2
            lift = cl * q * self.S
            drag = cd * q * self.S
            dynamic_pressure.append(q)
            load_factor.append(np.hypot(lift, drag) / (self.m * self.g0))
            heat_rate_proxy.append(np.sqrt(max(density, 0.0)) * velocity**3)
        return {
            "alpha": alpha,
            "bank": bank,
            "bank_command": bank_command,
            "dynamic_pressure_pa": np.asarray(dynamic_pressure),
            "load_factor_g": np.asarray(load_factor),
            "heat_rate_proxy": np.asarray(heat_rate_proxy),
        }
