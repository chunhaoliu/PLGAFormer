from __future__ import annotations

import numpy as np

from data_generation.multiregime_protocol import (
    CONTROL_PARAMETER_NAMES,
    MULTIREGIME_DATASET_PROTOCOL,
    ManeuverProfile,
    MultiregimeHGVSimulator,
    sample_maneuver_profile,
)
from utils.trajectory_protocol import TRAJECTORY_LEVEL_PROTOCOLS


def _profile(maneuver_type: str, count: int) -> ManeuverProfile:
    return ManeuverProfile(
        vertical_regime="skip_glide",
        maneuver_type=maneuver_type,
        alpha_ld_deg=10.0,
        alpha_max_deg=24.0,
        velocity_threshold_low_mps=3_000.0,
        velocity_threshold_high_mps=5_000.0,
        qeg_gamma_reference_deg=-0.1,
        bank_amplitude_deg=30.0 if count else 0.0,
        bank_initial_sign=1,
        bank_segment_count=count,
        bank_start_s=100.0 if count else 0.0,
        bank_durations_s=(100.0, 100.0 if count > 1 else 0.0, 100.0 if count > 2 else 0.0),
        bank_gaps_s=(50.0 if count > 1 else 0.0, 50.0 if count > 2 else 0.0),
        bank_rate_limit_deg_s=2.0 if count else 0.0,
    )


def test_multiregime_protocol_is_trajectory_level() -> None:
    assert MULTIREGIME_DATASET_PROTOCOL in TRAJECTORY_LEVEL_PROTOCOLS


def test_profile_parameter_vector_has_declared_width() -> None:
    profile = _profile("weaving", 3)
    assert profile.parameter_vector().shape == (len(CONTROL_PARAMETER_NAMES),)


def test_bank_semantics_for_primary_maneuver_classes() -> None:
    simulator = MultiregimeHGVSimulator()
    longitudinal = _profile("longitudinal", 0)
    turning = _profile("turning", 1)
    weaving = _profile("weaving", 3)

    assert simulator.bank_angle(150.0, longitudinal) == 0.0
    assert np.isclose(np.rad2deg(simulator.bank_angle(150.0, turning)), 30.0)
    assert simulator.bank_angle(250.0, turning) == 0.0
    assert np.isclose(np.rad2deg(simulator.bank_angle(150.0, weaving)), 30.0)
    assert np.isclose(np.rad2deg(simulator.bank_angle(300.0, weaving)), -30.0)
    assert np.isclose(np.rad2deg(simulator.bank_angle(450.0, weaving)), 30.0)
    assert np.isclose(np.rad2deg(simulator.bank_command(101.0, turning)), 30.0)
    assert np.rad2deg(simulator.bank_angle(101.0, turning)) < 1.0


def test_speed_law_respects_low_mid_high_commands() -> None:
    profile = _profile("longitudinal", 0)
    simulator = MultiregimeHGVSimulator()
    low = np.rad2deg(simulator._speed_law_alpha(2_000.0, profile))
    middle = np.rad2deg(simulator._speed_law_alpha(4_000.0, profile))
    high = np.rad2deg(simulator._speed_law_alpha(6_000.0, profile))
    assert np.isclose(low, 10.0)
    assert np.isclose(middle, 17.0)
    assert np.isclose(high, 24.0)


def test_sampled_profiles_preserve_taxonomy_constraints() -> None:
    rng = np.random.default_rng(42)
    for vertical in ("quasi_equilibrium", "skip_glide"):
        expected_counts = {"longitudinal": {0}, "turning": {1}, "weaving": {2, 3}}
        for maneuver, allowed_counts in expected_counts.items():
            profile = sample_maneuver_profile(
                rng,
                vertical_regime=vertical,
                maneuver_type=maneuver,
            )
            assert profile.bank_segment_count in allowed_counts
            assert 9.0 <= profile.alpha_ld_deg <= 12.0
            assert 20.0 <= profile.alpha_max_deg <= 25.0
            assert 5_100.0 <= profile.velocity_threshold_low_mps <= 5_500.0
            assert 5_900.0 <= profile.velocity_threshold_high_mps <= 6_400.0
