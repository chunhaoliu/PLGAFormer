import numpy as np

from scripts.generate_taes_horizon_error_profiles import (
    PRED_LEN,
    profile_from_accumulators,
)


def test_profile_from_accumulators_matches_metric_definitions():
    # Two trajectories with two windows each. Their per-lead displacement
    # means are 2 m and 4 m, so the equal-trajectory FDE curve is 3 m.
    lead_sum = np.vstack(
        [
            np.full(PRED_LEN, 4.0),
            np.full(PRED_LEN, 8.0),
        ]
    )
    window_counts = np.asarray([2, 2])

    # Four windows x three Cartesian components, each with squared error 9 m^2.
    component_square_sum = np.full(PRED_LEN, 4 * 3 * 9.0)
    result = profile_from_accumulators(
        lead_sum,
        window_counts,
        component_square_sum,
        total_windows=4,
    )

    np.testing.assert_allclose(result["fde"], 3.0)
    np.testing.assert_allclose(result["ade"], 3.0)
    np.testing.assert_allclose(result["rmse_cart_m"], 3.0)


def test_profile_from_accumulators_rejects_missing_trajectory_windows():
    lead_sum = np.zeros((2, PRED_LEN))
    window_counts = np.asarray([1, 0])
    square_sum = np.ones(PRED_LEN)

    try:
        profile_from_accumulators(lead_sum, window_counts, square_sum, 1)
    except ValueError as exc:
        assert "positive window count" in str(exc)
    else:
        raise AssertionError("Expected a missing-trajectory validation error.")
