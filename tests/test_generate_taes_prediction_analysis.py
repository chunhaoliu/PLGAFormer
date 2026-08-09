from __future__ import annotations

import numpy as np

from scripts.generate_taes_prediction_analysis import select_representative_windows


def test_select_representative_windows_uses_median_trajectory_and_central_window():
    trajectory_ids = np.asarray([1, 1, 1, 2, 2, 10, 10, 11, 11, 20, 20, 21, 21])
    labels = np.asarray(
        [
            "longitudinal",
            "longitudinal",
            "longitudinal",
            "longitudinal",
            "longitudinal",
            "turning",
            "turning",
            "turning",
            "turning",
            "weaving",
            "weaving",
            "weaving",
            "weaving",
        ]
    )
    metrics = {
        "1": {"ade": 10.0},
        "2": {"ade": 30.0},
        "10": {"ade": 7.0},
        "11": {"ade": 9.0},
        "20": {"ade": 100.0},
        "21": {"ade": 120.0},
    }

    selected = select_representative_windows(trajectory_ids, labels, metrics)

    assert selected["longitudinal"]["trajectory_id"] == 1
    assert selected["longitudinal"]["window_index"] == 1
    assert selected["turning"]["trajectory_id"] == 10
    assert selected["turning"]["window_index"] == 6
    assert selected["weaving"]["trajectory_id"] == 20
    assert selected["weaving"]["window_index"] == 10
