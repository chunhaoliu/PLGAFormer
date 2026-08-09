from __future__ import annotations

import numpy as np

from scripts.run_taes_dynamics_shift import central_window_rows


def test_central_window_rows_returns_one_ordered_row_per_trajectory():
    rows, ids = central_window_rows(np.asarray([9, 9, 2, 2, 2, 7]))

    assert ids.tolist() == [2, 7, 9]
    assert rows.tolist() == [3, 5, 1]
