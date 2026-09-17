from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_taes_altitude_skip import (
    plgaformer_leads_altitude,
    post_lock_extremum,
    steepest_post_lock_zoom,
)


def test_post_lock_extremum_ignores_lock_interval():
    time = np.arange(256, dtype=np.float64)
    altitude = 8.0 * np.exp(-0.5 * ((time - 20.0) / 6.0) ** 2)
    altitude += 2.0 * np.exp(-0.5 * ((time - 180.0) / 10.0) ** 2)

    assert post_lock_extremum(altitude) == 180


def test_steepest_zoom_is_post_lock_and_tracks_largest_span():
    altitude = np.linspace(60.0, 54.0, 256)
    altitude[90:147] = np.linspace(58.0, 50.0, 57)

    start, end, span = steepest_post_lock_zoom(altitude)

    assert start >= 64
    assert end - start == 57
    assert start == 90
    assert span == pytest.approx(8.0)


def test_zoom_altitude_gate_requires_plgaformer_first():
    assert plgaformer_leads_altitude(
        {"dlinear": 2.0, "baseline": 1.5, "itransformer": 1.1, "full": 0.4}
    )
    assert not plgaformer_leads_altitude(
        {"dlinear": 2.0, "baseline": 1.5, "itransformer": 0.3, "full": 0.4}
    )
