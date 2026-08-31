from __future__ import annotations

import numpy as np

from scripts import generate_confirmatory_holdout as confirmatory


def test_confirmatory_contract_is_exact_and_nonoverlapping() -> None:
    assert confirmatory.TRAJECTORY_COUNT == 360
    assert confirmatory.TRAJECTORY_ID_START == 1800
    assert confirmatory.GENERATION_SEED == 20260831
    assert confirmatory.POINTS == 1000
    assert confirmatory.SEQ_LEN == 256
    assert confirmatory.PRED_LEN == 256
    assert confirmatory.EVAL_WINDOW_STRIDE == 5
    assert set(range(1800)).isdisjoint(range(1800, 2160))


def test_confirmatory_design_has_sixty_rows_per_joint_stratum() -> None:
    rows = confirmatory.confirmatory_design_rows()
    assert len(rows) == 360
    counts: dict[tuple[str, str], int] = {}
    for vertical, maneuver, sample in rows:
        counts[(vertical, maneuver)] = counts.get((vertical, maneuver), 0) + 1
        assert np.asarray(sample).shape == (confirmatory.DESIGN_DIMENSION,)
        assert np.all((sample >= 0.0) & (sample <= 1.0))
    assert len(counts) == 6
    assert set(counts.values()) == {60}


def test_confirmatory_window_count_is_preregistered() -> None:
    windows_per_trajectory = (
        (confirmatory.POINTS - confirmatory.SEQ_LEN - confirmatory.PRED_LEN)
        // confirmatory.EVAL_WINDOW_STRIDE
    ) + 1
    assert windows_per_trajectory == 98
    assert confirmatory.TRAJECTORY_COUNT * windows_per_trajectory == 35_280


def test_generation_refuses_existing_output(tmp_path) -> None:
    output = tmp_path / "existing.npz"
    output.touch()
    try:
        confirmatory.main(["--output", str(output)])
    except FileExistsError as exc:
        assert "non-overwriting" in str(exc)
    else:
        raise AssertionError("Existing confirmatory output must never be replaced.")
