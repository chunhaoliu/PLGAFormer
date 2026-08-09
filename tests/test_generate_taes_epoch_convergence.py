import numpy as np

from scripts.generate_taes_epoch_convergence import (
    MAX_EPOCHS,
    MODEL_ORDER,
    SEEDS,
    aggregate_histories,
)


def test_aggregate_histories_uses_sample_standard_deviation():
    histories = {}
    for model_index, model_type in enumerate(MODEL_ORDER):
        histories[model_type] = {}
        for seed_index, seed in enumerate(SEEDS):
            base = float(model_index + seed_index + 1)
            histories[model_type][str(seed)] = {
                "train": np.full(MAX_EPOCHS - seed_index, base),
                "validation": np.full(MAX_EPOCHS - seed_index, 2.0 * base),
            }

    summary = aggregate_histories(histories)
    np.testing.assert_allclose(summary["plgaformer"]["train_mean"], 2.0)
    np.testing.assert_allclose(summary["plgaformer"]["train_std"], 1.0)
    np.testing.assert_allclose(summary["plgaformer"]["validation_mean"], 4.0)
    np.testing.assert_allclose(summary["plgaformer"]["validation_std"], 2.0)
    assert len(summary["plgaformer"]["train_mean"]) == MAX_EPOCHS - 2
    assert summary["plgaformer"]["common_epochs"] == MAX_EPOCHS - 2


def test_aggregate_histories_requires_all_formal_seeds():
    histories = {
        model_type: {
            "42": {
                "train": np.ones(MAX_EPOCHS),
                "validation": np.ones(MAX_EPOCHS),
            }
        }
        for model_type in MODEL_ORDER
    }
    try:
        aggregate_histories(histories)
    except RuntimeError as exc:
        assert "three formal seeds" in str(exc)
    else:
        raise AssertionError("Expected an incomplete-seed validation error.")
