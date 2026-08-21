import pytest

from scripts.generate_taes_main_results import (
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    _format_result,
    render_main_table,
    validate_complete,
)


def test_incomplete_formal_cache_is_rejected():
    records = {
        (seed, model): {"results": {str(h): {} for h in (32, 64, 128, 256)}}
        for seed in EXPECTED_SEEDS
        for model in EXPECTED_MODELS
    }
    records.pop((456, "PLGAFormer (proposed)"))

    with pytest.raises(RuntimeError, match="incomplete"):
        validate_complete(records)


def test_table_format_marks_best_and_second_without_changing_value():
    best = _format_result(1250.0, 50.0, 0)
    second = _format_result(1500.0, 75.0, 1)
    other = _format_result(2000.0, 100.0, 2)

    assert best == r"\textbf{1.250$\pm$0.050}"
    assert second == r"\underline{1.500$\pm$0.075}"
    assert other == r"2.000$\pm$0.100"


def test_table_format_omits_fake_uncertainty_for_deterministic_methods():
    assert _format_result(1250.0, 0.0, 0, deterministic=True) == r"\textbf{1.250}"
    assert _format_result(1500.0, 0.0, 1, deterministic=True) == r"\underline{1.500}"


def test_main_table_uses_horizon_metric_rows_and_only_learned_comparators():
    metrics = ("ade", "fde", "rmse_cart_m")
    stats = {
        model: {
            horizon: {
                metric: {
                    "mean_m": float(1000 + 10 * model_index + horizon),
                    "std_m": 10.0,
                }
                for metric in metrics
            }
            for horizon in (32, 64, 128, 256)
        }
        for model_index, model in enumerate(EXPECTED_MODELS)
    }
    table = render_main_table(stats)

    assert r"\multirow{3}{*}{32 s}" in table
    assert "RMSE" in table
    assert r"Rot.\ 3-DOF" not in table
    assert "Spherical" not in table
    assert "iTransformer" in table
    assert r"\multirow{3}{*}{256 s}" in table


def test_main_plga_factory_uses_prior_without_channel_residual():
    from models.model_factory import create_registered_model

    model = create_registered_model("plgaformer", input_dim=6, device="cpu")
    assert model.use_multi_head_output is True
    assert model.use_prior_fusion is True
    assert model.use_channel_residual is False
