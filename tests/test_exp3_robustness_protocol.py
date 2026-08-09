import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.exp3_robustness import robustness_analysis
from experiments.exp3_robustness.robustness_analysis import (
    add_noise_to_data,
    aggregate_model_seed_results,
    adjust_input_length,
    apply_random_missing,
    evaluate_model_robustness,
)


def test_short_context_keeps_forecast_origin_fixed():
    x = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    y = torch.zeros(1, 3, 1)

    shortened, unchanged_target = adjust_input_length(x, y, 3)

    assert torch.equal(shortened, x[:, -3:, :])
    assert unchanged_target is y


def test_perturbations_are_reproducible_for_shared_seed():
    x = torch.ones(2, 4, 3)
    first_noise = add_noise_to_data(
        x, 0.1, generator=torch.Generator().manual_seed(1001)
    )
    second_noise = add_noise_to_data(
        x, 0.1, generator=torch.Generator().manual_seed(1001)
    )
    first_missing = apply_random_missing(
        x, 0.2, generator=torch.Generator().manual_seed(1002)
    )
    second_missing = apply_random_missing(
        x, 0.2, generator=torch.Generator().manual_seed(1002)
    )

    assert torch.equal(first_noise, second_noise)
    assert torch.equal(first_missing, second_missing)


def test_channelwise_noise_scale_is_applied():
    x = torch.zeros(1, 3, 2)
    generator = torch.Generator().manual_seed(1001)
    standard_draw = torch.randn(x.shape, generator=generator)
    expected = standard_draw * torch.tensor([0.1, 0.2]).view(1, 1, -1) * 2.0

    actual = add_noise_to_data(
        x,
        2.0,
        generator=torch.Generator().manual_seed(1001),
        noise_std_scaled=[0.1, 0.2],
    )

    assert torch.allclose(actual, expected)


def test_physical_robustness_metrics_close_at_zero_error(monkeypatch):
    x = torch.zeros(4, 8, 6)
    y = torch.zeros(4, 4, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    def exact_prediction(**kwargs):
        return kwargs["y_true_scaled"]

    monkeypatch.setattr(
        robustness_analysis, "unified_predict_by_eval_protocol", exact_prediction
    )
    metrics = evaluate_model_robustness(
        model=torch.nn.Identity(),
        test_loader=loader,
        device=torch.device("cpu"),
        scaler_mean=torch.tensor([7_000_000.0, 0.0, 0.0]),
        scaler_scale=torch.ones(3),
        test_type="noise",
        test_param=0.0,
        trajectory_ids=[10, 10, 11, 11],
    )

    assert metrics["scaled_mse"] == 0.0
    assert metrics["rmse_cart_m"] == 0.0
    assert metrics["ade_m"] == 0.0
    assert metrics["fde_m"] == 0.0
    assert metrics["trajectory_count"] == 2


def test_missing_data_uses_causal_last_observation_carry_forward():
    x = torch.arange(1, 31, dtype=torch.float32).reshape(1, 5, 6)

    filled = apply_random_missing(
        x,
        0.8,
        generator=torch.Generator().manual_seed(1002),
    )

    assert torch.equal(filled[:, 0, :], x[:, 0, :])
    assert torch.isfinite(filled).all()
    for index in range(1, x.size(1)):
        is_original = torch.equal(filled[:, index, :], x[:, index, :])
        is_carried = torch.equal(filled[:, index, :], filled[:, index - 1, :])
        assert is_original or is_carried


def test_model_seed_aggregation_pools_perturbation_replicates():
    seed_results = [
        (
            42,
            {
                "replicates": [
                    {
                        "seed": 1001,
                        "scaled_mse": 1.0,
                        "rmse_cart_m": 2.0,
                        "ade_m": 3.0,
                        "fde_m": 4.0,
                        "trajectory_count": 102,
                    }
                ]
            },
        ),
        (
            123,
            {
                "replicates": [
                    {
                        "seed": 1001,
                        "scaled_mse": 3.0,
                        "rmse_cart_m": 4.0,
                        "ade_m": 5.0,
                        "fde_m": 6.0,
                        "trajectory_count": 102,
                    }
                ]
            },
        ),
    ]

    result = aggregate_model_seed_results(seed_results)

    assert result["model_seeds"] == [42, 123]
    assert result["replicate_count"] == 2
    assert result["ade_m"] == 4.0
    assert result["trajectory_count"] == 102
