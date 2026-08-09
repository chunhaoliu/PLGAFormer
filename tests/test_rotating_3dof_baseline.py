import numpy as np
import torch

from models.model_factory import create_registered_model
from models.plgaformer import spherical_to_cartesian_torch


def test_rotating_earth_baseline_rolls_out_finite_physical_state():
    state = torch.tensor(
        [6_458_000.0, 0.1, 0.2, 6_000.0, -0.01, 1.0], dtype=torch.float32
    )
    source = state.view(1, 1, 6).repeat(2, 32, 1)
    scaler_kwargs = {
        "input_scaler_mean": np.zeros(6, dtype=np.float32),
        "input_scaler_scale": np.ones(6, dtype=np.float32),
        "output_scaler_mean": np.zeros(3, dtype=np.float32),
        "output_scaler_scale": np.ones(3, dtype=np.float32),
        "sampling_interval_s": 1.0,
    }
    model = create_registered_model(
        "rotating_3dof", input_dim=6, device="cpu", plgaformer_kwargs=scaler_kwargs
    )

    prediction = model(source, target_length=16)

    assert prediction.shape == (2, 16, 3)
    assert torch.isfinite(prediction).all()
    assert not torch.equal(prediction[:, 0], prediction[:, -1])


def test_segmented_rk4_stays_close_to_one_second_rk4():
    state = torch.tensor(
        [6_438_000.0, 0.1, 0.2, 6_000.0, -0.01, 1.0], dtype=torch.float32
    )
    source = state.view(1, 1, 6).repeat(1, 32, 1)
    scaler_kwargs = {
        "input_scaler_mean": np.zeros(6, dtype=np.float32),
        "input_scaler_scale": np.ones(6, dtype=np.float32),
        "output_scaler_mean": np.zeros(3, dtype=np.float32),
        "output_scaler_scale": np.ones(3, dtype=np.float32),
        "sampling_interval_s": 1.0,
    }
    reference = create_registered_model(
        "rotating_3dof",
        input_dim=6,
        device="cpu",
        plgaformer_kwargs={**scaler_kwargs, "integration_stride": 1},
    )(source, target_length=64)
    segmented = create_registered_model(
        "rotating_3dof",
        input_dim=6,
        device="cpu",
        plgaformer_kwargs={**scaler_kwargs, "integration_stride": 4},
    )(source, target_length=64)

    error_m = torch.linalg.vector_norm(
        spherical_to_cartesian_torch(segmented) - spherical_to_cartesian_torch(reference),
        dim=-1,
    )
    assert float(error_m.max()) < 100.0
