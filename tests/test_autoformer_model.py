import pytest


def test_autoformer_is_inactive_and_active_registry_remains_exact():
    from models import create_registered_model, get_supported_model_types

    assert get_supported_model_types() == (
        "transformer",
        "kinematic",
        "rotating_3dof",
        "dlinear",
        "plgaformer",
        "patchtst",
        "itransformer",
    )
    with pytest.raises(
        ValueError,
        match=r"^Unsupported active model type: autoformer; supported=\(",
    ):
        create_registered_model("autoformer", input_dim=6, device="cpu")
