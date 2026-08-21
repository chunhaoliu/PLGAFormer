import pytest


def test_tslib_source_identity_is_pinned_in_code():
    from models.public_baselines import (
        PUBLIC_TSLIB_MODEL_TYPES,
        TSLIB_ADAPTER_DEFAULTS,
        TSLIB_FILE_SHA256,
        TSLIB_OFFICIAL_COMMIT,
        TSLIB_OFFICIAL_REPOSITORY,
    )

    assert PUBLIC_TSLIB_MODEL_TYPES == {
        "transformer",
        "dlinear",
        "patchtst",
        "itransformer",
    }
    assert TSLIB_OFFICIAL_REPOSITORY == "https://github.com/thuml/Time-Series-Library"
    assert len(TSLIB_OFFICIAL_COMMIT) == 40
    assert all(len(value) == 64 for value in TSLIB_FILE_SHA256.values())
    assert TSLIB_ADAPTER_DEFAULTS["transformer"] == {
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 1,
        "dropout": 0.1,
    }


def test_public_adapter_refuses_to_fall_back_without_checkout(monkeypatch):
    from models.public_baselines import TSLibForecastAdapter

    monkeypatch.delenv("HGV_TSLIB_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="official TSLib checkout"):
        TSLibForecastAdapter(model_type="dlinear")
