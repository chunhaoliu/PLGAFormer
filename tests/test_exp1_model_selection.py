import pytest

from experiments.exp1_sota.SOTA_comparison import _selected_comparison_models


def test_model_selection_normalizes_underscore_model_types(monkeypatch):
    monkeypatch.setenv("HGV_SOTA_MODELS", "rotating_3dof,itransformer")

    selected = _selected_comparison_models()
    selected_types = {config["model_type"] for config in selected.values()}

    assert selected_types == {"rotating_3dof", "itransformer"}


def test_model_selection_rejects_inactive_af_ciln_even_with_active_request(monkeypatch):
    monkeypatch.setenv("HGV_SOTA_MODELS", "rotating_3dof,af_ciln")

    with pytest.raises(ValueError, match="af_ciln"):
        _selected_comparison_models()
