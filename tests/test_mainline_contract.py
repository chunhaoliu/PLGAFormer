import json
from pathlib import Path

import pytest

import data_generation.data_paths as data_paths
from data_generation.data_paths import get_dataset_npz_path
import models.model_factory as model_factory
from models.model_factory import create_registered_model
import utils.mainline_contract as mainline_contract
from utils.mainline_contract import (
    ACTIVE_CONFIG_PATH,
    ACTIVE_DATASET_RELATIVE_PATH,
    ACTIVE_MODEL_KEY,
    ACTIVE_PLGAFORMER_FLAGS,
    ACTIVE_TRAINABLE_MODEL_KEYS,
    PROJECT_ROOT,
    load_mainline_config,
)


EXPECTED_FLAGS = {
    "use_sparse_attention": False,
    "use_physics_corrector": False,
    "use_multi_head_output": True,
    "use_prior_fusion": True,
    "use_channel_residual": False,
    "prior_type": "rotating_3dof",
    "prior_blend_mode": "adaptive",
}


def test_active_mainline_contract_matches_formal_v3():
    payload = load_mainline_config()

    assert ACTIVE_CONFIG_PATH.name == "formal_v3.json"
    assert payload["dataset"]["protocol"] == "hgv_multiregime_state_v2_1"
    assert Path(payload["dataset"]["relative_path"]) == ACTIVE_DATASET_RELATIVE_PATH
    assert (
        payload["dataset"]["sha256"]
        == "526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7"
    )
    assert ACTIVE_MODEL_KEY == "full"
    assert dict(ACTIVE_PLGAFORMER_FLAGS) == EXPECTED_FLAGS
    assert ACTIVE_TRAINABLE_MODEL_KEYS == (
        "baseline",
        "full",
        "dlinear",
        "patchtst",
        "itransformer",
    )


def test_active_plgaformer_flags_are_immutable():
    with pytest.raises(TypeError):
        ACTIVE_PLGAFORMER_FLAGS["use_prior_fusion"] = False


def test_default_dataset_path_uses_active_mainline_contract(monkeypatch):
    monkeypatch.delenv("HGV_DATASET_PATH", raising=False)
    monkeypatch.delenv("HGV_PROCESSED_DIR", raising=False)

    assert (
        getattr(data_paths, "ACTIVE_DATASET_RELATIVE_PATH", None)
        is ACTIVE_DATASET_RELATIVE_PATH
    )
    assert get_dataset_npz_path(PROJECT_ROOT) == (
        PROJECT_ROOT / ACTIVE_DATASET_RELATIVE_PATH
    )


def test_processed_data_dir_override_selects_active_dataset(tmp_path, monkeypatch):
    processed_dir = tmp_path / "diagnostic_processed"
    monkeypatch.delenv("HGV_DATASET_PATH", raising=False)
    monkeypatch.setenv("HGV_PROCESSED_DIR", str(processed_dir))

    assert get_dataset_npz_path(PROJECT_ROOT) == (
        processed_dir.resolve() / ACTIVE_DATASET_RELATIVE_PATH.name
    )


def test_dataset_path_override_takes_precedence(tmp_path, monkeypatch):
    dataset_path = tmp_path / "candidate_dataset.npz"
    processed_dir = tmp_path / "diagnostic_processed"
    monkeypatch.setenv("HGV_DATASET_PATH", str(dataset_path))
    monkeypatch.setenv("HGV_PROCESSED_DIR", str(processed_dir))

    assert get_dataset_npz_path(PROJECT_ROOT) == dataset_path.resolve()


def test_registered_plgaformer_uses_active_mainline_flags():
    assert (
        getattr(model_factory, "ACTIVE_PLGAFORMER_FLAGS", None)
        is ACTIVE_PLGAFORMER_FLAGS
    )
    model = create_registered_model("plgaformer", input_dim=6, device="cpu")

    for attribute, expected in ACTIVE_PLGAFORMER_FLAGS.items():
        assert getattr(model, attribute) == expected
    assert model.use_adaptive_fusion is True


def test_registered_plgaformer_applies_explicit_overrides_last():
    model = create_registered_model(
        "plgaformer",
        input_dim=6,
        device="cpu",
        plgaformer_kwargs={"use_prior_fusion": False},
    )

    assert model.use_prior_fusion is False


def test_existing_final_model_helpers_reference_the_mainline_contract():
    from utils.final_plgaformer import FINAL_MODEL_KEY, final_plgaformer_kwargs
    from utils.formal_evidence import FINAL_MODEL_FLAGS

    kwargs = final_plgaformer_kwargs()
    assert FINAL_MODEL_FLAGS is ACTIVE_PLGAFORMER_FLAGS
    assert FINAL_MODEL_KEY == ACTIVE_MODEL_KEY
    assert kwargs == EXPECTED_FLAGS
    assert kwargs is not ACTIVE_PLGAFORMER_FLAGS

    kwargs["use_prior_fusion"] = False
    assert final_plgaformer_kwargs() == EXPECTED_FLAGS
    assert dict(ACTIVE_PLGAFORMER_FLAGS) == EXPECTED_FLAGS


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "message"),
    [
        (
            ("dataset", "relative_path"),
            "data_generation/data/processed/other_dataset.npz",
            "dataset relative path",
        ),
        (
            ("main_results", "trainable_model_keys"),
            ["full", "baseline", "dlinear", "patchtst", "itransformer"],
            "trainable model keys",
        ),
    ],
)
def test_load_mainline_config_rejects_contract_drift(
    tmp_path, monkeypatch, field_path, invalid_value, message
):
    payload = json.loads(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload[field_path[0]][field_path[1]] = invalid_value
    drifted_config = tmp_path / "formal_v3.json"
    drifted_config.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(mainline_contract, "ACTIVE_CONFIG_PATH", drifted_config)

    with pytest.raises(ValueError, match=message):
        load_mainline_config()
