import json
from pathlib import Path
import subprocess
import sys

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
    PAPER_INFERENCE_POLICY,
    PROJECT_ROOT,
    load_mainline_config,
)

ACTIVE_MODEL_TYPES = (
    "transformer",
    "kinematic",
    "rotating_3dof",
    "dlinear",
    "plgaformer",
    "patchtst",
    "itransformer",
)
INACTIVE_MODEL_TYPES = (
    "baseline",
    "pit",
    "af_ciln",
    "autoformer",
    "informer",
    "fedformer",
    "timesnet",
    "kalman",
    "legacy_sota",
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
EXPECTED_PAPER_POLICY = {
    "name": "short_horizon_physics_prior_lock_tau450_power2.5",
    "lock_steps": 64,
    "time_constant_s": 450.0,
    "decay_power": 2.5,
}
EXPECTED_PAPER_KWARGS = {
    **EXPECTED_FLAGS,
    "physics_prior_lock_steps": 64,
    "physics_prior_time_constant_s": 450.0,
    "physics_prior_decay_power": 2.5,
}


def test_active_model_registry_is_exact():
    assert model_factory.get_supported_model_types() == ACTIVE_MODEL_TYPES


@pytest.mark.parametrize("model_type", INACTIVE_MODEL_TYPES)
def test_inactive_model_types_are_not_constructible(model_type):
    with pytest.raises(
        ValueError,
        match=(
            "^Unsupported active model type: "
            + model_type
            + r"; supported=\("
        ),
    ):
        create_registered_model(model_type, input_dim=6, device="cpu")


def test_legacy_factory_names_import_and_active_sota_delegates():
    from models import (
        create_baseline_model,
        create_pit_model,
        create_sota_model,
    )

    assert callable(create_baseline_model)
    assert callable(create_pit_model)
    model = create_sota_model("transformer", input_dim=6, device="cpu")
    assert model.__class__.__name__ == "StandardTransformer"
    assert model_factory.get_supported_model_types() == ACTIVE_MODEL_TYPES


def test_legacy_sota_shim_constructs_exact_active_plgaformer(monkeypatch):
    import models

    captured = {}

    def capture_registered_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(models, "create_registered_model", capture_registered_model)

    models.create_sota_model("plgaformer", input_dim=6, device="cpu")

    assert captured == {
        "model_type": "plgaformer",
        "input_dim": 6,
        "device": "cpu",
        "plgaformer_kwargs": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_prior_fusion", False),
        ("dropout", 0.2),
        ("num_encoder_layers", 1),
        ("d_model", 64),
    ],
)
def test_legacy_sota_shim_rejects_plgaformer_architecture_overrides(field, value):
    from models import create_sota_model

    with pytest.raises(TypeError, match="PLGAFormer compatibility shim"):
        create_sota_model("plgaformer", input_dim=6, device="cpu", **{field: value})


def test_legacy_sota_shim_maps_source_root_to_tslib_root(monkeypatch):
    import models

    captured = {}

    def capture_registered_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(models, "create_registered_model", capture_registered_model)

    models.create_sota_model(
        "patchtst", input_dim=6, device="cpu", source_root="official-tslib"
    )

    assert captured["plgaformer_kwargs"] == {
        "source": "tslib",
        "tslib_root": "official-tslib",
    }


def test_legacy_sota_shim_rejects_conflicting_source_roots():
    from models import create_sota_model

    with pytest.raises(ValueError, match="source_root.*tslib_root"):
        create_sota_model(
            "itransformer",
            source_root="first",
            tslib_root="second",
        )


def test_legacy_sota_shim_rejects_unknown_kwargs():
    from models import create_sota_model

    with pytest.raises(TypeError, match="Unsupported create_sota_model keyword"):
        create_sota_model("dlinear", arbitrary_override=True)


@pytest.mark.parametrize("model_type", INACTIVE_MODEL_TYPES)
def test_legacy_sota_shim_rejects_inactive_model_types(model_type):
    from models import create_sota_model

    with pytest.raises(
        ValueError,
        match=(
            "^Unsupported active model type: "
            + model_type
            + r"; supported=\("
        ),
    ):
        create_sota_model(model_type, input_dim=6, device="cpu")


def test_legacy_pit_shim_is_explicitly_unsupported():
    from models import create_pit_model

    with pytest.raises(
        ValueError,
        match=r"^Unsupported active model type: pit; supported=\(",
    ):
        create_pit_model(input_dim=6, device="cpu")


def test_legacy_factory_imports_do_not_load_inactive_modules():
    code = """
import sys
from models import create_baseline_model, create_pit_model, create_sota_model
for construct in (
    lambda: create_sota_model('autoformer'),
    lambda: create_pit_model(),
):
    try:
        construct()
    except ValueError:
        pass
    else:
        raise AssertionError('inactive compatibility shim unexpectedly constructed a model')
inactive = {'models.sota_models', 'models.PIT', 'models.external_baselines'}
loaded = inactive.intersection(sys.modules)
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


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
    assert dict(PAPER_INFERENCE_POLICY) == EXPECTED_PAPER_POLICY
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
    with pytest.raises(TypeError):
        PAPER_INFERENCE_POLICY["lock_steps"] = 0


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
    assert model.physics_prior_lock_steps == 0
    assert model.physics_prior_time_constant_s == 450.0
    assert model.physics_prior_decay_power == 2.0


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
    assert kwargs == EXPECTED_PAPER_KWARGS
    assert kwargs is not ACTIVE_PLGAFORMER_FLAGS

    kwargs["use_prior_fusion"] = False
    assert final_plgaformer_kwargs() == EXPECTED_PAPER_KWARGS
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
