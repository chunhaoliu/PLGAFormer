from argparse import Namespace
from copy import deepcopy

import pytest

from utils.formal_runtime import (
    formal_runtime_defaults,
    parse_prediction_horizons,
    validate_formal_runtime,
)
from utils.mainline_contract import ACTIVE_CONFIG_PATH, load_mainline_config


def _runtime_args(frozen_config=None, **overrides):
    frozen_config = frozen_config or load_mainline_config()
    values = {
        **formal_runtime_defaults(frozen_config),
        "config": str(ACTIVE_CONFIG_PATH),
        "selection_only": False,
        "subset_ratio": 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_runtime_defaults_match_the_frozen_contract():
    defaults = formal_runtime_defaults(load_mainline_config())

    assert defaults["seeds"] == "42,123,456"
    assert defaults["epochs"] == 50
    assert defaults["batch_size"] == 128
    assert defaults["prediction_length"] == 256
    assert defaults["prediction_horizons"] == "32,64,128,256"
    assert defaults["label_len"] == 128
    assert defaults["amp"] is False


def test_prediction_horizons_are_sorted_unique_and_positive():
    assert parse_prediction_horizons("128,32,64,64") == [32, 64, 128]
    assert parse_prediction_horizons("", fallback=96) == [96]


def test_prediction_horizons_reject_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        parse_prediction_horizons("32,0,64")


def test_runtime_accepts_non_empty_seed_subset():
    config = load_mainline_config()
    defaults = formal_runtime_defaults(config)

    validate_formal_runtime(_runtime_args(config, seeds="42"), config)


def test_runtime_accepts_seed_whitespace_and_requested_order():
    config = load_mainline_config()

    validate_formal_runtime(_runtime_args(config, seeds=" 456, 42 "), config)


def test_runtime_rejects_prediction_length_drift():
    config = load_mainline_config()
    defaults = formal_runtime_defaults(config)

    with pytest.raises(ValueError, match=r"prediction_length.*256"):
        validate_formal_runtime(_runtime_args(config, prediction_length=128), config)


def test_runtime_rejects_unknown_seed():
    config = load_mainline_config()
    defaults = formal_runtime_defaults(config)

    with pytest.raises(ValueError, match=r"seeds.*42.*123.*456"):
        validate_formal_runtime(_runtime_args(config, seeds="999"), config)


@pytest.mark.parametrize("seeds", ["", "42,42", "42,bad", "42,,123"])
def test_runtime_rejects_empty_duplicate_or_malformed_seeds(seeds):
    config = load_mainline_config()
    defaults = formal_runtime_defaults(config)

    with pytest.raises(ValueError, match="seeds"):
        validate_formal_runtime(_runtime_args(config, seeds=seeds), config)


@pytest.mark.parametrize(
    "subset_ratio",
    [True, "1.0", float("nan"), float("inf"), 0.0, -0.1, 1.1],
)
def test_runtime_rejects_invalid_subset_ratio(subset_ratio):
    config = load_mainline_config()

    with pytest.raises(ValueError, match="subset_ratio"):
        validate_formal_runtime(
            _runtime_args(config, subset_ratio=subset_ratio), config
        )


def test_runtime_rejects_reduced_subset_for_final_run():
    config = load_mainline_config()

    with pytest.raises(ValueError, match=r"subset_ratio.*1\.0"):
        validate_formal_runtime(_runtime_args(config, subset_ratio=0.2), config)


def test_runtime_allows_reduced_subset_for_selection_only():
    config = load_mainline_config()

    validate_formal_runtime(
        _runtime_args(config, subset_ratio=0.2, selection_only=True), config
    )


def test_runtime_rejects_alternative_config_path(tmp_path):
    config = load_mainline_config()

    with pytest.raises(ValueError, match=r"config.*formal_v3\.json"):
        validate_formal_runtime(
            _runtime_args(config, config=str(tmp_path / "formal_v3.json")), config
        )


def test_runtime_rejects_bool_as_numeric_argument():
    config = load_mainline_config()

    with pytest.raises(ValueError, match=r"workers.*0"):
        validate_formal_runtime(_runtime_args(config, workers=False), config)


def test_runtime_normalizes_integral_scientific_real_to_float():
    config = deepcopy(load_mainline_config())
    config["training"]["gradient_clip_norm"] = 1

    defaults = formal_runtime_defaults(config)

    assert defaults["gradient_clip_norm"] == 1.0
    assert type(defaults["gradient_clip_norm"]) is float


@pytest.mark.parametrize(
    ("section", "field", "malformed"),
    [
        ("training", "epochs_max", True),
        ("training", "gradient_clip_norm", True),
        ("training", "learning_rate", float("nan")),
        ("training", "weight_decay", float("inf")),
        ("training", "amp", 0),
        ("training", "seeds", [42, True, 456]),
        ("task", "pred_len", False),
        ("task", "seq_len", True),
        ("task", "training_origins", False),
        ("task", "reporting_horizons", [32, 64, True, 256]),
        ("task", "eval_protocol", 1),
        ("training", "test_policy", 1),
    ],
)
def test_runtime_rejects_malformed_frozen_config_types(
    section, field, malformed
):
    config = deepcopy(load_mainline_config())
    config[section][field] = malformed

    with pytest.raises((TypeError, ValueError), match=field):
        formal_runtime_defaults(config)
