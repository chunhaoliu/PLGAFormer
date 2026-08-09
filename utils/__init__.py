#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from .repro import set_global_seed, seed_worker, build_torch_generator
    from .train_protocol import (
        build_adamw_optimizer,
        build_warmup_cosine_scheduler,
        should_update_best,
        build_optimizer_by_profile,
        build_scheduler_by_profile,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
from .experiment_io import (
    ensure_dir,
    timestamp_str,
    get_experiment_dirs,
    save_json,
    save_run_metadata,
    save_experiment_results,
)
from .experiment_summary import (
    PROTOCOL_COLUMNS,
    UNIFIED_COLUMNS,
    collect_protocol_consistency_rows,
    collect_unified_metric_rows,
    save_unified_metrics_csv,
)
from .trajectory_protocol import (
    TRAJECTORY_LEVEL_PROTOCOL,
    LEGACY_WINDOW_PROTOCOL,
    build_windows_for_trajectories,
    dataset_protocol_name,
    dataset_uses_trajectory_level_split,
    get_split_maneuver_labels,
    split_trajectory_ids,
    validate_disjoint_trajectory_splits,
    validate_npz_trajectory_splits,
)
from .trajectory_metrics import aggregate_window_metrics_by_trajectory
from .inference_protocol import (
    DEFAULT_ONESHOT_MODELS,
    default_causal_mask,
    is_oneshot_model,
    build_decoder_context_for_eval,
    predict_by_eval_protocol,
)
from .model_provenance import (
    MODEL_PROVENANCE,
    get_model_provenance,
    build_comparison_model_provenance,
    validate_comparison_model_types,
)
from .seq2seq_protocol import (
    build_supervision_windows as build_seq2seq_supervision_windows,
    training_forward as seq2seq_training_forward,
)
from .console import ensure_utf8_console

__all__ = [
    "set_global_seed",
    "seed_worker",
    "build_torch_generator",
    "build_adamw_optimizer",
    "build_warmup_cosine_scheduler",
    "should_update_best",
    "build_optimizer_by_profile",
    "build_scheduler_by_profile",
    "ensure_dir",
    "timestamp_str",
    "get_experiment_dirs",
    "save_json",
    "save_run_metadata",
    "save_experiment_results",
    "PROTOCOL_COLUMNS",
    "UNIFIED_COLUMNS",
    "collect_protocol_consistency_rows",
    "collect_unified_metric_rows",
    "save_unified_metrics_csv",
    "TRAJECTORY_LEVEL_PROTOCOL",
    "LEGACY_WINDOW_PROTOCOL",
    "build_windows_for_trajectories",
    "dataset_protocol_name",
    "dataset_uses_trajectory_level_split",
    "get_split_maneuver_labels",
    "split_trajectory_ids",
    "validate_disjoint_trajectory_splits",
    "validate_npz_trajectory_splits",
    "aggregate_window_metrics_by_trajectory",
    "DEFAULT_ONESHOT_MODELS",
    "default_causal_mask",
    "is_oneshot_model",
    "build_decoder_context_for_eval",
    "predict_by_eval_protocol",
    "MODEL_PROVENANCE",
    "get_model_provenance",
    "build_comparison_model_provenance",
    "validate_comparison_model_types",
    "build_seq2seq_supervision_windows",
    "seq2seq_training_forward",
    "ensure_utf8_console",
]
