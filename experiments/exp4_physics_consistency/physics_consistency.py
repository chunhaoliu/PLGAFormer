"""Compatibility shim for the canonical mechanism physics evaluator.

The implementation moved to ``experiments.mechanism_analysis`` so the
paper-level study owns its evaluator. Legacy scripts, tests, and summaries
still import this numbered path, so the public module surface is re-exported
here and the protocol constants remain literal for the legacy text parser.
"""

from __future__ import annotations

from experiments.mechanism_analysis.physics_consistency import *  # noqa: F401,F403
from experiments.mechanism_analysis.physics_consistency import main

# Keep the legacy protocol-summary parser compatible with the previous source
# file. These values mirror the active formal-v3 defaults.
EVAL_PROTOCOL = "strict_autoregressive"
EVAL_AR_SEED_MODE = "zero"
TRAIN_SUPERVISION_PROTOCOL = "shifted_next_step"
OPTIMIZER_PROFILE = "enhanced"
STRICT_REPRO_MODE = TRAIN_CONFIG.get("strict_repro_mode", False)
