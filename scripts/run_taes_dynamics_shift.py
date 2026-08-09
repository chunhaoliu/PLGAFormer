"""Compatibility alias for the canonical robustness evaluator.

The implementation lives in ``experiments.generalization_robustness.dynamics_shift``.
The historical script name is retained so tests and older commands continue
to resolve the same module object.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.generalization_robustness import dynamics_shift as _canonical

_sys.modules[__name__] = _canonical


if __name__ == "__main__":
    _canonical.main()
