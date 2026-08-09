"""Compatibility alias for the canonical Overall Prediction engine.

The implementation lives in ``experiments.overall_prediction.main_results``.
The historical module name is registered as an alias so imports and
monkeypatches through the old path operate on the same module object.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.overall_prediction import main_results as _canonical

# Preserve the old import path while ensuring all attributes and patches are
# applied to the canonical module namespace.
_sys.modules[__name__] = _canonical


if __name__ == "__main__":
    _canonical.main()
