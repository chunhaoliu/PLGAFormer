"""Compatibility shim for the canonical efficiency benchmark.

The implementation lives in :mod:`experiments.efficiency.efficiency_experiment`.
This module remains at the historical Exp6 path because the formal runner,
legacy CLI, and protocol tests still import it.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.efficiency.efficiency_experiment import *
from experiments.efficiency.efficiency_experiment import main


if __name__ == "__main__":
    main()
