# Compatibility boundary

The Overall Prediction implementation now lives at `experiments/overall_prediction/main_results.py`.

This directory is retained temporarily for compatibility with legacy scripts, tests, and previously recorded output paths. `SOTA_comparison.py` is only a thin import shim; it is not an independent implementation or a separate paper experiment.

New formal runs should use `python run.py formal main`. Existing result directories are preserved until every reader, manifest, and runner has been verified against the canonical path.
