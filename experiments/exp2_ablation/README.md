# Compatibility boundary

The canonical mechanism-analysis implementation now lives at `experiments/mechanism_analysis/ablation_study.py`.

This directory is retained temporarily for compatibility with legacy imports, wrappers, visualizers, and previously recorded output paths. `ablation_study.py` is only a module alias; it is not an independent implementation or a separate paper experiment.

New formal runs should use `python run.py formal mechanism`. Existing result directories remain in place until every reader, manifest, and runner has been checked against the canonical path.
