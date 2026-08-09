# Overall prediction

This directory contains the paper's main overall-prediction implementation.

- Canonical engine: `experiments/overall_prediction/main_results.py`
- Formal entry point: `python run.py formal main`
- Historical output roots and result metadata remain unchanged during the source migration so existing records stay traceable.
- Dataset preparation, model definitions, and shared evaluation utilities remain in their existing project modules.

The old `experiments/exp1_sota/` path is retained only as a compatibility boundary for older scripts and imports. It is not a separate paper study.
