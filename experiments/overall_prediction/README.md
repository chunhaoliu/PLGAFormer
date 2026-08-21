# Overall prediction

This directory contains the paper's main overall-prediction implementation.

- Canonical engine: `experiments/overall_prediction/main_results.py`
- Formal entry point: `python run.py formal main`
- The paper-facing main comparison excludes PIT and AF-CILN. PIT remains available only for isolated analysis, and AF-CILN remains an explicit isolated option pending protocol-aligned revalidation from its official external checkout.
- Transformer, DLinear, PatchTST, and iTransformer must be constructed from the pinned MIT-licensed TSLib checkout (`--tslib-root` or `HGV_TSLIB_ROOT`); the repository-local reimplementations are compatibility paths, not paper-facing baselines.
- Historical output roots and result metadata remain unchanged during the source migration so existing records stay traceable.
- Dataset preparation, model definitions, and shared evaluation utilities remain in their existing project modules.

The old `experiments/exp1_sota/` path is retained only as a compatibility boundary for older scripts and imports. It is not a separate paper study.
