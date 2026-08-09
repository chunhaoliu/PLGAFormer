# Top-Journal Protocol Refactor Plan

Goal: converge HGVTP-PLGAformer to a cleaner long-trajectory forecasting protocol aligned with Autoformer/FEDformer-style experiment organization and AF-CILN-style HGV task details, without adding extra model innovation points beyond the existing three.

## Scope

- Data must use `trajectory_level_v1`: split complete trajectories first, then generate windows within each split.
- Experiments must fail loudly on legacy random-window data unless explicitly run in compatibility mode.
- Data quality validation must check split leakage, array shapes, scaler leakage, physical ranges, continuity, window metadata, and maneuver balance.
- Evaluation must support both window-level metrics and trajectory-level aggregation by `trajectory_id`.
- Project layout should continue moving toward `data_provider/`, `experiments/`, `models/`, `utils/`, `scripts/`, and `run.py`.
- Old generated cache and temporary smoke data can be removed after path safety checks.

## Reference Principles

- Autoformer/FEDformer: central data provider, explicit `seq_len/label_len/pred_len`, train-fitted scaler, deterministic setting strings, train/vali/test flow, no hidden random split across evaluation windows.
- AF-CILN: HGV data uses `[B, input_len, 6] -> [B, pred_len, 3]`, reports HGV task metrics and missing-data robustness, but its random sample split should not be copied for our long-trajectory setting.
- HGVTP-PLGAformer: retain only the current three innovation points and make their ablation cleaner instead of adding new model modules.

## Implementation Phases

1. Add tests for a strict data provider, dataset quality validator, and trajectory-level metric aggregation.
2. Implement `data_provider/` for loading `trajectory_level_v1` npz files with split metadata.
3. Implement dataset validation reports for generated HGV data.
4. Implement trajectory-level ADE/FDE aggregation by complete trajectory id.
5. Wire experiments to the strict provider step by step.
6. Clean generated caches and legacy smoke artifacts after verification.
7. Regenerate and validate the default processed data before running long experiments.

## Success Criteria

- Unit tests and `py_compile` pass.
- Strict loader rejects legacy random-window data.
- New generated data passes trajectory split, shape, scaler, metadata, and physical sanity checks.
- Smoke pipeline runs on `trajectory_level_v1` data.
- Experiment outputs record the dataset protocol and trajectory split diagnostics.
