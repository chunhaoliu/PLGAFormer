# Overall Prediction Performance

Paper-level study for the matched Main Results comparison. The implementation
remains under `experiments/exp1_sota/`; this directory is a logical registry
entry and does not duplicate code or formal records.

- CLI: `python run.py formal main`
- Bundle: `main_run_set_manifest.json`
- Inputs: frozen `hgv_multiregime_state_v2_1`, complete test trajectories,
  and the formal-v3 model/seed matrix
- Outputs: Main Results records and a validated Main bundle
- Paper mapping: Overall Performance

Missing model/seed units are blockers. Pilot, quick, partial, and historical
protocol outputs are not fallback evidence.
