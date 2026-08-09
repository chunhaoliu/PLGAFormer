# Main Results implementation boundary

This directory contains the Main Results implementation and multiple historical
result generations. It is not a standalone paper-facing workflow.

## Current formal entrypoint

Run from the repository root and specify the intended model and seed units:

```powershell
python run.py formal status --json
python run.py formal main --models MODEL_KEYS --seeds SEEDS
```

Current formal-v3 records may write only under:

`results/formal_v3/hgv_multiregime_state_v2_1/final/`

The formal coordinator validates dataset identity, configuration hash,
checkpoint identity, required physical metrics, test-evaluation policy, and the
complete model/seed matrix before paper aggregation.

## Existing implementation

- `SOTA_comparison.py` is the shared training and evaluation engine.
- `run_sota.py` is an existing numbered-experiment wrapper.
- `visualize_results.py` reads diagnostic prediction caches.
- `scripts/run_formal_sota_unit.py` is the resumable formal runner and currently
  imports shared implementation from `SOTA_comparison.py`.

The future status and decomposition of these files are intentionally deferred
to the project-structure discussion. Do not move or delete
`SOTA_comparison.py` before its import dependencies are separated and checked.

## Evidence boundary

`results/formal/`, `results/formal_v2/`, `partial_runs.json`, and timestamped
trajectory caches are historical or diagnostic evidence. They must not be
merged with formal-v3 results or used directly in manuscript tables.
