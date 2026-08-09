# Matched Ablation implementation boundary

This directory contains ablation implementations and multiple historical result
generations. The current paper-facing study is the formal-v3
`phase4_final_mechanism_controls` matrix.

## Current formal entrypoint

Run from the repository root and specify the intended model and seed units:

```powershell
python run.py formal status --json
python run.py formal ablation --phase phase4_final_mechanism_controls --models MODEL_KEYS --seeds SEEDS
```

Current formal-v3 records may write only under:

`results/formal_v3/hgv_multiregime_state_v2_1/final/`

The minimum configured paper rows are:

- matched Transformer baseline, reused from Main Results;
- spherical prior with adaptive fusion;
- rotating-Earth prior with a fixed horizon schedule;
- full disagreement-aware PLGAFormer, reused from Main Results.

Any additional gate-input removal claimed by the manuscript must be explicitly
added to the formal contract and completed for all required seeds before it is
reported.

## Existing implementation

- `ablation_study.py` is the shared ablation engine.
- `scripts/run_formal_ablation_unit.py` is the resumable formal runner and
  currently imports shared implementation from `ablation_study.py`.

The future status and decomposition of these files are intentionally deferred
to the project-structure discussion. Do not move or delete
`ablation_study.py` before its import dependencies are separated and checked.

## Evidence boundary

`formal_v2/`, earlier phase outputs, candidate-search results, and the rejected
`pit_aligned_radar_v1` protocol are provenance only. They must not be appended to
or aggregated with formal-v3 evidence.
