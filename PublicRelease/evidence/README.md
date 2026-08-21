# Public Evidence Snapshot

This directory contains compact, machine-readable tables derived from the
frozen HGV experiment bundles. It intentionally excludes generated datasets,
per-run trajectory records, checkpoints, training histories, and local paths.

The evidence boundary is simulation only: 1,800 complete trajectories under
the `hgv_multiregime_state_v2_1` protocol, split by source trajectory into
1,260/180/360 train/validation/test sets. Learned methods use seeds 42, 123,
and 456. The independent statistical unit is the held-out source trajectory.

Files:

- `main_results.csv`: multi-horizon ADE, FDE, and Cartesian RMSE.
- `maneuver_results_256s.csv`: longitudinal, turning, and weaving results.
- `strongest_comparator.csv`: paired PLGAFormer comparisons with Holm control.
- `ablation_256s.csv`: matched mechanism controls at 256 s.
- `adaptive_gate_paired.csv`: full versus fixed-schedule paired evidence.
- `robustness.csv`: nominal and frozen dynamics-shift results.
- `efficiency.csv`: parameters, FLOPs, memory, latency, and throughput.
- `evidence_manifest.json`: protocol, bundle identities, and file hashes.

Regenerate this snapshot only in a complete local experiment workspace:

```bash
python scripts/generate_release_evidence.py
```

The command verifies formal configuration, dataset, bundle, robustness, and
efficiency hashes before writing output. It does not train or evaluate a model.
