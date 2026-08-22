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
- `strongest_comparator.csv`: internal analytical-comparator audit; it is not the paper-facing learning-baseline ranking.
- `ablation_256s.csv`: matched mechanism controls at 256 s.
- `capacity_control_256s.csv`: three-seed learned-only PLGAFormer backbone control.
- `adaptive_gate_paired.csv`: retained diagnostic paired analysis from the earlier policy record; it is not used for the promoted paper numbers.
- `robustness.csv`: nominal and frozen dynamics-shift results.
- `efficiency.csv`: retained computational diagnostic; efficiency is not a current manuscript experiment.
- `evidence_manifest.json`: protocol, bundle identities, and file hashes.

The checked-in snapshot is bound to the promoted Main/Ablation/robustness artifacts. Before regenerating it in a complete local workspace, verify that the discovered bundle IDs and policy match `evidence_manifest.json`:

```bash
python scripts/generate_release_evidence.py
```

The command verifies formal configuration, dataset, bundle, capacity-control,
robustness, and efficiency hashes before writing output. It does not train or
evaluate a model.
