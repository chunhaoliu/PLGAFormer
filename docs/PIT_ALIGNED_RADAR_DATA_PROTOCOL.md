# PIT-Aligned Radar Tracking Data Protocol

## Status

`pit_aligned_radar_v1` is a rejected diagnostic protocol. Its array, split,
tracking-causality, and hash checks passed, but a later line-of-sight audit
showed that only 12.96% of ideal measurements have positive elevation and
every trajectory starts below the fixed radar's horizon. The artifact is
retained for provenance only and must not be used for Main Results, ablations,
or continuous-radar claims. The active replacement is
`hgv_multiregime_state_v2_1`; see
`DATASET_REFERENCE_GAP_AUDIT_20260730.md`.

## Reference-Aligned Elements

- 1,000 complete simulated HGV trajectories in the planned formal artifact.
- 1,000 observations per trajectory at a 2 s radar update interval.
- Initial flight-path angle sampled over -1--1 degrees.
- Radar site at longitude 12 degrees, latitude 0.5 degrees, and altitude 1 km.
- Independent nominal measurement errors: 100 m range, 0.5 mrad azimuth, and
  0.5 mrad elevation.

These settings are inspired by the local Physics-Informed Transformer paper,
but the present protocol does not copy its ambiguous point/window split.

The reference reports an initial-speed range of 5,000--7,000 m/s. In the local
point-mass HGV simulator, entries near 5,000 m/s decelerate below the intended HGV glide
envelope before the end of a 1,998 s record. The candidate protocol therefore
samples 6,300--7,000 m/s and accepts only complete records that remain at
30--80 km altitude and at or above 3,000 m/s. This is an explicit
simulator-feasibility decision, not a claim that the reference used this
narrower range.

## Project-Specific Improvements

1. The six-state HGV representation retains heading:
   `[r, longitude, latitude, speed, flight-path angle, heading]`.
2. Clean simulator truth, noisy radar measurement, and filtered tracking state
   are stored as distinct evidence layers.
3. The ECEF constant-velocity Kalman filter is strictly causal. It processes
   only the current and previous radar measurements.
4. Complete trajectory IDs are stratified into train, validation, and test
   before window construction.
5. The model input is the causally filtered radar track; the future target is
   the clean simulator truth.
6. A 32-step (64 s) tracking burn-in is excluded from forecast origins.
7. Training uses 16 uniformly distributed origins per complete trajectory.
   Validation and test retain a denser five-step origin stride.
8. Clean truth is integrated with adaptive DOP853 (`rtol=1e-8`,
   `atol=1e-10`) and stored only at the 2 s observation timestamps. A 2 s
   versus 4 s maximum-step convergence check over all three maneuvers changed
   ECEF position by less than 0.6 m while the 4 s setting reduced generation
   time; thus numerical integration and radar sampling remain separate
   contracts.

## Time Interface

- Observation interval: 2 s.
- Complete trajectory duration: 1,998 s.
- Input: 256 steps = 512 s.
- Direct forecast: 256 steps = 512 s.
- Reported physical horizons:
  - 32 steps = 64 s;
  - 64 steps = 128 s;
  - 128 steps = 256 s;
  - 256 steps = 512 s.

The paper must label horizons in seconds. Step counts may appear in the
experimental setup for reproducibility.

## Evidence Boundary

- The nine-trajectory output is a pilot only.
- The verified full artifact is the input to new formal experiments; its
  existence alone is not model-performance evidence.
- Results from `trajectory_level_v1` and `pit_aligned_radar_v1` must never be
  combined in one mean, ranking, or significance test.
- Simulation-only data must not be described as flight-test or operational
  radar validation.

## Archived diagnostic commands

These commands reconstruct the rejected artifact for audit purposes only. They
do not authorize training:

```powershell
python scripts/generate_pit_radar_dataset.py --force
```

Full diagnostic reconstruction:

```powershell
python scripts/generate_pit_radar_dataset.py --formal
```

Do not pass the resulting artifact to a formal experiment runner. The retained
`formal_v3/pit_aligned_radar_v1/` directory name is provenance only.
