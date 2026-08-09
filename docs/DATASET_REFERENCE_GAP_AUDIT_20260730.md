# HGV dataset reference-gap audit

Date: 2026-07-30
Status: historical v2 audit, superseded by
`docs/MULTIREGIME_V21_DATA_PROTOCOL.md`.  The v2.1 data gate passed on
2026-07-31; this file is retained only to document the decision path.

## Executive decision

The existing `pit_aligned_radar_v1` artifact is numerically reproducible but is
not acceptable as the next paper-facing dataset. Its single-site radar geometry
places every trajectory below the local horizon at the initial sample, only
12.96% of all samples have positive ideal elevation, and no trajectory is
visible for even 25% of its 1,998 s record. The artifact is therefore retained
as a diagnostic snapshot only; it must not be described as continuous radar
tracking evidence.

The older `trajectory_level_v1` artifact has a different limitation. It uses
one deterministic angle-of-attack schedule and one fixed bank schedule for each
of the three maneuver labels. Initial-state variation alone does not provide
the longitudinal-mode and control-parameter diversity shown by the reference
datasets.

The replacement route is `hgv_multiregime_state_v2`. It keeps the manuscript's
three primary labels, adds an independently recorded quasi-equilibrium/skip
factor, randomizes control parameters by complete trajectory, stores the
commands and load diagnostics, splits joint strata by trajectory ID before
windowing, and makes no unsupported single-radar visibility claim.

## Reference comparison

| Evidence item | PIT (TAES 2023) | DCBNN (IEEE Access 2021) | AF-CILN (AST 2026) | Existing project data | New protocol |
|---|---|---|---|---|---|
| Point-mass dynamics | HGV equations | HGV equations | Nonrotating spherical 3-DOF | Rotating spherical 3-DOF | Rotating spherical 3-DOF |
| Longitudinal regimes | Skip glide | One common longitudinal law | 4,000 quasi-equilibrium + 4,000 skip-glide | One skip-like family only | Balanced quasi-equilibrium + skip-glide |
| Lateral taxonomy | C/S paths | zero / one turn / weaving | 0--3 lateral maneuvers | longitudinal / turning / weaving | Same three primary labels; weaving contains 2--3 reversals |
| Control diversity | Limited disclosure | Three fixed laws | Per-trajectory randomized attack/bank parameters | Fixed class-level commands | Per-trajectory randomized attack/bank parameters |
| Complete trajectories | 640 | 7,500 across three data sets | 8,000 | 1,000 | 1,200; 200 per joint stratum |
| Record | 1,000 points, 2 s | up to 2,000 s | 1,000 points, 1 s | 1,000 points, 1 s or PIT candidate at 2 s | 1,000 points, 1 s |
| Initial envelope | 62 km; 5--7 km/s; flight-path angle -1--1 deg | 55--80 km; 4--6 km/s | 50/60 km; Mach 8--12; -0.5/0/0.5 deg | PIT candidate 60--72 km and 6.3--7.0 km/s | 55--70 km and 6.0--7.0 km/s; mode-consistent flight-path angle |
| Observation claim | Noisy radar plus filtering | Simulated states | Simulated ECEF states plus masks | Invalid all-route single-radar geometry in PIT candidate | Clean simulator/post-tracking state benchmark; sensor degradation kept separate |
| Split evidence | Reported 80/10/10; unit not fully explicit | Pretrain/validation/test description | 80/10/10; unit not fully explicit | Complete-ID split | Joint-stratified complete-ID split before windows |
| Stored provenance | Partial | Partial | Control ranges disclosed | Controls stored only in older raw artifact | States, commands, profile parameters, loads, seeds, splits, and hashes |

## Quantitative diagnosis of `pit_aligned_radar_v1`

The following values were recomputed from
`data_generation/data/processed/pit_aligned_radar_dataset.npz` and the ideal
geometry, not from a plot:

- Ideal positive-elevation sample fraction: 0.129647.
- Ideal elevation greater than 1 degree: 0.111827.
- Per-trajectory positive-elevation coverage: 0.108--0.150.
- Initial ideal elevation: -3.824 to -2.609 degrees.
- Terminal ideal elevation: -51.050 to -36.393 degrees.
- Ideal slant range median: 4,721 km; 95th percentile: 8,732 km.
- Raw radar position-error mean/p95 already recorded by the candidate audit:
  2.733/7.083 km; filtered track mean/p95: 1.265/2.938 km.

The problem is structural rather than a noise-standard-deviation tuning issue.
The HGV travels roughly 9,486--12,804 km during the record, so one fixed ground
site cannot provide the assumed continuous line of sight.

## `hgv_multiregime_state_v2` contract

1. Generate 1,200 complete trajectories at 1 Hz for 999 s.
2. Preserve the three primary maneuver labels:
   `longitudinal`, `turning`, and `weaving`.
3. Balance two vertical regimes inside every primary label:
   `quasi_equilibrium` and `skip_glide`.
4. Use 200 trajectories in each of the six joint strata.
5. Randomize attack-angle thresholds, attack-angle limits, bank magnitude,
   maneuver direction, start time, duration, interval, and reversal count once
   per complete trajectory.
6. Store alpha and bank commands, control-parameter vectors, dynamic pressure,
   load factor, and a heating-rate proxy alongside the complete states.
7. Enforce altitude 30--80 km, speed 3.0--7.2 km/s, dynamic pressure no greater
   than 100 kPa, and load factor no greater than 5 g.
8. Enforce zero or one prominent altitude extremum for quasi-equilibrium
   trajectories and at least two for skip-glide trajectories.
9. Split complete trajectories jointly by vertical regime and primary maneuver
   into 960/120/120 train/validation/test trajectories.
10. Use 16 fixed, evenly spaced training origins per training trajectory and a
    five-step evaluation stride. This reduces redundant overlapping training
    windows without changing the independent statistical unit.
11. Treat command histories as simulation evidence only; do not expose them to
    any predictor at test time.
12. Keep sensor noise, missing samples, and tracking filters as separately
    identified robustness protocols rather than silently embedding an invalid
    observation network into the main benchmark.

## Pilot gate

The deterministic 36-trajectory pilot passed:

- six trajectories in each joint stratum;
- 24/6/6 disjoint complete-trajectory split;
- altitude 35.57--78.91 km;
- speed 4.219--6.931 km/s;
- maximum dynamic pressure 95.33 kPa;
- maximum load factor 0.980 g;
- quasi-equilibrium prominent-extrema count exactly 0;
- skip-glide prominent-extrema count 4--7;
- longitudinal bank exactly zero;
- turning/weaving bank magnitudes 20.37--34.38 degrees;
- all weaving trajectories contain both bank signs;
- generation acceptance rate 36/42 = 85.7%.

This pilot is a protocol test only. It is not paper evidence and does not
authorize formal training.

## Formal evidence gate

Formal training remains blocked unless all of the following pass on the
1,200-trajectory artifact:

- exact six-stratum balance and joint-stratified split;
- finite state/control/load arrays and immutable SHA-256 manifest;
- physical-envelope and maneuver-semantic checks;
- quasi-equilibrium/skip separation across every stratum;
- representative-trajectory and operating-envelope figure review;
- manuscript data counts, timing, control ranges, and limitations regenerated
  from the same formal artifact;
- no legacy metric, figure, scaler, checkpoint, or table is relabeled as a
  result from the new protocol.
