# Multi-Regime HGV Data Protocol v2.1

Date: 2026-07-31

## Decision

`hgv_multiregime_state_v2_1` is the only active source for new Main Results
and ablations.  It replaces, but does not overwrite, the 1,200-trajectory v2
artifact.

## Why v2.1 was necessary

The v2 audit found four scientific weaknesses before training:

1. the low-speed branch of the skip-glide attack law had zero material
   occupancy;
2. bank commands entered the dynamics as discontinuous steps;
3. every trajectory began at the same longitude and latitude and within a
   narrow heading interval;
4. exact discrete stratum balance did not guarantee balance of continuous
   initial-state and control covariates across splits.

## Frozen contract

- 1,800 complete trajectories, 1,000 samples each at 1 Hz;
- six exact strata: quasi-equilibrium/skip glide crossed with longitudinal,
  turning, and weaving;
- 300 complete trajectories per joint stratum;
- split-wise, joint-stratum Latin-hypercube candidate designs followed by
  physical-constraint rejection and resampling of infeasible candidates;
- initial altitude 55--70 km, speed 6.0--7.0 km/s, longitude -20--20 deg,
  latitude -25--25 deg, and heading 75--105 deg;
- attack-law thresholds 5.1--5.5 and 5.9--6.4 km/s;
- 20--35 deg ideal bank commands and 1.5--3.0 deg/s raised-cosine achieved
  responses stored as separate arrays;
- complete-trajectory split 1,260/180/360, or 210/30/60 per joint stratum;
- 16 evenly spaced training origins and evaluation stride 5;
- 256-step observed history and one direct 256-step forecast, with metrics read
  at 32/64/128/256-step checkpoints from that same forecast;
- no command or future-state input to any predictor.

## Derived window views

Input-length sensitivity and horizon-specific diagnostics are derived from the
stored complete trajectories; they do not regenerate the simulator artifact.
Every derived view preserves complete-trajectory IDs and split membership and
records the frozen parent SHA-256. The paper-facing input sensitivity is
`64/128/256 -> 256`. Separately trained `256 -> 64/128/256` runs remain internal diagnostics and are not part of the current paper evidence.

## Formal evidence

- Artifact: `data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz`
- SHA-256: `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`
- Formal audit: passed
- Split overlap: none
- Window counts: 20,160/17,640/35,280
- Altitude: 34.638--79.961 km
- Speed: 4.105--6.999 km/s
- Maximum dynamic pressure: 99.953 kPa
- Maximum load factor: 1.164 g
- Skip-law branch occupancy: 10.09% low, 49.80% transition, 40.11% high
- Maximum audited bank rate: 2.994 deg/s
- Maximum continuous-covariate standardized mean difference: 0.235
- Strict DOP853 audit: 20 trajectories per stratum, 120 total; maximum ECEF
  discrepancy 22.04 m against rtol 1e-10, atol 1e-12, max step 1 s

## Claim boundary

This protocol supports a controlled simulation-state forecasting benchmark.
It does not establish flight-test validity, operational radar observability,
full six-degree-of-freedom vehicle fidelity, calibrated thermal feasibility,
or generalization to an independently developed simulator.  Those boundaries
must remain explicit in the manuscript.
