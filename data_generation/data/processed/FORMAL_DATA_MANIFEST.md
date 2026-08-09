# Formal HGV Dataset Manifest

- Generated: 2026-07-31 (Asia/Shanghai)
- Protocol: `hgv_multiregime_state_v2_1`
- Artifact: `hgv_multiregime_dataset_v2_1.npz`
- SHA-256: `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`
- Generation/split seeds: 42/42
- Dynamics: rotating spherical-Earth 3-DOF point mass
- Atmosphere: 1976 standard-atmosphere layer model
- Aerodynamics: published polynomial HGV parameterization
- Trajectories: 1800; 300 per quasi-equilibrium/skip-glide x longitudinal/turning/weaving stratum
- Samples per trajectory: 1000 at 1 Hz (999 s duration)
- Initial-condition/control design: split-wise, joint-stratum Latin hypercube
- Bank handling: ideal command and 1.5--3.0 deg/s rate-limited achieved response stored separately
- Complete-trajectory split: 1260/180/360 train/validation/test
- Window protocol: 256-step history, 256-step target; 16 fixed training origins and evaluation stride 5
- Window counts: 20160/17640/35280 train/validation/test
- Formal data audit: passed
- Numerical convergence audit: passed on 20 trajectories per joint stratum (120 total); maximum ECEF discrepancy 22.04 m

The NPZ is the only authorized source for new Main Results and ablations.  The
older `trajectory_level_v1`, rejected `pit_aligned_radar_v1`, and predecessor
`hgv_multiregime_state_v2` artifacts remain provenance only.
