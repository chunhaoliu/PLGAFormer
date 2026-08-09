# TAES Review Revision Matrix

This file is the evidence ledger for the post-AST revision. A reviewer point is
closed only when code, an output artifact, and manuscript text agree.

| ID | Reviewer concern | Required evidence | Status |
|---|---|---|---|
| R1-1 | Analytical HGV predictors and Earth rotation omitted | Rotating-Earth 3-DOF equations, documented approximation, model-driven baselines | Open |
| R1-3 | Six states mislabeled as 6-DOF | 3-DOF terminology in code, figures, and manuscript | Open |
| R1-4 | Altitude plotted above 6000 km | Altitude defined as `r - R_earth` in all paper figures | Open |
| R1-5 | Undefined or implausible concepts | Symbol table, exact implementation equations, language audit | Open |
| R1-6 | Contribution resembles a generic residual predictor | Explicit HGV phase, spherical geometry, and dynamics-projection evidence | Open |
| R2 | Weak HGV-specific rationale and novelty | Theory-linked priors, interpretable gates, domain baselines, generalization tests | Open |
| R3-1/2 | Horizon and table inconsistencies | One canonical protocol and artifact-generated tables | Open |
| R3-3 | Window leakage | Trajectory-level split validation and independent-unit reporting | Partial |
| R3-4/5 | Learned similarities called physical priors; unclear corrector | Explicit priors and code-matched corrector equations | Open |
| R3-6 | Invalid spherical ADE/FDE and heterogeneous losses | ECEF metrics and dimensionless constraint residuals | Open |
| R3-7 | No significance analysis | Multi-seed estimates, 95% CIs, paired trajectory-level tests | Open |
| R3-8 | No direct rollout evidence | Per-maneuver 3D/ground-track/altitude/error panels | Open |
| R4 | Missing gate curves, A-only ablation, and deployment workflow | Logged gates, expanded ablation, latency and online workflow | Open |
| R5 | Baselines, inference time, and aerodynamic fidelity | Modern neural + physics/model baselines, measured latency, validated aero model | Open |

## Closure rule

`Open -> Partial -> Closed` requires, in order: implementation, passing tests,
formal result artifact, manuscript integration, and final PDF verification.
