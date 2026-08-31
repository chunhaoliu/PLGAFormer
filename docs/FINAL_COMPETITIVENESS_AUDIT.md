# PLGAFormer Final Competitiveness and Submission-Validity Audit

Date: 2026-08-31

## Decision

**Submission readiness: blocked pending an untouched confirmatory holdout.**

The retained evidence is complete and PLGAFormer is clearly competitive
against the five listed trainable methods. However, historical candidate
development inspected subsets of the original test partition before the final
paper inference policy was frozen. The current 360-trajectory test partition
is therefore not an untouched confirmatory set for that policy.

This finding does not invalidate the stored measurements. It changes their
role: they are valid development evidence, but they are not sufficient as the
sole confirmatory basis for the final paper claim.

## Evidence Identity

- Dataset protocol: `hgv_multiregime_state_v2_1`
- Dataset SHA-256:
  `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`
- Formal configuration SHA-256:
  `967310425a89d5a0acdb098c77939647aa22548335719846cdf274a8dcf04165`
- Main bundle:
  `82208600652c5cebb249c9563a9b34e28e2a50d66673a71d6ffe3933d1924427`
- Ablation bundle:
  `0657442fbc7042d6b3ce1a9e6f26e074424029fe5c7ba3cd4b6951f19bd799ba`
- Formal status and audit: `paper_eligible=true`, 17 Main records, six
  mechanism records, and no evidence blockers.

`paper_eligible` verifies artifact completeness and provenance. It does not
by itself establish test-set independence or unrestricted superiority.

## Competitiveness Against Listed Learning-Based Methods

PLGAFormer ranks first in all 12 combinations of four horizons and three
metrics among DLinear, Transformer, PatchTST, iTransformer, and PLGAFormer.

| Horizon | Metric | PLGAFormer (km) | Strongest other learned method | Reduction |
|---:|---|---:|---|---:|
| 32 s | ADE | 0.036 | DLinear, 0.584 | 93.9% |
| 32 s | FDE | 0.090 | DLinear, 1.469 | 93.9% |
| 32 s | RMSE | 0.044 | DLinear, 0.459 | 90.3% |
| 64 s | ADE | 0.145 | iTransformer, 2.041 | 92.9% |
| 64 s | FDE | 0.465 | iTransformer, 2.710 | 82.9% |
| 64 s | RMSE | 0.207 | iTransformer, 1.440 | 85.6% |
| 128 s | ADE | 0.784 | iTransformer, 3.002 | 73.9% |
| 128 s | FDE | 2.649 | iTransformer, 5.550 | 52.3% |
| 128 s | RMSE | 0.938 | iTransformer, 2.292 | 59.1% |
| 256 s | ADE | 4.102 | iTransformer, 7.059 | 41.9% |
| 256 s | FDE | 13.301 | iTransformer, 18.061 | 26.4% |
| 256 s | RMSE | 4.420 | iTransformer, 6.546 | 32.5% |

At 256 s, PLGAFormer also ranks first among the listed learning-based methods
for ADE and FDE in each of the longitudinal, turning, and weaving groups.

Supported paper wording is therefore restricted to:

> Among the five evaluated learning-based methods, PLGAFormer obtains the
> lowest mean ADE, FDE, and Cartesian RMSE at all four reported horizons under
> the evaluated simulation protocol.

The evidence does not support an unrestricted state-of-the-art claim.

## Internal Analytical-Anchor Boundary

The rotating-Earth 3-DOF propagator remains an internal analytical anchor and
is not part of the paper-facing learning-method ranking. Its comparison is
still necessary for internal claim control because PLGAFormer uses that
propagator as its prior.

- At 32 and 64 s, PLGAFormer is effectively identical to the anchor because
  the paper policy locks the prior for the first 64 forecast steps.
- At 256 s, PLGAFormer improves the aggregate result over the anchor, but the
  gain is maneuver dependent.
- Relative to the anchor at 256 s, PLGAFormer is worse on longitudinal and
  turning ADE/FDE, but better on weaving ADE/FDE.

Consequently, the paper may claim improved long-horizon learning-based
forecasting and weaving-regime correction. It must not claim uniform
superiority over the analytical propagator or imply that the learned branch
drives the short-horizon result.

## Mechanism Evidence

The evidence supports two method mechanisms and one capacity control:

1. an online-identified rotating-Earth dynamics prior;
2. bounded, disagreement-aware adaptive fusion;
3. an architecture-matched learned-only backbone used to rule out capacity as
   the explanation for the final gain.

The learned-only backbone obtains 26.301/36.033/17.343 km
ADE/FDE/RMSE at 256 s. The rotating-Earth prior is the dominant source of the
long-horizon reduction. Adaptive fusion improves the fixed schedule by only
0.197 km ADE, 0.357 km FDE, and 0.096 km RMSE in the three-seed mean. These
adaptive-fusion differences are descriptive and must not be described as
statistically significant or uniformly seed robust.

The manuscript should therefore present two technical mechanisms plus one
capacity-matched control, rather than inventing three independent novel
modules.

## Robustness Boundary

Under the two configured simulator shifts, PLGAFormer reduces ADE/FDE relative
to Transformer by 76.3%/67.1% for the aerodynamic shift and 68.8%/57.0% for
the ballistic shift. This supports robustness only to those two registered
dynamics perturbations. It does not support claims about measurement noise,
missing observations, arbitrary maneuvers, flight data, radar data, or
deployment.

## Manuscript Synchronization Audit

The canonical manuscript currently matches the compact evidence for:

- 1,800 complete trajectories and the 1,260/180/360 split;
- 32, 64, 128, and 256 s horizons;
- the five-method learning-based comparison;
- the 256 s Main values and relative reductions;
- the learned-only, prior, and fusion controls;
- the two registered dynamics shifts;
- the simulation-only claim boundary.

Required editorial changes after confirmatory evidence is available:

1. organize the experiment section as Experimental Setup, Overall Prediction
   Performance, Ablation Study, and Generalization and Robustness;
2. merge maneuver-resolved evidence into Overall Prediction Performance;
3. remove the defensive PIT/AF-CILN source-code explanation from the experiment
   prose while retaining those methods in Related Work;
4. keep efficiency outside the main experiment structure;
5. retain the 3-DOF mechanism in the method, while omitting it from the
   paper-facing learning-method ranking as previously decided.

## Reproducibility Defect and Repair

The promoted records use the inference-only policy
`lock_steps=64`, `time_constant_s=450`, and `decay_power=2.5`.
During single-mainline consolidation, the result records were retained but the
minimal lock/power implementation was not carried back into `main`.

The active repair:

- restores constructor controls while preserving training defaults
  `lock_steps=0`, `time_constant_s=450`, and `decay_power=2.0`;
- defines one immutable paper policy;
- applies it only after training and before final evaluation;
- routes final-checkpoint consumers through the same constructor identity;
- records the resolved policy in model audits and new formal records.

This repairs source-to-checkpoint reconstruction. It does not cure the
historical test-subset use described below.

## Confirmatory Holdout Requirement

The final policy family and lock length were informed by earlier diagnostics
on subsets of the original test partition. Although decay power 2.5 was later
selected on the complete validation split before its corresponding full-test
metrics were read, the complete final policy was not developed without prior
test feedback.

The scientifically clean next experiment is therefore:

1. generate and freeze a new, non-overlapping confirmatory set with the same
   six strata and simulation contract;
2. publish its generation seed, manifest, trajectory IDs, and SHA-256 before
   model evaluation;
3. make no further model, policy, scaler, checkpoint, or threshold changes;
4. evaluate the existing three-seed checkpoints for all five trainable methods
   under the same horizons and metrics;
5. use that untouched result as the final Main table, regardless of outcome.

The existing original test partition should then be described as development
evidence for policy selection, not as the untouched final test. A new
confirmatory set is a protocol extension and must be explicitly approved
before generation; it must not overwrite the frozen v2.1 artifact.
