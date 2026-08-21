# PLGAFormer Paper Evidence Map

## Authority and Claim Boundary

- Dataset protocol: `hgv_multiregime_state_v2_1`.
- Dataset SHA-256: `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`.
- Formal configuration SHA-256: `967310425a89d5a0acdb098c77939647aa22548335719846cdf274a8dcf04165`.
- Main bundle: `ff4a707476fa98c8270880bbc2f7be23c9a73b8ad89462c18cf8a6a434d208ac`.
- Ablation bundle: `afa2867d0a7a5d0f9e0d38e6b3691268e5cbdae836642053192029637bd6026f`.
- Complete paper bundle: `ebbcec4b2321dafeb825878b873eddf710ade15d5a242673ee170a494f8585a6`.
- Adaptive-gate analysis: `2453b0a7c1c01a8b8d3e6bd826542d872fbd8cde8aa8332e6ca7773775bc16b8`.
- Claim boundary: complete simulated HGV trajectories only; no flight, radar,
  hardware-in-the-loop, or deployment validation.

Only evidence carrying the identities above can support the current paper.
Historical records, convergence pilots, PIT, AF-CILN, and standalone analytical
propagator results are not part of the paper-facing comparison matrix. The
rotating-Earth 3-DOF model remains part of PLGAFormer as its identified prior.

## Frozen Protocol

| Item | Verified setting |
|---|---|
| Complete trajectories | 1,800; 1,000 samples each at 1 Hz |
| Joint strata | Six; 300 trajectories per stratum |
| Complete-trajectory split | 1,260 / 180 / 360 train/validation/test |
| Input and direct forecast | 256 s / 256 s |
| Reporting horizons | 32, 64, 128, and 256 s |
| Training windows | 20,160 from 16 fixed origins per training trajectory |
| Validation/test windows | 17,640 / 35,280 with stride 5 |
| Learned-model seeds | 42, 123, and 456 |
| Training budget | At most 50 epochs; 5 warm-up epochs; patience 15 |
| Test policy | One frozen evaluation after validation checkpoint selection |
| Statistical unit | Held-out source trajectory, not an overlapping window |

## Public Evidence Routing

The path-free tables under `PublicRelease/evidence` are the compact public
record. `evidence_manifest.json` binds every CSV hash to the dataset,
configuration, Main, Ablation, complete paper, and adaptive-gate identities.

| Evidence question | Public artifact |
|---|---|
| Overall multi-horizon accuracy | `main_results.csv` |
| Maneuver-resolved accuracy | `maneuver_results_256s.csv` |
| Internal analytical-comparator audit | `strongest_comparator.csv` |
| Mechanism controls | `ablation_256s.csv` |
| Learned-only capacity control | `capacity_control_256s.csv` |
| Adaptive-gate marginal contribution | `adaptive_gate_paired.csv` |
| Dynamics-shift robustness | `robustness.csv` |
| Computational cost | `efficiency.csv` |

## Supported Quantitative Statements

1. Among the learning-based methods, PLGAFormer has the lowest mean ADE, FDE, and
   Cartesian RMSE at every reported horizon. At 256 s it obtains 4.816 km ADE,
   14.350 km FDE, and 4.590 km RMSE.
2. Relative to iTransformer, the strongest learning-based comparator at 256 s,
   PLGAFormer reduces ADE, FDE, and Cartesian RMSE by 31.8%, 20.5%, and 29.9%,
   respectively.
3. At 256 s, the spherical-prior, fixed-schedule rotating-prior, and final
   adaptive-fusion variants obtain 6.202/17.406, 4.943/14.746, and
   4.816/14.350 km ADE/FDE, respectively.
4. Relative to the fixed schedule at 256 s, adaptive fusion reduces mean ADE
   by 126.620 m (2.562%) and mean FDE by 396.776 m (2.691%). The conditional
   trajectory-level Holm value is 0.00490 for both metrics, but the crossed
   seed-and-trajectory 95% intervals include zero: [-447.719, 162.227] m for
   ADE and [-1394.255, 942.892] m for FDE. Only two of three seeds favor the
   final model, so the 256 s marginal gate benefit is not seed-robust.
5. The learned-only PLGAFormer backbone obtains 26.301/36.033/17.343 km
   ADE/FDE/RMSE at 256 s across three seeds. It uses the same three-encoder,
   two-decoder, 256-dimensional learned backbone as the final model, so the
   final gain is not explained by a deeper learned path alone under this
   training protocol.
6. Under the aerodynamic shift, PLGAFormer obtains 4.855/12.796 km ADE/FDE,
   compared with 20.130/38.904 km for Transformer. Under the ballistic shift,
   PLGAFormer obtains 5.029/13.142 km, compared with 15.869/30.597 km for
   Transformer.

7. On the measured NVIDIA GeForce RTX 4090, PLGAFormer has 4.546 M parameters,
   3,021.7 MFLOPs, and 491.44 ms batch-one latency for the complete 256 s
   forecast. The paper reports these fields as hardware-specific feasibility evidence.

## Unsupported or Restricted Claims

- Do not claim overall state of the art or uniform superiority; longitudinal
  256 s FDE is lower for iTransformer than for PLGAFormer.
- Do not promote internal standalone analytical-propagator results into the
  paper-facing comparison unless a reviewer explicitly requests that analysis.
- Do not claim that the adaptive gate has a seed-robust 256 s advantage over
  the fixed schedule.
- Do not include PIT or AF-CILN in the paper-facing Main table.
- Do not claim random-missing-observation, sensor-noise, shortened-history,
  flight-data, radar-data, or deployment robustness from the current bundle.
- Do not describe the final model as using a channel-residual head; selected
  records set `use_channel_residual=false`.
- Do not treat overlapping windows as independent trajectories.
- Do not describe PLGAFormer as generally lightweight.

## Manuscript Synchronization Status

The canonical TAES manuscript has been synchronized to the frozen trajectory
count, split, training budget, learning-based public baseline matrix, four-row
mechanism ablation, dynamics-shift evidence, and restrained adaptive-gate
conclusion. Compilation and visual QA must be rerun after each manuscript sync.

Remaining manuscript work is editorial rather than experimental: verify every
table and prose number against the public evidence CSVs, check references and
terminology, refine figures, and run a final reviewer-style submission audit.
