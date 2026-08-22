# PLGAFormer Paper Evidence Map

## Authority and Claim Boundary

- Dataset protocol: `hgv_multiregime_state_v2_1`.
- Dataset SHA-256: `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`.
- Formal configuration SHA-256: `967310425a89d5a0acdb098c77939647aa22548335719846cdf274a8dcf04165`.
- Main bundle: `82208600652c5cebb249c9563a9b34e28e2a50d66673a71d6ffe3933d1924427`.
- Ablation bundle: `0657442fbc7042d6b3ce1a9e6f26e074424029fe5c7ba3cd4b6951f19bd799ba`.
- Complete paper bundle: `602dc912221d810ee79122317f9f7824a2c8083d1f36c6bcd9229d652eda35c7`.
- Adaptive-gate analysis: `2453b0a7c1c01a8b8d3e6bd826542d872fbd8cde8aa8332e6ca7773775bc16b8` (historical diagnostic; not used for the promoted paper-facing values).
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
configuration, Main, Ablation, complete paper, and retained diagnostic
identities. The adaptive-gate paired analysis and computational-cost table are
kept for traceability and are not current manuscript evidence.

| Evidence question | Public artifact |
|---|---|
| Overall multi-horizon accuracy | `main_results.csv` |
| Maneuver-resolved accuracy | `maneuver_results_256s.csv` |
| Internal analytical-comparator audit | `strongest_comparator.csv` |
| Mechanism controls | `ablation_256s.csv` |
| Learned-only capacity control | `capacity_control_256s.csv` |
| Adaptive-gate marginal contribution (historical diagnostic) | `adaptive_gate_paired.csv` |
| Dynamics-shift robustness | `robustness.csv` |
| Computational cost (retained diagnostic) | `efficiency.csv` |

## Supported Quantitative Statements

1. Among the listed learning-based methods, PLGAFormer has the lowest mean ADE,
   FDE, and Cartesian RMSE at every reported horizon. At 256 s it obtains
   4.102 km ADE, 13.301 km FDE, and 4.420 km RMSE.
2. Relative to iTransformer, the strongest listed learning-based comparator at
   256 s, PLGAFormer reduces ADE, FDE, and Cartesian RMSE by 41.9%, 26.4%, and
   32.5%, respectively.
3. At 256 s, the spherical-prior, fixed-schedule rotating-prior, and final
   adaptive-fusion variants obtain 8.782/19.715, 4.299/13.658, and
   4.102/13.301 km ADE/FDE, respectively.
4. Relative to the fixed schedule at 256 s, the promoted adaptive policy gives
   descriptive mean reductions of 0.197 km ADE, 0.357 km FDE, and 0.096 km
   RMSE across the three seeds. These differences are not presented as
   inferentially significant.
5. The learned-only PLGAFormer backbone obtains 26.301/36.033/17.343 km
   ADE/FDE/RMSE at 256 s across three seeds. It uses the same three-encoder,
   two-decoder, 256-dimensional learned backbone as the final model, so the
   final gain is not explained by a deeper learned path alone under this
   training protocol.
6. Under the promoted-policy robustness artifact, PLGAFormer obtains
   4.775/12.796 km ADE/FDE under the aerodynamic shift and 4.952/13.142 km
   under the ballistic shift, compared with 20.130/38.904 km and
   15.869/30.597 km for Transformer, respectively.
7. The paper-facing policy is inference-only: it locks the adaptive prior for
   64 forecast steps, then uses $\tau=450$ s and $p=2.5$; formal checkpoints,
   data, splits, and training parameters are unchanged.
## Unsupported or Restricted Claims

- Do not claim overall state of the art or uniform superiority beyond the listed
  learning-based methods; the current comparison does not establish superiority
  over unreplicated related methods or the analytical propagator.
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

The canonical TAES manuscript has been synchronized to the promoted Main/Ablation/robustness artifacts, the frozen trajectory count and split, the learning-based public baseline matrix, the four-row mechanism ablation, and the inference-only policy boundary. Compilation and visual QA must be rerun after each manuscript sync.
Remaining manuscript work is editorial rather than experimental: verify every
table and prose number against the public evidence CSVs, check references and
terminology, refine figures, and run a final reviewer-style submission audit.
