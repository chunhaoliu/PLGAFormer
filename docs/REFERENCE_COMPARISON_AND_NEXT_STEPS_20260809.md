# Reference comparison and next evidence steps

Date: 2026-08-09
Scope: the single PLGAFormer paper line and its current frozen experiment protocol

This document converts the local HGV reference set into an evidence plan. It is
not a ranking of published RMSE values: the papers use different dynamics,
coordinates, horizons, sampling rates, split units, and observation models, so
their numerical results must not be pooled with the current project results.

## 1. Evidence sources

The comparison was built from the following local materials:

| Evidence source | Use in this audit |
|---|---|
| docs/REFERENCE_VERIFICATION_20260722.md | Bibliographic audit of the 26 entries in the current TAES reference file; all have DOI, unique DOI, journal status, and verified metadata. |
| HGV_Project/References/Long-Term_Trajectory_Prediction_of_Hypersonic_Glide_Vehicle_Based_on_Physics-Informed_Transformer.pdf | Closest parent study: physics-informed Transformer, long-horizon prediction, ablation, missing observations, and small-data tests. |
| HGV_Project/References/Dual-Channel_and_Bidirectional_Neural_Network_for_Hypersonic_Glide_Vehicle_Trajectory_Prediction.pdf | HGV maneuver taxonomy, rolling prediction interface, and model-driven versus data-driven comparison routine. |
| HGV_Project/References/Hypersonic glide vehicle trajectory prediction based on frequency.pdf | Recent lightweight HGV baseline: FECA-LSMN, auxiliary-feature fusion, input-length and efficiency evidence. |
| HGV_Project/References/AST_Convolutional interactive learning network with auxiliary features for.pdf | Recent HGV baseline: AF-CILN, randomized controls, missing-data protocol, ablation, and efficiency evidence. |
| docs/EXPERIMENT_PROTOCOL_REGISTRY.md and the current machine-readable configuration | Authority for the active dataset, task, model matrix, seeds, and paper-evidence gates. |

The bibliography audit establishes reference authenticity, not reproduction
provenance. Whether an external implementation is official, reimplemented, or
parameter-free must remain an artifact-level property of the experiment.

## 2. Closest reference routines

| Reference | Motion/data/task protocol | Method and evidence routine | What we can borrow | Gap that the current line must close |
|---|---|---|---|---|
| Ren et al., IEEE TAES 2023, PIT | Simulated HGV data; 640 trajectories, 1,000 points per trajectory, 2 s sampling; five state features in the prediction interface; skip-glide with C/S lateral maneuvers; noisy radar observations followed by filtering; reported long horizons 64/128/256/512. | Stage encoding plus physics-informed loss, top-τ mean attention, generative decoder; Transformer/LSTM/Informer comparisons; ten repetitions; ablation of attention and physical knowledge; missing rates 1/5/8%; small-data test with 140 trajectories. | Closest paper-level routine: explicit motion challenge, physics mechanism, long-horizon table, ablation, missing-data and small-data checks. | Its spherical nonrotating dynamics, filtered radar observation, five-feature interface, and horizon/split details are not identical to the active protocol. Therefore its numbers are context, not current Main evidence. The registered PIT comparator still needs all three formal seeds. |
| Xie et al., IEEE Access 2021, DCBNN | Homogeneous nonrotating sphere; three maneuver families (longitudinal-only, turning, weaving); 1 Hz data with trajectories extending to about 2,000 s; six-dimensional ECEF-position plus motion-state interface; rolling one-step prediction. | Dual global/local temporal channels, Bi-GRU, and linear branch; pretraining/retraining discussion; comparison with six models across maneuver-specific data sets. | Keep maneuver labels physically interpretable and show that a model is tested on more than one lateral pattern. | It is primarily an autoregressive one-step/rolling protocol with fixed maneuver laws, so it is not a direct 256-step direct-decoder comparator. The current line should use its taxonomy as context, not silently reproduce its metric protocol. |
| Cai and Zhuang, Defence Technology 2025, FECA-LSMN | Homogeneous nonrotating sphere; 4,000 simulated trajectories, split into 2,000 skip-glide and 2,000 quasi-equilibrium; 1,000 samples at 1 Hz; six state channels; 0--3 lateral maneuvers; randomized initial/control parameters; 80/10/10 split. | Frequency-enhanced channel attention plus lightweight sampling-oriented MLP and RevIN; RMSE/MAE/MAPE; input-length, ablation, and computational-cost analyses. | Include a recent lightweight/efficiency reference and separate accuracy from deployment cost. | FECA-LSMN is not part of the frozen registered model matrix. Adding it now would expand the protocol and require a full matched rerun; it is a candidate follow-up baseline, not a reason to delay the currently missing Main units. |
| Cai et al., Aerospace Science and Technology 2026, AF-CILN | 8,000 trajectories: 4,000 quasi-equilibrium and 4,000 skip-glide; 1,000 samples at 1 Hz; six ECEF/motion channels; randomized controls and 0--3 lateral maneuvers; 80/10/10 split. Missing-data tests use 5/10/20/40% random or continuous masks; input lengths include 64/128/256 s. | AHWT auxiliary-feature fusion, convolutional interactive learning, mask normalization; CNN/RNN/Transformer/MLP comparisons; missing-data ablation; FLOPs/memory and input-length analysis. | Use as the recent HGV comparator for auxiliary states, missing observations, and efficiency. Keep its external code revision and source hash attached to every AF-CILN result. | Its simulator uses simplified dynamics while the active project uses a rotating-Earth state benchmark and keeps sensor degradation separate from the nominal data protocol. The current Main set already registers AF-CILN; its three seeds are complete, but this does not make the overall Main bundle eligible while PLGAFormer/PIT are missing. |

## 3. Protocol alignment: references versus the current line

| Dimension | Common pattern in the references | Current frozen project setting | Interpretation |
|---|---|---|---|
| Scientific task | HGV future-trajectory prediction under long-range maneuvering | Forecast the next 256 s from a 256 s observed six-state history | Same domain question, but not the same forecasting interface as rolling one-step papers. |
| State and output | Position plus selected velocity/attitude variables; coordinate systems vary | Input: six physical states; target: three ECEF Cartesian positions after inverse transformation | Physical distance metrics are comparable within the project matrix; cross-paper metric values are not directly interchangeable. |
| Dynamics | Most local references use a homogeneous nonrotating sphere or a simplified point-mass model | Rotating-Earth 3-DOF point-mass dynamics with six joint strata: two vertical regimes × three lateral maneuver families | This is the main domain-specific distinction, but it must be supported by analytical control and physical-consistency evidence. |
| Data diversity | Some papers vary initial states; newer papers randomize controls and distinguish skip/equilibrium modes | 1,800 complete trajectories, 300 per joint stratum, randomized initial/control designs, complete-trajectory split before windowing | This closes the historical single-family/fixed-control weakness documented in the data audit. |
| Sampling and horizon | 1--2 s sampling is common; horizons range from one-step rolling to 512 s | 1 s sampling; direct 256-step target; report 32/64/128/256 s checkpoints from the same forecast | Do not label the 32/64/128 rows as independently trained horizon-specific models. |
| Split unit | Reference papers often report 80/10/10 but do not always state whether the unit is a complete trajectory | 1,260/180/360 complete trajectories; windows are generated after the split | The independent statistical unit is the held-out complete trajectory, not an overlapping window. |
| Repeated runs | PIT reports ten repetitions; many papers report a single run or unclear repetition policy | Seeds 42/123/456, validation selection, one frozen test evaluation per run | Main tables can use mean ± standard deviation only after the required seed matrix is complete. |
| Robustness | PIT and AF-CILN emphasize missing observations; other papers emphasize model/data variation | Nominal data remain clean and fixed; noise, shortened history, and dynamics shifts are separate robustness conditions | This separation prevents an invalid sensor geometry or an unreported imputation rule from contaminating the nominal benchmark. |
| Efficiency | PIT, FECA-LSMN, and AF-CILN report computation-related evidence in different ways | Parameters, FLOPs, batch-one latency, throughput, and peak memory under a synchronized fixed setting | “Real-time” must be a measured conditional statement, not inferred from model name or FLOPs alone. |

## 4. What the paper can and cannot claim

### Claims supported by the planned evidence

1. **Matched-protocol accuracy:** PLGAFormer is better or worse than the
   registered comparators under the same active HGV data, split, task, loss,
   and evaluation procedure.
2. **Mechanism contribution:** the rotating-Earth prior and the learned
   refinement contribute only if the matched spherical_prior, schedule_only,
   and complete-model rows support that conclusion across the required seeds.
3. **Motion-regime behavior:** the six joint strata and trajectory-level
   analysis can show whether the method handles quasi-equilibrium, skip-glide,
   longitudinal, turning, and weaving cases consistently.
4. **Deployment evidence:** noise, shortened-history, dynamics-shift, and
   efficiency studies can support bounded reliability statements under the
   stated conditions.

### Claims that are not yet supported

- “PLGAFormer is better than all published HGV methods.” Published numerical
  results use incompatible protocols. The defensible wording is “better than
  the registered comparators under the matched protocol,” followed by a
  qualitative reference comparison.
- “The physics prior is beneficial.” The Main bundle is currently incomplete,
  and the matched ablation rows are absent.
- “The model is robust to radar missingness.” The nominal data protocol and
  the current robustness route must be kept separate; a missing-data claim
  requires the corresponding executed stress condition and artifact.
- “The model is real-time.” This requires the synchronized latency and memory
  record on the specified hardware and batch setting.

## 5. Current evidence status and exact next runs

The latest read-only formal audit reports:

- Dataset: passed; observed SHA-256 matches the frozen dataset record.
- Main Results: 23 records, not paper-eligible.
- Missing Main units: full:456 and pit:42, pit:123, pit:456.
- Ablation: no complete matched mechanism bundle.
- Missing ablation units: spherical_prior at 42/123/456 and
  schedule_only at 42/123/456; the complete model at seed 456 also cannot be
  reused until full:456 exists.

The execution order is therefore:

1. **Finish Main:** run the four missing units only: PLGAFormer seed 456, then
   PIT seeds 42/123/456 according to the existing resource order. Do not change
   the dataset, split, horizon, loss, optimizer, or model definition in these
   runs.
2. **Re-audit Main:** run the formal status check and require the complete
   three-seed matrix plus frozen test records before writing any performance
   table.
3. **Run matched ablation:** reuse the eligible baseline and complete-model
   controls, then train the six missing rows for spherical_prior and
   schedule_only at seeds 42/123/456.
4. **Run the two supporting studies:** use the eligible Main checkpoints for
   generalization/robustness and efficiency. These studies do not justify
   retraining or changing the nominal Main protocol.
5. **Synchronize the manuscript:** the local TAES experiment source still
   contains the older 1,000-trajectory and 799/99/102 split description, while
   the active registry is 1,800 trajectories and 1,260/180/360. Update data,
   task, training, metrics, and limitation statements only from the frozen
   registry and completed evidence manifests.
6. **Generate and inspect paper artifacts:** regenerate tables/figures from
   the eligible bundle, compile the manuscript, and perform visual QA. No
   historical or diagnostic result may be relabeled as evidence from the
   current protocol.

## 6. Scope decision on additional references

FECA-LSMN is a scientifically relevant recent reference, but it is not a
current execution blocker because the registered comparator set already
contains DLinear, PatchTST, iTransformer, AF-CILN, analytical controls, PIT,
and the matched Transformer/PLGAFormer pair. Adding FECA-LSMN before closing
the four missing Main units would change the experiment contract and multiply
the required seed runs. Reconsider it only after the Main bundle is complete,
and only as a deliberate protocol change with a new full matrix—not as an
unlogged one-off comparison.

The immediate goal is therefore evidence closure, not another architecture.
