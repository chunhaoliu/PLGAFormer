# TRANS Readiness Audit

This audit tracks whether the codebase and manuscript are ready for a Transactions-level submission or migration. It is intentionally evidence-first: claims are marked ready only when the supporting code, result artifact, and manuscript statement can be traced to one another.

## Current Verdict

Status: locally evidence-aligned for the historical 32/64/128-step Transactions-level draft; the code protocol has now been upgraded to a 1 Hz, 1000-trajectory, 256-step formal setting, but that new setting still requires full dataset regeneration and multi-seed reruns before final Transactions-level SOTA claims.

The project now has a reproducibility spine and an evidence-aligned manuscript draft: root `run.py`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `.gitignore`, dependency files, `PublicRelease/` boundaries, formal Exp1--Exp4/Exp6 artifacts, refreshed paper figures, and compiled manuscript PDFs are present. On 2026-06-23, a `pred_len=256` diagnostic dataset was generated/validated under the older sampling/window setting, Exp1 was made resumable via `partial_runs.json`, and seed-42 256-step runs were completed for Transformer, PLGAFormer, and Kalman with physical `RMSE/MAE/FDE/ADE` table output. The current code default is now the formal protocol: 1000 complete trajectories, 1000 points per trajectory, 1 Hz sampling, three primary maneuver classes, `seq_len=256`, `pred_len=256`, and `window_stride=5`. The key scientific blocker is that the new protocol still needs full dataset regeneration, PLGAFormer optimization, and multi-seed reruns before claiming 256-step superiority. The remaining external release blockers are target-template confirmation and pushing the release branch/tag to GitHub.

## Code And Repository Gates

| Gate | Status | Evidence | Required closeout |
|---|---|---|---|
| Git/GitHub migration structure | Ready locally | Local git repository exists; `origin` is configured as `https://github.com/chunhaoliu/PLGAFormer.git`; local release commit exists. | Push clean release branch/tag when network/authentication is available. |
| Human README | Ready | `README.md` documents purpose, environment, layout, commands, public boundary, and experiment status. | Keep updated after formal Exp5/Exp7 regeneration. |
| Agent instructions | Ready | `AGENTS.md` is the authoritative project instruction file; `CLAUDE.md` is a compatibility shim. | Do not reintroduce competing agent instructions. |
| Ignore policy | Ready | `.gitignore` exists. | Re-check after full experiment reruns for large result/checkpoint files. |
| Dependencies | Ready | `D:\ProgramData\anaconda3\envs\tslib\python.exe` provides Python 3.11, PyTorch 2.5.1+cu121, CUDA, NumPy/SciPy/scikit-learn, and matplotlib. | Use the `tslib` environment for experiment execution unless a newer project-specific environment is approved. |
| Unified entrypoint | Ready | `run.py` supports core tasks, enhancement tasks, `--quick`, and `--dry-run`; help and dry-run work without training imports. | Keep new experiments callable through this root entrypoint. |
| Public release boundary | Ready | `PublicRelease/README.md` documents included/excluded artifacts and reproducibility expectations. | Add dataset/checkpoint inclusion decision at release time. |

## Experiment Gates

| Experiment | Status | Evidence | Required closeout |
|---|---|---|---|
| Exp1 SOTA | Ready for historical 32/64/128; new 1 Hz 256-step protocol not SOTA-ready | Formal 3-seed artifacts under `experiments/exp1_sota/results/formal/` cover 32/64/128 under the older protocol. Seed-42 256-step diagnostic runs exist for Transformer, PLGAFormer, and Kalman in `experiments/exp1_sota/results/partial_runs.json`; current paper tables can include those diagnostic physical `RMSE/MAE/FDE/ADE` values. | Regenerate the formal 1000-trajectory, 1 Hz dataset; optimize PLGAFormer or training protocol for 256-step performance; rerun at least the core comparison over multiple seeds before claiming final 256-step superiority. |
| Exp2 ablation | Ready | Formal artifacts under `experiments/exp2_ablation/results/formal/`. | Ensure manuscript ablation table uses the same values. |
| Exp3 robustness | Ready | Formal JSON/CSV regenerated under `experiments/exp3_robustness/results/`; figures refreshed from the same JSON and synced into both manuscript figure directories. | Keep robustness described as PLGAFormer-only stress testing unless a comparative robustness experiment is added. |
| Exp4 physics consistency | Ready | Formal JSON/CSV/figures regenerated under `experiments/exp4_physics_consistency/results/`; figures synced into both manuscript figure directories. | Keep denormalization and constraint definitions explicit in manuscript. |
| Exp5 missing-data model comparison | Excluded from manuscript claims | `run.py --task exp5 --quick` completed all 27 model/missingness scenarios; quick artifacts are isolated under `results/quick/`. Manuscript no longer uses Exp5 comparative claims, and `table_missing_data.tex` intentionally remains pending for full Exp5. | Run full Exp5 only if the manuscript is expanded to include comparative missing-data claims. |
| Exp6 efficiency | Ready | `run.py --task exp6` completed with batch=64, warmup=10, repeats=50 and refreshed JSON/CSV/PNG outputs plus paper table. | Re-run only if model definitions or target hardware change. |
| Exp7 extended ablation | Excluded from manuscript claims | `run.py --task exp7 --quick` completed; quick artifacts are isolated under `results/quick/`. Manuscript does not rely on Exp7 extended-ablation claims. | Run full Exp7 only if the manuscript is expanded to include extended-ablation claims. |

## Manuscript Gates

| Gate | Status | Risk | Required closeout |
|---|---|---|---|
| Template/target alignment | Needs external decision | `Submit_AST/` uses Elsevier CAS files; an IEEE Transactions submission would require a different template and citation style. | Confirm whether target is IEEE Transactions or an Elsevier Transactions-style journal. |
| Code availability statement | Ready pending push | Non-anonymous manuscript gives the intended GitHub URL; anonymous manuscript uses blinded release wording; local `origin` now matches the intended repository URL. | Push release tag before final submission. |
| Data availability statement | Ready | Both manuscript versions include a Data and Code Availability section aligned with `PublicRelease/README.md`: code/lightweight artifacts public, large datasets/checkpoints regenerated or requested. | Replace placeholder release wording with final URL/tag after publication decision. |
| Claim-evidence alignment | Ready for historical claims; new 1 Hz 256-step superiority not supported yet | Main 32/64/128 SOTA, ablation, robustness, and physics-consistency values match historical formal artifacts or generated paper tables. Current 256-step diagnostic tables are evidence-backed, but show PLGAFormer worse than Transformer for seed 42 under the older protocol. | Do not write 256-step superiority claims until the new 1 Hz protocol has regenerated data and multi-seed reruns support them. |
| Missing-data claims | Calibrated | Missing observations are discussed only inside PLGAFormer-only Exp3 robustness; comparative Exp5 claims are excluded until full Exp5 exists. | Add Exp5 only after formal full-scale regeneration. |
| Statistical reporting | Mixed | Historical 32/64/128 SOTA table reports mean/std over three seeds. Current 256-step active table reports seed-42 only, so std is zero and should be treated as diagnostic. The SOTA artifact generator labels the active source in the caption. | Run the 256-step core comparison over multiple seeds and regenerate artifacts before final submission. |

## Verification Log

- `python -B -c ast.parse(...)` passed for edited entrypoints and experiment scripts.
- `python -B run.py --help` passed.
- `python -B run.py --task enhancements --quick --dry-run` passed.
- Bundled Python lacked `torch`/`matplotlib`; a repository-local `.venv` install attempt timed out and was abandoned.
- `D:\ProgramData\anaconda3\envs\tslib\python.exe` was identified as the working experiment environment.
- `tslib` verification passed: PyTorch 2.5.1+cu121, CUDA available, matplotlib available.
- `tslib` ran `run.py --help` and `run.py --task enhancements --quick --dry-run` successfully.
- `tslib` ran `run.py --task exp5 --quick` successfully.
- `tslib` ran `run.py --task exp6` successfully.
- `tslib` ran `run.py --task exp7 --quick` successfully.
- `tslib` ran `run.py --task enhancements --quick` successfully end-to-end.
- `tslib` reran formal `run.py --task exp3` and `run.py --task exp4` after smoke-output isolation, restoring official root result artifacts.
- `tslib` ran `run.py --task validate-data`; dataset protocol `trajectory_level_v1` passed with disjoint train/val/test trajectories and finite physical values.
- `tslib` ran `run.py --task smoke` successfully with outputs isolated under `results/smoke`; Exp4 smoke explicitly skipped LaTeX figure copying.
- `tslib` ran `experiments/generate_paper_artifacts_v2.py` successfully after experiment outputs existed.
- `tslib` reran `experiments/generate_paper_artifacts_v2.py` after adding dynamic formal-SOTA aggregation and `table_sota_physical_units.tex`; the generated tables now note that 256-step values and Cartesian physical RMSE/MAE require formal Exp1 reruns.
- `tslib` ran read-only `ast.parse` checks on `experiments/generate_paper_artifacts_v2.py`, `experiments/exp1_sota/SOTA_comparison.py`, and `data_generation/data_generator.py`; all parsed successfully.
- `tslib` reran `pytest tests/test_protocol_utils.py tests/test_paper_artifacts.py tests/test_model_provenance.py`; 18 tests passed after the 256-step/physical-metric support changes.
- `tslib` previously regenerated a diagnostic dataset with `HGV_PRED_LEN=256` and `HGV_PREDICTION_HORIZONS=32,64,128,256`; dataset validation passed with `trajectory_level_v1`, disjoint trajectories, finite values, and no physical violations.
- The current code default has been upgraded to the formal 1 Hz protocol: `HGV_TRAJECTORY_COUNT=1000`, `HGV_POINTS_PER_TRAJECTORY=1000`, `HGV_SAMPLING_INTERVAL_S=1.0`, `HGV_SEQ_LEN=256`, `HGV_PRED_LEN=256`, and `HGV_WINDOW_STRIDE=5`.
- `tslib` verified GPU execution on NVIDIA GeForce RTX 4090 with PyTorch 2.5.1+cu121 and CUDA enabled.
- `tslib` ran Exp1 seed=42 for Transformer, PLGAFormer, and Kalman at 32/64/128/256 horizons using `HGV_TRAIN_MODE=max_perf`. PLGAFormer did not beat Transformer in this diagnostic 256-step setting.
- `tslib` regenerated `experiments/paper_artifacts_v2/`; `table_sota_comparison.tex` now includes 256-step current Exp1 values, and `table_sota_physical_units.tex` reports `RMSE (m)`, `MAE (m)`, `FDE (m)`, and `ADE (m)`.
- `tslib` ran `experiments/exp3_robustness/visualize_results.py` successfully and refreshed robustness figures from formal Exp3 JSON.
- `tslib` compiled `Submit_AST/Manuscript/Manuscript.tex` and `Submit_AST/Anonymous Manuscript/Anonymous Manuscript.tex` twice with TeX Live 2026; both PDFs were produced. Remaining warnings are float-placement and minor overfull/underfull boxes, not compile failures.
- `tslib` reran `pytest tests/test_protocol_utils.py tests/test_paper_artifacts.py tests/test_model_provenance.py`; 18 tests passed.

## Closeout Definition

This project should be considered release-complete only after:

1. The target template is confirmed and, if necessary, migrated from Elsevier CAS to the required Transactions template.
2. The release branch and local release tag are pushed to GitHub.
3. Final dataset/checkpoint inclusion decisions are recorded in `PublicRelease/README.md`.
4. A final clean git status is confirmed after packaging generated manuscript files intentionally.
