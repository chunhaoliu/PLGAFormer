# Multi-regime manuscript synchronization record

Date: 2026-07-30
Historical dataset protocol: `hgv_multiregime_state_v2`

This record is frozen provenance.  The active manuscript synchronization is
documented in `MULTIREGIME_V21_DATA_PROTOCOL.md`.
Dataset SHA-256:
`e5fc39c39cb8bec703d2bdc78c5278430b684ae130622edb8b64f19c473cb1b5`

## Completed synchronization

- Updated the abstract and contributions to describe 1,200 complete
  trajectories, two vertical regimes, three primary maneuver labels, randomized
  controls, and trajectory-level separation.
- Replaced the fixed-control dataset description with the factorial
  quasi-equilibrium/skip-glide control protocol.
- Updated the physical integrator to DOP853 with a 4 s internal maximum step
  and 1 s stored output interval.
- Replaced the old three-class dataset figures with:
  - a six-panel vertical-regime--maneuver representative grid;
  - a complete-artifact control, operating-envelope, and physical-gate figure.
- Updated the formal counts to 1,200 complete trajectories,
  960/120/120 complete-trajectory splits, and
  15,360/11,760/11,760 supervised windows.
- Updated the experimental ceiling to 50 epochs with validation early stopping.
- Removed every rendered result, significance, robustness, convergence, and
  efficiency value inherited from `trajectory_level_v1`.
- Preserved the legacy complete manuscript under
  `Archive/Manuscript_Snapshots/TAES_trajectory_level_v1_before_multiregime_20260730/`.
- Reduced the active `Main_Manuscript/figures/` directory to the six PDFs
  actually consumed by the compiled manuscript.
- Removed LaTeX build intermediates from the submission-facing directory.

## Compilation and visual QA

- TeX Live 2026 `latexmk` completed successfully.
- Bibliography and cross-references resolved.
- Final log contained no overfull boxes and no undefined references.
- The nine-page PDF was rendered at 120 dpi and reviewed as a page contact
  sheet; the two new double-column figures are readable at final layout size,
  panel labels and captions remain inside bounds, and no blank or orphan page
  was observed.

## Artifact hashes

- Main TeX:
  `9db8d36ec354a1c725f3a37d2a58572721d31a1561200bcc542578a4d8c70593`
- Compiled PDF:
  `6d07fb55980e39a38ea3ca6298de9db29086a4d16bebcae58621ffd85ad5ac0f`
- Representative-trajectory figure:
  `efce58ab71fa28780294e0cf8a3eae7c52cdae9438730062e4b0283740e8c7ca`
- Dataset-characterization figure:
  `3f30b0727c2b5c322917b03b89383a7e84da5eba55bebd7097832bbb9bf75f1e`

## Evidence boundary

The manuscript is now data-protocol consistent but intentionally incomplete as
a submission: Main Results and ablation values are absent. No formal model
training was launched in this data-and-manuscript phase. The next phase may
start only from the frozen dataset hash above and must populate numerical
claims from new run manifests rather than the archived legacy manuscript.
