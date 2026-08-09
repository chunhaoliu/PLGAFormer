# TAES Submission Artifact Staging

This directory is the canonical staging area for generated TAES tables,
figures, and evidence manifests.

Experiment and artifact-generation scripts may update `generated/`. The clean
submission snapshot under the outer research workspace:

`HGV_Project/Init_Submit_TAES/Main_Manuscript/`

must not be used as a live experiment-output directory. Copy only the final,
manuscript-used figures or inline the final tables into the submission TeX,
then compile and verify the clean snapshot independently.

The generated artifacts retain the HGV formal-protocol boundary: complete
trajectories are split by trajectory ID before 256-step input/output windows
are created, and formal multi-seed claims use seeds 42, 123, and 456.

`generate_taes_dataset_figures.py` recreates the reusable dataset and maneuver
views from `data_generation/data/processed/raw_hgv_trajectories.npz`. It
validates the 1 Hz, 1000-point protocol and maneuver counts before exporting:

- three independently composed, velocity-coded 3-D representative trajectories;
- one shared trajectory-projection/endpoint key and speed scale;
- one bank-command comparison and one maneuver-resolved speed-response panel;
- one shared maneuver key;
- the altitude--speed operating envelope formed from time-downsampled points
  from all 1,000 complete trajectories;
- `dataset_figure_manifest.json` with source/output hashes and representative IDs.

Do not substitute the similarly named AST-era figures because those files
predate the formal raw artifact and use a different display range and
trajectory duration.
