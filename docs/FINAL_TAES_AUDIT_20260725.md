# Final TAES Evidence Audit (2026-07-25)

## Verdict

The current TAES manuscript is locally submission-ready at the evidence,
implementation, and typesetting levels. All numerical claims in the active
manuscript are generated from the formal 1 Hz trajectory-separated protocol.
The remaining pre-submission actions are administrative: confirm author and
funding metadata, select the final submission date, and push the intended
public release revision.

## Canonical Deliverables

- Source: `../Init_Submit_TAES/Main_Manuscript/IEEEtaes_Manuscript.tex`
- PDF: `../Init_Submit_TAES/Main_Manuscript/IEEEtaes_Manuscript.pdf`
- Main evidence: `experiments/taes_submission_artifacts/generated/main_results_summary.json`
- Secondary evidence: `experiments/taes_submission_artifacts/generated/secondary_results_manifest.json`
- Qualitative evidence: `experiments/taes_submission_artifacts/generated/prediction_analysis_manifest.json`

## Evidence Identity

- Formal Exp1 base signature:
  `9bab50725adad47955de0839d9f3988ec2e9b08cc0d94e5b0e11f3bcabe52f50`
- Final-model evidence signature:
  `22dc4a08d385f1145a4bd29951a2694e2709ab36fd03cf234512aac4b7972392`
- Final architecture: rotating-Earth prior fusion without the channel residual.
- Seeds: 42, 123, and 456.
- Final checkpoint SHA-256 prefixes: `7507e666`, `0bfc36b7`, and `44b72fad`.

The final-model resolver is `utils/final_plgaformer.py`. It rejects mixed
architectures, missing records, seed mismatches, and checkpoint-hash drift
before downstream figures or tables can be generated.

## Formal Findings

- At 256 s, PLGAFormer obtains `1.509 +/- 0.183 km` ADE and
  `4.268 +/- 0.558 km` FDE.
- Relative to AF-CILN, the strongest neural comparator at 256 s, the reductions
  are 63.5% in ADE and 50.8% in FDE; both paired trajectory-level comparisons
  remain significant after Holm correction (`p = 0.003`).
- The rotating-Earth 3-DOF model has the best 32 s ADE. PLGAFormer has the best
  32 s FDE and both best metrics from 64 s onward.
- Prior-only fusion wins the validation-stage structural ablation and is the
  final selected architecture. The unused channel residual is not presented as
  part of the proposed model.
- PLGAFormer remains below 1.75 km ADE under both controlled simulator-parameter
  shifts.
- The nominal unfiltered sensor-noise test is an explicit failure case
  (`28.011 km` ADE), not a robustness claim.
- Batch-one inference is `346.24 ms` for a complete 256 s forecast on an RTX
  4090, including analytical propagation.

## Verification

- Full repository test suite: `117 passed`; the focused protocol and artifact
  subset accounts for 40 of these tests.
- LaTeX: successful BibTeX plus repeated pdfLaTeX compilation.
- PDF: 11 letter-size pages.
- Log: zero overfull boxes, undefined references, undefined citations, or LaTeX
  errors.
- References: 26 journal articles, 26 unique DOI fields, years 2021--2026, and
  no unresolved metadata in `REFERENCE_VERIFICATION_20260722.md`.
- Visual QA: all 11 pages checked; no clipping, overlap, blank columns, or
  post-conclusion experiment floats.

## Deliberate Boundaries

- Simulation evidence only; no flight-test or operational radar data.
- The analytical proposal assumes equivalent controls remain constant during
  each forecast.
- Clean or state-estimated tracks are required; a dedicated estimator is not
  part of this paper.
- The checked-in `IEEEtaes.cls` identifies internally as `IEEEphot`. Its hash
  matches the laboratory TAES reference template, so it is retained, but the
  final files downloaded from the TAES author center should be compared before
  upload.
