# Reproducibility and source scope

This supplement exposes the scientific implementations and selected aggregate results used in the September 2026 manuscript revision. The original main-cohort workflow remains in `src/`, `notebooks/`, and `model/`. The additions are source snapshots for inspection, not a validated standalone execution package.

## Analysis map

| Manuscript component | Implementation under analysis_source | Aggregate results |
| --- | --- | --- |
| Calibration, validation-selected thresholds and alert burden; Methods 1.5, Tables S6–S7, Figures S2 and S4 | `project_control/REVISION_ANALYSIS_20260906/analysis.py` | `aggregate_results/calibration/` |
| Mortality comparators and paired uncertainty; Methods 1.6, Tables S8–S10, main Figure 2 | `project_control/REVISION_ANALYSIS_20260906/comparators.py` and `paired_comparison.py` | `aggregate_results/comparators/` |
| Current-encounter phenotypes; Methods 1.7, Tables S11–S12, Figure S5 | `project_control/PHENOTYPE_DETERIORATION_20260907_v1/scripts/derive_phenotypes.py` and `analyze.py` | `aggregate_results/phenotypes_and_care/` |
| Post-ECG ICU transfer and recorded organ support; Methods 1.8–1.9, Tables S13–S16 | The same source folder: `derive_icu.py`, `derive_support.py`, `analyze_support.py` | `aggregate_results/phenotypes_and_care/` |
| Cardiac-diagnosis restriction and mortality-negative subsequent care; Methods 1.10–1.11, Tables S17–S18, main Figures 3–4 | `project_control/FOCUSED_ROBUSTNESS_20260907_v1/scripts/` | `aggregate_results/focused_analyses/` |
| Conditional ECG integrated gradients and QRS alignment; Methods 1.12, Figure S8 | `tmp/med_ig_20260915/` and `tmp/med_ig_risk_C_20260916/` | `aggregate_results/interpretability/` |

Original script and folder names have been retained to support provenance. Some historical figure labels differ from final manuscript numbering; use this map. The source snapshots include selected independent checking scripts. Inclusion of a historical checking script does not mean that its private inputs are distributed.

## What was preserved

Scientific calculations, thresholds, model settings and seeds in the selected source files are unchanged. Personal absolute path prefixes were replaced with `/path/to/research`. The two IG model classes were extracted exactly from the recovered architecture used by the original analysis and are supplied in `analysis_source/model_architecture_for_ig.py`. They were not replaced with a newly reconstructed architecture.

`SOURCE_MANIFEST.json` maps every copied or extracted source/result file to its origin and records the original and prepared SHA-256 digests. `SHA256SUMS.txt` covers the prepared repository additions, excluding that checksum file itself. Source digests identify file versions; they do not assert that the entire study has a single result lock.

## Inputs needed for execution

Re-execution belongs in a separately configured, authorized research workspace. The source snapshots retain their original relative hierarchy. Typical inputs are the processed main-cohort table, fixed patient splits, saved Fusion-DL and ECG-only predictions, ECG arrays and clinical inputs, authorized MIMIC clinical tables, and the retained model checkpoints. Some statistical scripts also require the prepared external-cohort prediction table. This package does not provide external-cohort waveform processing or certify its upstream score provenance.

The phenotype/support and focused analyses additionally depend on derived encounter tables, locally generated source-export manifests, parent result locks and plan-freeze records. The IG pipeline additionally depends on the original sample metadata, input arrays, numerical prediction gates, overlap-reuse metadata, and blinded per-record QRS review decisions. These items may contain individual records or mappings and are not included. Even anonymized display columns for individual encounters in Figure S8A are excluded.

Before execution, configure the placeholder paths, supply the exact authorized inputs, and restore or regenerate the relevant manifests in a separately versioned research workspace after verifying input identity. Point the IG architecture reference to the supplied exact class definitions where appropriate. Regenerated manifests must record the changed source-file digests. Do not delete provenance assertions or treat a failed hash assertion as a software nuisance. An exact historical replay requires the historical inputs and review decisions; the public supplement alone cannot establish it.

## Environments and checks

`requirements-supplement.txt` lists additional imported Python packages; it is a dependency list, not a fully pinned or newly tested environment. The original base requirements are preserved. The focused statistical QA recorded Python 3.13.1, NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1 and statsmodels 0.14.6. These are historical observations rather than a cross-platform compatibility guarantee. The R scripts use packages including ggplot2, patchwork, jsonlite, data.table and sandwich. Deep-learning and IG execution additionally requires compatible torch, tsai and NeuroKit2 environments.

Preparation checks cover source-copy fidelity, Python/R syntax, selected result-lock digests, aggregate table structure, key manuscript-number agreement, and accidental sensitive-file/credential inclusion. The study was not retrained or rerun during preparation. The bundle is suitable for code inspection and aggregate-result inspection, with the execution dependencies described above.

## Interpretation

The calibration, comparator, phenotype, subsequent-care and attribution work includes exploratory analyses. Arrival-based 72-hour mortality and ECG-based care horizons are distinct. Phenotype associations and model attributions do not establish causal mechanisms, and retrospective threshold analyses do not establish clinical benefit.
