# Aggregate result guide

These tables are selected from the verified local analysis packages and mapped to the supplied September 2026 manuscript in `../REPRODUCIBILITY.md`. Rows represent estimates, groups, bins, model settings or lead-by-time summaries. They do not contain individual clinical records.

| Folder | Contents and units |
| --- | --- |
| `calibration` | Internal-test and primary external-score calibration bins, calibration metrics, risk strata and threshold metrics; validation-only calibration coefficients and thresholds |
| `comparators` | AUROC, average precision and calibrated Brier estimates and confidence intervals; paired Fusion-DL-minus-comparator differences; training-only hyperparameter search and aggregate missingness |
| `phenotypes_and_care` | Adjusted odds ratios, confidence intervals, P/q values, cohort counts, ICU discrimination and stratified recorded care proportions |
| `focused_analyses` | Restricted noncardiac phenotype associations, mortality-negative care proportions and paired absolute contrasts, and flow counts |
| `interpretability` | Mean lead shares, group-median QRS-aligned ECG waveforms, high-risk mean absolute IG, sampling stability and pointwise bootstrap intervals |

`external_article_saved` denotes the primary saved external score used by the supplied manuscript, not a newly reconstructed external prediction pipeline. The separate historical alternate-score sensitivity outputs are excluded. Comparator tables contain only the four primary models and their paired comparisons; original numeric values are unchanged.

Read explicit metric and effect fields before interpreting units. Mortality threshold sensitivities, specificities, predictive values and proportions are stored as fractions unless a column says `per_1000`. Odds ratios are unitless; model coefficients are log-odds coefficients. Care contrasts follow the table's effect field. Probability intervals are nominal unless the analysis specifies otherwise.

In the IG summaries, time is seconds relative to the QRS anchor and waveforms are in mV. Panel-B lead shares are percentages. High-risk mean absolute IG is conditional ECG attribution to the mean raw ensemble score with fixed clinical inputs and a zero-ECG reference; it is not attribution to the displayed group-median waveform or a high-minus-low attribution difference. The panel-C summaries include 507 high-risk and 1,022 low-risk encounters after QC. Pointwise intervals use the historical 300-replicate patient-cluster bootstrap and are not simultaneous bands.

The result files are unchanged copies except for selecting primary-model rows and selecting public calibration parameter fields. Provenance and transformations are recorded in `../SOURCE_MANIFEST.json`.
