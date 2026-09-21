# AI ECG phenotype and post ECG ICU transfer extension

Date: 2026-09-07. Status: analysis specification recorded before new association estimates. This is an author-requested post-review exploratory extension. Existing mortality results have already been examined. It is not a retrospectively registered primary analysis.

## Author steering before association estimation

The author explicitly requested retaining the original missing-data handling and discussing any concerns later. Formal M1/M2 analyses will therefore use the existing processed parent-table values without new filling or replacement. Earlier raw-triage reconstruction proposals are preserved as superseded decisions in DEVIATIONS.md. The raw missingness audit is retained only for later author review and will not be inserted as an agreed manuscript correction. The proposed additional heart-rate sensitivity is deferred. The author also requested identifying and exporting ICU event tables through Navicat Premium to assess ventilation and vasopressor endpoints. Those additional endpoints will be specified and analyzed separately after source verification; they do not change the phenotype or ICU-transfer estimands below.

## Scope and unchanged evidence

Use the existing internal test cohort and original patient partitions. Preserve all source data, model weights, mortality labels, calibration, and thresholds. No new model training or external inference. The manuscript base is the scientific revision v1, not the original submission. New endpoint-specific subsets and exclusions will be reported separately and will not overwrite the parent cohort.

The clinical question is whether the fixed ECG-only mortality marker is associated with focused current-encounter phenotypes beyond age, sex and initial vital signs, and whether fixed ECG-only and fusion mortality scores identify subsequent observed ICU transfer in the linked care episode.

## Gate 1 and 2 decisions

The phenotype analysis is a targeted current-encounter diagnostic association analysis, not a full PheWAS or a study of incident diseases. It adds clinical characterization of the existing marker. The six groups are acute myocardial infarction, heart failure, atrial fibrillation/flutter, sepsis, respiratory failure, and acute kidney injury. Use explicit ICD-9-CM and ICD-10-CM definitions, raw ED/hospital diagnosis dictionaries, same-encounter links, and a source-to-label map. Store complete code lists and source-code matches. Missing diagnosis ascertainment is not a negative phenotype. Do not call discharge codes baseline comorbidities or mechanistic ground truth.

Raw ED, hospital diagnosis, admission and transfer tables are locally available. ICU transfer is the care-escalation endpoint covered by this specification. Mechanical ventilation and vasopressor initiation are a separate author-requested extension requiring verified source events, definitions, and a separate multiplicity family. ICU transfer is a care-escalation proxy, not proof of new physiological deterioration or preventable mortality.

## Score provenance and transforms

Primary ECG-only marker: the saved ten-run ensemble corresponding to the original ECG-only discrimination benchmark, subject to verified generation provenance, patient partition consistency and row matching. The separate saved single-run ECG-only scores are a prespecified sensitivity analysis, never selected according to the new results. Fusion: the same ten-run mean used in revision v1.

Use clip epsilon 1e-8, logit-transform scores and scale with the mean and population SD of the corresponding original training partition. Do not choose cut points from the new outcomes. Descriptive risk strata use the original fusion raw-score thresholds 0.456303122639656 and 0.6162797212600708. These remain mortality-score strata, not ICU probability thresholds.

## Phenotype estimands

Estimate current-encounter phenotype odds ratios per one training-SD increase in logit ECG-only score in test encounters with ascertainable diagnoses. Associations need not be causal or independent of all disease severity.

M1 adjusts for age and sex. M2 (primary) adds initial temperature, respiratory rate, oxygen saturation and systolic blood pressure. Use the existing processed clinical values in data/ecg_muti.csv, preserving the original study's missing-data handling without raw-value replacement, reconstructed observation masks, or new imputation. Age and each vital sign use a three-knot restricted cubic spline at training 10th, 50th and 90th percentiles of these retained values where these are distinct; if a variable has coincident quantiles use a prespecified linear term and record that fact. Records identified as having unknown age/sex by the existing demographic fields are excluded from adjusted analyses and counted. Do not adjust for the phenotype under study or for diagnoses assigned later in the same encounter. Raw-triage missingness recovery remains audit-only and the proposed heart-rate sensitivity is deferred.

Use logistic GLM and patient-cluster robust covariance. Report n, event counts, OR, 95% CI, nominal p and Benjamini-Hochberg q across the six primary M2 phenotype tests. Display M1 and M2 together without choosing a favorable model. If fewer than 30 outcome events or model separation/convergence failure occurs, do not force adjusted inference; report counts and the limitation.

Prespecified sensitivities: (a) first chronological parent-test encounter per patient, restricted to ascertainable records; (b) single-run ECG-only marker with the same M2 specification. Parallel fusion M2 associations are descriptive secondary results and do not establish waveform-specific information. No pooled shared factor or new disease-model training.

## Post ECG ICU transfer estimands

Time zero is the recorded index ECG acquisition time. Identify ICU units from the raw transfer careunit dictionary and preserve the complete list. Exclude any record already in an ICU or with an earlier ICU entry in the same linked episode. Require a valid index timestamp, a deterministic encounter/hospital link and ascertainable episode disposition. Do not attach transfers from other admissions using patient ID alone.

Primary endpoint: first observed ICU entry strictly after the index ECG and within 24 hours, before the linked episode ends. Secondary endpoint: the analogous 72-hour entry. Count death and terminal ED/hospital discharge before ICU separately. If a recorded death timestamp is later than terminal discharge, preserve the earliest terminal boundary for ICU-event ascertainment and report a terminal-timestamp-conflict category separately from discharge. These are competing episode outcomes; discharge does not prove absence of later out-of-hospital deterioration. Records with unresolved follow-up are excluded, not made negative. All denominators and exclusions must be explicit.

Report event counts and cumulative observed proportions by the unchanged fusion mortality-score strata, with patient-cluster bootstrap 95% intervals (2,000 resamples; seed 20260907 for all eligible encounters and 20260908 for the landmark subset). For discrimination, use seeds 20260917 and 20260918 for 24 and 72 hours, respectively, with all fixed markers evaluated within the same patient resamples. Record the requested and effective number of bootstrap replicates for each result. Assess associations per one training-SD logit score using the same M1 and M2 covariates. The principal ICU association is ECG-only M2 at 24 hours; 72-hour, fusion and sensitivity estimates are secondary. Report the two ECG-only M2 ICU p values with BH q within their two-test family, without labelling any result confirmatory.

Discrimination of already-fitted ECG-only, fusion and the existing clinical plus conventional ECG XGBoost mortality comparator is descriptive: AUROC and average precision with patient-cluster resampling. They are fixed mortality markers applied to a different outcome, not separately developed ICU-prediction models. Do not evaluate their mortality probability calibration as ICU calibration or select a new ICU threshold.

Sensitivity: restrict to encounters remaining alive and in the observed linked episode, with no ICU entry, at ECG plus 60 minutes; assess subsequent transfer until the original 24/72-hour horizons. This reduces inclusion of nearly simultaneous ICU triage decisions but does not establish causation. Also describe the first-encounter-per-patient subset. Do not reinterpret mortality false positives as true-positive mortality predictions if other adverse events occur.

## QA, figure and manuscript contract

Before association estimation, verify raw source checksums, all one-to-one joins, patient partitions, chronology, episode boundaries, duplicate encounter contributions, code coverage and event counts. Implement clinical definitions in code rather than manual dataset edits. Record any source-driven change below before estimating new associations.

Run end-to-end from immutable inputs and perform an independent derivation/statistical spot-check. Deliver machine-readable aggregate tables, reproducible scripts, source hashes, a narrow result lock and a Chinese change note. Individual rows remain local research data and are not a public release.

Figures use the established R workflow, white backgrounds and vector PDF/SVG plus 600-dpi PNG. Figure S6 will display the six phenotype association estimates with demographics and additional vital-sign adjustment. Figure S7 will display 24/72-hour observed ICU transfer by original mortality-score strata with event counts and uncertainty. Plots must follow the estimates including null results. Full tabulations will extend the existing supplementary tables.

Add concise Methods/Results/Discussion/Limitations text and the new supplement sections to a versioned v2 clean manuscript and supplement. Preserve existing original figures, authorship and unrelated results. Provide corresponding red-marked copies consistent with the v1 convention; identify colour marking as distinct from native Word revisions. Render all four documents and verify relevant numbers, references, table/figure numbering and layout before delivery.
