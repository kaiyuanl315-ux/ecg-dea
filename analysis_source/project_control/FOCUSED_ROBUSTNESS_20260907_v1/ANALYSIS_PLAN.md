# Focused AI-ECG robustness and outcome-negative characterization

Version: 1.0; date: 2026-09-07. This is an author-requested exploratory extension of the already locked phenotype/care-escalation analyses. The parent results and Croon et al. were known when designing it. The present restricted-cohort effects and mortality-negative stratum proportions have not been inspected before this specification. A timestamped SHA-256 freeze record will precede their estimation.

## Question, scope, and feasibility

Question A: after excluding encounters with any recorded current-encounter acute myocardial infarction (AMI), heart failure (HF), or atrial fibrillation/flutter (AF), does the fixed ECG-only mortality score remain associated with coded sepsis/septic shock, respiratory failure, and acute kidney injury (AKI)?

Question B: among encounters not meeting the original 72-hour-from-ED-arrival mortality endpoint, what proportions in the unchanged low/intermediate/high fusion-score strata have observed ICU transfer or first ICU-system-recorded organ support within 24 hours of the ECG?

The immediate deliverable is a checked analysis package and an interpretation of its incremental value for the existing manuscript. Manuscript v2 and the parent analysis package will remain immutable. No new prediction-model training, score generation, cut-point optimization, calibration, raw-source export, or missing-data processing is required.

The parent package is `project_control/PHENOTYPE_DETERIORATION_20260907_v1`. Its status is `LOCKED_EXPLORATORY_EXTENSION`. Available immutable inputs are `derived/test_base.csv.gz`, `derived/phenotypes.csv.gz`, `derived/icu.csv.gz`, `derived/support.csv.gz`, `qa/input_qa.json`, `results/associations.csv`, and `RESULT_LOCK.json`. All files will be hashed, identity joins validated, and relevant parent estimates reproduced before accepting new results.

Feasibility, before new effect estimation: the phenotype-ascertainable cohort has 35,906 encounters. Excluding any of AMI/HF/AF leaves 26,239 encounters, including 778 coded sepsis, 747 respiratory failure, and 2,338 AKI records. These marginal counts support proceeding. Parent test N is 35,948 with 302 original mortality-positive encounters. Endpoint-specific eligibility, sample sizes, and exclusions for Question B will be enumerated after source verification.

## Unchanged data and variable roles

- Encounter is the analysis unit; patient is the clustering unit. Patient-level original training/validation/test assignments remain unchanged.
- The ECG-only ten-run score is the primary continuous marker for A. Its logit transform uses clipping at 1e-8 and the original training mean/SD. The saved single-run ECG-only marker is a prespecified sensitivity, not a replacement selected by results.
- B uses original raw fusion mortality-score strata: Low <0.456303122639656; Intermediate >=0.456303122639656 and <0.6162797212600708; High >=0.6162797212600708. They are not probabilities or treatment thresholds for care escalation.
- Preserve original processed age, sex, temperature, respiratory rate, oxygen saturation and systolic pressure, original demographic eligibility, and original training-derived spline transforms exactly. Do not refill, reconstruct missingness, substitute raw vitals, add heart rate, or update the manuscript's missingness description.
- In A, age/sex and initial vital signs provide adjusted clinical characterization. No causal sufficiency is claimed. Concurrent cardiac diagnoses define a restricted population; they may reflect shared disease, downstream illness, or coding/selection processes. Restriction therefore does not identify a causal direct effect or a population confirmed free of cardiac disease.
- In B, the original mortality-negative label is a post-index selection variable and may induce selection/collider bias. This is descriptive characterization, not a prospective validation cohort and not evidence of prevented deaths or successful treatment.

## A. Concurrent cardiac-diagnosis restriction

Population: original test encounters with `diagnosis_observed==1`. All six coded phenotypes must be observed; absence of diagnosis ascertainment is never treated as a negative label.

Principal restriction: `ami==0 & heart_failure==0 & af_flutter==0`. Outcomes, each analyzed separately: sepsis, respiratory_failure, aki. Retain all other parent eligibility rules.

Principal model: logistic GLM, ECG-only score per one original-training SD of logit score, M2 covariates and original training-defined splines, intercept, and patient-cluster sandwich SE with the same finite-sample correction as the parent study. Report encounters, patients, events, OR, 95% CI, nominal p and BH q across exactly these three M2 tests. Label them principal exploratory results, never confirmatory tests. Reproduce the three full-cohort parent M2 results as references, with their original six-test-family q values.

Prespecified secondary checks, all shown regardless of direction: (1) principal restriction with M1 age/sex only; (2-4) M2 excluding AMI alone, HF alone, or AF alone; (5) M2 with the principal restriction among first parent-test encounters; (6) M2 principal restriction with the saved single-run ECG-only score. Select the first encounter chronologically from the full parent test population before any phenotype restriction. These sensitivity results receive nominal CIs and p values; no claim is selected by the most favorable model. There are 21 new fits plus three parent-reference reproductions.

Do not perform new disease-model comparisons, formal cross-phenotype ranking, post-hoc subgroup selection, or a causal attenuation/mediation calculation. Full and restricted estimates will be compared descriptively with their sample composition. Null or attenuated results are informative and will be retained.

## B. Care escalation among original mortality-negative encounters

First apply `death_72h==0` to the locked parent label, without replacing it using newly observed timestamps. Refer to this population as not meeting the original mortality endpoint, not as proven 72-hour survivors under a new ECG-based definition. Mortality starts at ED arrival; care-escalation horizons start at ECG acquisition. The two clocks must remain explicit.

Use the exact parent endpoint-specific eligibility and binary labels: `eligible` for ICU transfer; `imv_eligible` and `pressor_eligible` for support. Inherit all baseline-support/ICU exclusions, chronology and linkage checks, episode boundaries, and uncovered-ICU-source exclusions. No diagnosis-ascertainment requirement is added to B.

Principal horizon: 24 hours after ECG for ICU transfer, first ICU-system-recorded invasive ventilation, and first ICU-system-recorded vasopressor support. ICU transfer is the principal clinical narrative endpoint; the two support outcomes provide corroborating descriptions. Events need not be mutually exclusive. No composite endpoint will be created.

For each endpoint report total and stratum-specific encounter and patient counts, events, observed proportions, nominal 95% CIs, and paired absolute proportion differences (High minus Low and Intermediate minus Low) with CIs. Proportion differences will be displayed in percentage points. Do not report a new AUROC, p-for-trend, alert net benefit, calibrated probability, or adjusted survivor-only association model.

Uncertainty: 2,000 nonparametric patient-cluster bootstrap draws per endpoint-specific eligible population; sample as many patients as are eligible, with replacement, retaining all their selected encounters and recomputing all three strata and paired contrasts in each draw. Percentile 2.5/97.5 intervals; report requested/valid replicate counts. Seeds: `202609070 + 10 * subset_index + endpoint_index`, where subset index is 0=principal24, 1=secondary72, 2=first_parent24, 3=landmark60_24; endpoint index is 0=ICU, 1=IMV, 2=pressor. No multiplicity-adjusted inference is claimed; CIs describe uncertainty for the complete prespecified set.

Secondary descriptions: (a) 72-hour events with original ascertainment boundaries; (b) 24-hour events among first parent-test encounters, selected before mortality and endpoint filtering; (c) 24-hour events using the parent's 60-minute landmark definitions, retaining the original ECG-based 24-hour horizon. ICU landmark requires no ICU entry; each support landmark requires absence of the corresponding recorded support and re-applies baseline support exclusions, without additionally requiring ICU-free status.

Discharge/death before the endpoint closes observed episode ascertainment and must not be described as verified absence of later events outside the episode. First ICU-system recording cannot establish actual treatment initiation; missing ED/prehospital/ward support remains a limitation. Reconcile parent mortality labels with recorded episode-death timestamps as an audit only, separately for the arrival- and ECG-based windows. Do not correct labels silently. Any material inconsistency limits wording rather than authorizing new label processing.

## QA and interpretation rules

Before estimation: validate hashes against the parent lock; row/file/patient identity and one-to-one joins; unchanged labels, scores, processed covariates and strata; all retained binary endpoints; deterministic parent-first-encounter selection; sample flow and missingness; endpoint nesting and support coverage; and subgroup counts. Local participant-level artifacts remain within the research workspace.

For each GLM require both cases and noncases >=30, full-rank design, convergence, finite coefficients/SEs and no complete-separation warning. Record condition number, score range, iterations and warnings. If these fail, suppress that adjusted estimate and report counts; do not search for an alternative model. Check extreme fitted probabilities as a diagnostic without changing the model automatically. No score-linearity claim beyond the fixed parent specification is added.

Independently rebuild restricted cohorts and refit all 24 GLMs in R using natural splines and an explicit patient-cluster sandwich; reconcile counts, coefficients, SEs, CIs and q values with Python. Independently verify B numerator/denominator construction and every stored bootstrap draw using a separate aggregation implementation; cross-check point estimates in R. Repeat the full new pipeline into a separate output folder and compare deterministic tables. Preserve source and manuscript hashes before/after.

Freeze a narrow result lock only after successful QA. Deliver the frozen plan, source manifest, flow and diagnostic tables, complete results, deviations, Chinese interpretation, and figure/source-data files suitable for deciding manuscript placement. Results may support clinical characterization and reasons to study reassessment. They cannot establish a shared mechanism, a new disease-specific diagnostic test, actual post-ECG treatment initiation, prevention of death, or clinical benefit of an alert.

Interpretation is based on effect sizes, absolute proportions/differences, uncertainty, event counts, and consistency. Favorable p values are not a gate for retaining or reporting the analyses.
