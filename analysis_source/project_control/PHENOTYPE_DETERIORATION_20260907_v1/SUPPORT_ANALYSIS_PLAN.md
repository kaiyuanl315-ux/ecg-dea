# Supplementary recorded organ-support analysis specification

Date: 2026-09-07. Prepared before examining any new organ-support associations. This is an author-requested exploratory extension of the existing ICU-transfer supplement. The parent mortality and newly added ICU results have already been examined. No new model development or incident-treatment claim is planned.

## Population and estimand

Retain all 35,948 original test rows and row_id. The parent risk set is `derived/icu.csv.gz:eligible==1`, excluding already/current/prior ICU and unresolved index episode links as in the recorded ICU-transfer plan. Define separate invasive-ventilation and vasopressor observability/eligibility masks. Exact subject_id+hadm_id+ICU stay_id links are required. Do not coalesce the 20 ambiguous ED-to-hospital links or join later admissions by subject_id.

The endpoints are the first support events with occurrence times strictly after the ECG and within ECG+24h (primary) or ECG+72h (secondary), strictly before the linked episode terminal boundary, documented in the ICU clinical information system. This is a recorded-support endpoint, not proof of true initiation after ECG. Charttime/starttime is the recorded occurrence time; storetime is retained to audit later entry of the record. Do not label absence of an ICU-system record as absence of ED, prehospital or ward treatment.

For a full observed episode without ICU records, zero means no recorded ICU-system support in that episode. If an observed ICU transfer interval cannot be represented by the exported icustays dimension, the episode is unobservable for this extension and excluded rather than assigned a negative support outcome. Actual missing source records and incomplete exports must be resolved before result lock. A restricted description among ICU-observed encounters may be added, clearly conditional on subsequent ICU care.

Exclude known prior or cross-ECG corresponding support within the linked admission. For ventilation, raw classified invasive ventilation/tracheostomy state or explicit pre-ECG invasive ventilation/intubation/tracheostomy procedure is a conservative baseline airway/support exclusion; these mechanisms are tabulated separately. Lack of pre-ECG records means baseline support was not recorded, not that its absence is proven. For vasopressors, any earlier valid five-drug continuous infusion is a baseline exclusion, including intervals crossing ECG and rate-change segments connected to an earlier infusion.

## Ventilation definition

Rebuild ventilator_setting, oxygen_delivery and ventilation using the pinned official SQL logic, with pure local computation and no database edits. Main event status is exactly `InvasiveVent`; `Tracheostomy` alone is not combined with it. Preserve the official category priority, device strings including significant trailing spaces, multiple oxygen devices, and source handling of simultaneous observations. The 14-hour rule uses differences between timestamps truncated to hour boundaries, as confirmed in the official generated PostgreSQL SQL. Preserve `HAVING MIN(charttime) != MAX(charttime)`, which removes a state episode represented at only one charttime.

The pinned oxygen_delivery SQL filters `ce.rn=1` after its full outer join, so device-only timestamps without an oxygen-flow record are not retained. Reproduce that behavior, quantify its consequences, and retain a raw-device/setting classification sensitivity that does not require an oxygen-flow record. Do not silently fix the official algorithm and label it unchanged. The main result follows the pinned official reconstruction; single-state/device-only and explicit post-ECG invasive procedure signals are supplementary QA/sensitivity markers, not substituted based on favorable associations.

Compare rebuilt episode tuples (stay_id,starttime,endtime,status) with the existing exported derived ventilation table and report exact overlaps and mismatches by status. Existing table names do not establish their generating SQL version. No acceptance of equivalence solely from similar event counts.

## Vasopressor definition

Include norepinephrine 221906, epinephrine 221289, phenylephrine 221749, vasopressin 222315 and dopamine 221662. Exclude dobutamine and milrinone from this endpoint. Project-specific delivery cleaning (not a claim about the official single-drug SQL): rate>0, nonmissing start/end, end>start and statusdescription not Rewritten. Preserve all raw rows and tabulate each excluded reason and original status. Positive-amount bolus/missing-rate rows are not automatically treated as continuous infusion. Use actual rate, not originalrate.

Merge overlapping or directly contiguous valid same-drug intervals within a linked admission to prevent dose changes from being interpreted as new treatment starts; retain orderid/linkorderid and raw segment counts for QA. Binary recorded support does not depend on converting rate units, but tabulate actual units; apply only the explicitly verified official conversions when reporting normalized doses. Unknown units or invalid body weight cannot support dose-standardized comparisons. Report drug composition, and retain a sensitivity excluding dopamine or requiring dopamine>=5 mcg/kg/min as a separate research definition. Epinephrine bolus is distinct from ongoing support.

## Boundaries, sensitivity, and output QA

Support at or before index is not a post-index event. If the first post-index support time equals the terminal boundary, exclude that endpoint as a timestamp ambiguity; ignore and count source support entries later than terminal. A 60-minute landmark excludes any earlier corresponding support, death or terminal episode exit and preserves original ECG+24/72h horizons. Reapply the same baseline exclusion rules at ECG+60min: for IMV this includes raw invasive/tracheostomy states and pre-landmark intubation/invasive-ventilation/tracheostomy procedures, even when the official duration algorithm has not yet created a retained episode.

The preparation script will not fit associations. It will write source hashes, code/source versions, parsing and join QA, ICU coverage gaps, prior-support exclusions, raw-derived ventilation comparison, valid infusion segment/episode counts, outcome flows, and a local row-level dataset. Source exports require explicit completeness QA against query counts before inferential use. Current script generation and synthetic boundary checks are preparation only, not completed patient analysis.

## Source package

Official text and hash manifest: `qa/mimic_official_sources`. Code commit `303d26c623dcc9c49cc0f204468d4acc2f063797`; docs commit `81822278432bc33e101116427e05fa4f69265a01`. Input filenames: `data_exports/icustays.csv`, `chartevents_resp.csv.gz`, `inputevents_pressors.csv.gz`, `procedureevents_support.csv.gz`, `ventilation.csv.gz`.
