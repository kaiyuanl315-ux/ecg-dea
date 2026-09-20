#!/usr/bin/env python3
"""Derive six current-encounter administrative phenotypes from immutable MIMIC tables.

No association models are fitted. Individual-level output is local research data.
The parent cohort and its original order are preserved; diagnosis absence is
represented separately from absence of a selected diagnosis code.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
RAW = Path('/path/to/research/mimiciv')
SOURCES = {
    'core_test': PROJECT / 'data/proba_test_72.csv',
    'cohort': PROJECT / 'data/ecg_muti.csv',
    'ed_stays': RAW / 'mimiciv-ed/edstays.csv',
    'ed_diagnoses': RAW / 'mimiciv-ed/diagnosis.csv',
    'hospital_admissions': RAW / '3.0/hosp/admissions.csv',
    'hospital_diagnoses': RAW / '3.0/hosp/diagnoses_icd.csv',
    'icd_dictionary': RAW / '3.0/hosp/d_icd_diagnoses.csv',
    'analysis_plan': OUT / 'ANALYSIS_PLAN.md',
    'script': Path(__file__).resolve(),
}

# Freeze before association estimation. "prefix" and "exact" are distinct.
DEFINITIONS = {
    'ami': {
        'label': 'Acute myocardial infarction',
        9: {'prefix': ['410'], 'exact': []},
        10: {'prefix': ['I21', 'I22'], 'exact': []},
    },
    'heart_failure': {
        'label': 'Heart failure',
        9: {'prefix': ['428'], 'exact': []},
        10: {'prefix': ['I50'], 'exact': []},
    },
    'af_flutter': {
        'label': 'Atrial fibrillation/flutter',
        9: {'prefix': [], 'exact': ['42731', '42732']},
        10: {'prefix': ['I48'], 'exact': []},
    },
    'sepsis': {
        'label': 'Explicitly coded sepsis/septic shock',
        9: {'prefix': ['038'], 'exact': ['99591', '99592', '78552']},
        10: {'prefix': ['A40', 'A41', 'R652'], 'exact': []},
    },
    'respiratory_failure': {
        'label': 'Respiratory failure, any',
        9: {'prefix': [], 'exact': ['51881', '51883', '51884']},
        10: {'prefix': ['J96'], 'exact': []},
    },
    'aki': {
        'label': 'Acute kidney injury/failure',
        9: {'prefix': ['584'], 'exact': []},
        10: {'prefix': ['N17'], 'exact': []},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ids(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='raise')
    assert np.equal(numeric.dropna(), np.floor(numeric.dropna())).all()
    return numeric.astype('Int64')


def timestamps(series: pd.Series) -> pd.Series:
    values = series.astype('string')
    slash = values.str.contains('/', na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')
    parsed.loc[slash] = pd.to_datetime(values.loc[slash], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    parsed.loc[~slash] = pd.to_datetime(values.loc[~slash], format='ISO8601', errors='coerce')
    return parsed


def normalize_codes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # Leading zeros are clinically meaningful for ICD-9 (for example 038.x).
    frame['icd_code'] = frame['icd_code'].astype('string').str.strip().str.upper().str.replace('.', '', regex=False)
    assert frame['icd_code'].notna().all()
    assert frame['icd_code'].ne('').all()
    assert frame['icd_version'].isin([9, 10]).all()
    return frame


def matches(frame: pd.DataFrame, definition: dict) -> pd.Series:
    matched = pd.Series(False, index=frame.index)
    for version in [9, 10]:
        rules = definition[version]
        is_code = frame.icd_code.isin(rules['exact'])
        if rules['prefix']:
            is_code |= frame.icd_code.str.startswith(tuple(rules['prefix']))
        matched |= frame.icd_version.eq(version) & is_code
    return matched


def member_keys(left: pd.DataFrame, right: pd.DataFrame, left_key: str, right_key: str) -> np.ndarray:
    left_index = pd.MultiIndex.from_arrays([left.subject_id, left[left_key]])
    right_index = pd.MultiIndex.from_arrays([right.subject_id, right[right_key]])
    return left_index.isin(right_index)


def main() -> None:
    for folder in ['derived', 'qa', 'results']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(SOURCES['core_test'], dtype={'file_name': str})
    full = pd.read_csv(SOURCES['cohort'], dtype={'file_name': str},
                       usecols=['file_name', 'subject_id', 'ed_stay_id', 'hadm_id', 'ecg_time', 'edregtime'])
    assert len(test) == 35948 and test.death_72h.sum() == 302
    assert not test.file_name.duplicated().any()
    assert not full.file_name.duplicated().any()
    core = test[['file_name', 'subject_id']].copy()
    core.insert(0, 'row_id', np.arange(len(core), dtype=np.int64))
    core = core.merge(full, on=['file_name', 'subject_id'], how='left', validate='one_to_one', indicator=True, sort=False)
    assert core._merge.eq('both').all()
    core = core.drop(columns='_merge')
    assert core.file_name.tolist() == test.file_name.tolist()
    for col in ['subject_id', 'ed_stay_id', 'hadm_id']:
        core[col] = ids(core[col])

    ed_stays = pd.read_csv(SOURCES['ed_stays'], usecols=['subject_id', 'stay_id', 'hadm_id', 'intime', 'outtime', 'disposition'])
    for col in ['subject_id', 'stay_id', 'hadm_id']:
        ed_stays[col] = ids(ed_stays[col])
    assert not ed_stays.duplicated(['subject_id', 'stay_id']).any()
    linked = core.merge(ed_stays.rename(columns={'hadm_id': 'ed_hadm_id', 'stay_id': 'ed_stay_id'}),
                        on=['subject_id', 'ed_stay_id'], how='left', validate='many_to_one', sort=False, indicator=True)
    linked['ed_link_observed'] = linked._merge.eq('both')
    linked = linked.drop(columns='_merge')
    conflict = linked.hadm_id.notna() & linked.ed_hadm_id.notna() & linked.hadm_id.ne(linked.ed_hadm_id)
    conflict = conflict.fillna(False)
    # Twenty raw ED hospital links concern a later episode: do not coalesce them.
    ambiguous_hospital_link = linked.hadm_id.isna() & linked.ed_hadm_id.notna() & linked.ed_link_observed
    backfilled = pd.Series(False, index=linked.index)
    linked['resolved_hadm_id'] = linked.hadm_id.copy()
    linked.loc[conflict, 'resolved_hadm_id'] = pd.NA

    # Verify that a retained hospitalization belongs to the same subject.
    admissions = pd.read_csv(SOURCES['hospital_admissions'], usecols=['subject_id', 'hadm_id', 'edregtime', 'admittime', 'dischtime'])
    for col in ['subject_id', 'hadm_id']:
        admissions[col] = ids(admissions[col])
    assert not admissions.duplicated('hadm_id').any()
    admission_linked = member_keys(linked, admissions, 'resolved_hadm_id', 'hadm_id')
    admission_invalid = linked.resolved_hadm_id.notna() & ~admission_linked
    ed_link_invalid = linked.ed_stay_id.notna() & ~linked.ed_link_observed
    linked = linked.merge(admissions.rename(columns={'hadm_id': 'resolved_hadm_id', 'edregtime': 'hospital_edregtime'}),
                          on=['subject_id', 'resolved_hadm_id'], how='left', validate='many_to_one', sort=False)
    for col in ['ecg_time', 'edregtime', 'intime', 'outtime', 'hospital_edregtime', 'admittime', 'dischtime']:
        linked[col] = timestamps(linked[col])
    hospitalization = linked.hadm_id.notna()
    episode_start = linked.hospital_edregtime.fillna(linked.admittime).where(hospitalization, linked.intime)
    episode_end = linked.dischtime.where(hospitalization, linked.outtime)
    temporal_invalid = (linked.ecg_time.isna() | episode_start.isna() | episode_end.isna()
                        | linked.ecg_time.lt(episode_start) | linked.ecg_time.gt(episode_end))
    hospital_registration_conflict = (hospitalization & linked.hospital_edregtime.notna()
                                      & linked.edregtime.ne(linked.hospital_edregtime))
    unresolved = (conflict | admission_invalid | ed_link_invalid | ambiguous_hospital_link
                  | temporal_invalid | hospital_registration_conflict)
    linked.loc[admission_invalid, 'resolved_hadm_id'] = pd.NA

    ed = pd.read_csv(SOURCES['ed_diagnoses'], dtype={'icd_code': str})
    ed_total_rows = len(ed)
    ed = ed[member_keys(ed, linked, 'stay_id', 'ed_stay_id')].copy()
    ed = normalize_codes(ed)
    hosp = []
    hosp_total_rows = 0
    for chunk in pd.read_csv(SOURCES['hospital_diagnoses'], dtype={'icd_code': str}, chunksize=500000):
        hosp_total_rows += len(chunk)
        hosp.append(chunk[member_keys(chunk, linked, 'hadm_id', 'resolved_hadm_id')])
    hosp = normalize_codes(pd.concat(hosp, ignore_index=True))
    dictionary = normalize_codes(pd.read_csv(SOURCES['icd_dictionary'], dtype={'icd_code': str}))
    assert not dictionary.duplicated(['icd_version', 'icd_code']).any()

    source_ed = member_keys(linked, ed, 'ed_stay_id', 'stay_id')
    source_hosp = member_keys(linked, hosp, 'resolved_hadm_id', 'hadm_id')
    wide_union_observed = source_ed | source_hosp
    # A hospital-linked episode requires hospital diagnosis ascertainment even
    # when an ED diagnosis exists. An ED-only episode requires an ED diagnosis.
    source_ascertained = (hospitalization & source_hosp) | (~hospitalization & source_ed)
    diagnosis_observed = source_ascertained & ~unresolved
    output = linked[['row_id', 'file_name', 'subject_id']].copy()
    output['diagnosis_observed'] = diagnosis_observed.astype('int8')
    output['source_ed'] = source_ed.astype('int8')
    output['source_hosp'] = source_hosp.astype('int8')
    output['wide_union_diagnosis_observed'] = wide_union_observed.astype('int8')
    output['source_ascertained_before_linkage_checks'] = source_ascertained.astype('int8')
    output['ascertainment_stratum'] = np.select(
        [hospitalization & source_hosp & source_ed, hospitalization & source_hosp & ~source_ed,
         hospitalization & ~source_hosp & source_ed, hospitalization & ~source_hosp & ~source_ed,
         ~hospitalization & source_ed, ~hospitalization & ~source_ed],
        ['hospital_both_sources', 'hospital_hosp_only', 'hospital_missing_hosp_ed_only',
         'hospital_neither_source', 'ed_only_observed', 'ed_only_no_diagnoses'], default='unexpected')
    output['hadm_backfilled_from_exact_ed_link'] = backfilled.astype('int8')
    output['hadm_nonmissing_conflict'] = conflict.astype('int8')
    output['hospital_link_ambiguous'] = ambiguous_hospital_link.astype('int8')
    output['episode_time_invalid'] = temporal_invalid.astype('int8')
    output['linkage_unresolved'] = unresolved.astype('int8')
    output['resolved_hadm_id'] = linked.resolved_hadm_id
    codebook_frames = []
    counts = {}
    for phenotype, definition in DEFINITIONS.items():
        ed_mask = matches(ed, definition)
        hosp_mask = matches(hosp, definition)
        ed_flag = member_keys(linked, ed.loc[ed_mask], 'ed_stay_id', 'stay_id')
        hosp_flag = member_keys(linked, hosp.loc[hosp_mask], 'resolved_hadm_id', 'hadm_id')
        flag = pd.Series(ed_flag | hosp_flag, index=linked.index).astype('Int8')
        flag.loc[~diagnosis_observed] = pd.NA
        output[phenotype] = flag

        codebook = dictionary.loc[matches(dictionary, definition), ['icd_version', 'icd_code', 'long_title']].copy()
        # Prove every observed selected code has a dictionary title.
        selected_codes = pd.concat([ed.loc[ed_mask, ['icd_version', 'icd_code']],
                                    hosp.loc[hosp_mask, ['icd_version', 'icd_code']]]).drop_duplicates()
        check = selected_codes.merge(codebook, on=['icd_version', 'icd_code'], how='left', validate='one_to_one', indicator=True)
        assert check._merge.eq('both').all(), f'Unmatched dictionary codes in {phenotype}'
        codebook.insert(0, 'phenotype', phenotype)
        codebook.insert(1, 'phenotype_label', definition['label'])
        codebook['definition_rule'] = codebook.icd_version.map(
            {v: json.dumps(definition[v], sort_keys=True) for v in [9, 10]})
        for diagnosis, mask, name in [(ed, ed_mask, 'ed_diagnosis_rows'), (hosp, hosp_mask, 'hospital_diagnosis_rows')]:
            n_rows = diagnosis.loc[mask].groupby(['icd_version', 'icd_code']).size().rename(name).reset_index()
            codebook = codebook.merge(n_rows, on=['icd_version', 'icd_code'], how='left', validate='one_to_one')
            codebook[name] = codebook[name].fillna(0).astype(int)
        codebook['dictionary_match'] = True
        codebook_frames.append(codebook)
        counts[phenotype] = {
            'diagnosis_observed_records': int(diagnosis_observed.sum()),
            'positive_records': int(flag.sum()),
            'negative_records': int(flag.eq(0).sum()),
            'unknown_records': int(flag.isna().sum()),
            'ed_positive_records': int((ed_flag & diagnosis_observed).sum()),
            'hospital_positive_records': int((hosp_flag & diagnosis_observed).sum()),
            'positive_in_both_sources': int((ed_flag & hosp_flag & diagnosis_observed).sum()),
            'positive_backfilled_hadm_records': int((flag.eq(1) & backfilled).sum()),
            'wide_union_positive_records': int((ed_flag | hosp_flag).sum()),
            'ed_positive_ambiguous_hospital_link_records': int((ed_flag & ambiguous_hospital_link).sum()),
            'codebook_rows': len(codebook),
            'selected_observed_codes_missing_dictionary': int(check._merge.ne('both').sum()),
        }
        assert flag.dropna().isin([0, 1]).all()
        assert counts[phenotype]['positive_records'] + counts[phenotype]['negative_records'] == int(diagnosis_observed.sum())
        assert flag.isna().eq(~diagnosis_observed).all()

    codebook = pd.concat(codebook_frames, ignore_index=True).sort_values(['phenotype', 'icd_version', 'icd_code']).reset_index(drop=True)
    assert output.row_id.tolist() == list(range(35948))
    assert output.file_name.tolist() == test.file_name.tolist()
    assert output.subject_id.astype(int).tolist() == test.subject_id.tolist()
    assert output.diagnosis_observed.sum() > 0
    derived_path = OUT / 'derived/phenotypes.csv.gz'
    # Stable gzip header allows identical clean-run hashes.
    with derived_path.open('wb') as raw_handle:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_handle, mtime=0) as compressed:
            compressed.write(output.to_csv(index=False, lineterminator='\n').encode('utf-8'))
    codebook_path = OUT / 'results/phenotype_codebook.csv'
    codebook.to_csv(codebook_path, index=False, lineterminator='\n')
    reread = pd.read_csv(derived_path, dtype={'file_name': str})
    assert len(reread) == 35948 and reread.row_id.tolist() == list(range(35948))
    assert reread.file_name.tolist() == test.file_name.tolist()
    for phenotype in DEFINITIONS:
        assert int(reread[phenotype].sum()) == counts[phenotype]['positive_records']
        assert int(reread[phenotype].isna().sum()) == counts[phenotype]['unknown_records']

    qa = {
        'analysis_version': 'PHENOTYPE_DETERIORATION_20260907_v1',
        'script': str(Path(__file__).resolve()),
        'purpose': 'Current-encounter administrative phenotype derivation; no association estimation',
        'sources': {name: {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size} for name, path in SOURCES.items()},
        'software': {'pandas': pd.__version__, 'numpy': np.__version__},
        'definitions': DEFINITIONS,
        'source_rows': {'ed_diagnoses_total': ed_total_rows, 'hospital_diagnoses_total': hosp_total_rows,
                        'linked_ed_diagnoses': len(ed), 'linked_hospital_diagnoses': len(hosp)},
        'linkage': {
            'parent_test_records': len(test), 'parent_test_patients': int(test.subject_id.nunique()),
            'parent_deaths_72h': int(test.death_72h.sum()), 'cohort_join_matched': len(core),
            'records_missing_original_ed_stay_id': int(linked.ed_stay_id.isna().sum()),
            'records_missing_original_hadm_id': int(linked.hadm_id.isna().sum()),
            'ed_stays_matched_by_subject_and_stay': int(linked.ed_link_observed.sum()),
            'unique_nonmissing_core_ed_stays': int(linked.ed_stay_id.nunique()),
            'original_hadm_nonmissing_records': int(linked.hadm_id.notna().sum()),
            'hadm_backfilled_from_exact_ed_link': int(backfilled.sum()),
            'ambiguous_ed_to_later_hospital_link_records': int(ambiguous_hospital_link.sum()),
            'nonmissing_hadm_conflicts': int(conflict.sum()),
            'resolved_hadm_nonmissing_records': int(linked.resolved_hadm_id.notna().sum()),
            'resolved_hadm_missing_or_wrong_subject_in_admissions': int(admission_invalid.sum()),
            'nonmissing_ed_stay_without_exact_subject_match': int(ed_link_invalid.sum()),
            'unresolved_linkage_records': int(unresolved.sum()),
            'invalid_ecg_episode_time_records': int(temporal_invalid.sum()),
            'hospital_core_registration_time_conflicts': int(hospital_registration_conflict.sum()),
            'hospital_ecg_after_discharge_records': int((hospitalization & linked.ecg_time.gt(linked.dischtime)).sum()),
            'ed_only_ecg_after_ed_discharge_records': int((~hospitalization & linked.ecg_time.gt(linked.outtime)).sum()),
            'diagnosis_observed_records': int(diagnosis_observed.sum()),
            'wide_union_any_diagnosis_records': int(wide_union_observed.sum()),
            'strict_source_ascertainment_before_linkage_checks_records': int(source_ascertained.sum()),
            'wide_union_but_not_strict_primary_records': int((wide_union_observed & ~diagnosis_observed).sum()),
            'unknown_diagnosis_records': int((~diagnosis_observed).sum()),
            'records_with_ed_diagnoses': int(source_ed.sum()),
            'records_with_hospital_diagnoses': int(source_hosp.sum()),
            'both_diagnosis_sources_records': int((source_ed & source_hosp).sum()),
            'unknown_diagnosis_with_linkage_conflict': int((~diagnosis_observed & conflict).sum()),
        },
        'source_ascertainment_strata': output.groupby('ascertainment_stratum').agg(
            records=('row_id', 'size'), primary_eligible=('diagnosis_observed', 'sum'),
            wide_union_observed=('wide_union_diagnosis_observed', 'sum'),
            linkage_unresolved=('linkage_unresolved', 'sum')).astype(int).to_dict(orient='index'),
        'phenotype_counts': counts,
        'assertions': {'parent_row_order_preserved': True, 'parent_row_count_preserved': True,
                       'core_one_to_one_join': True, 'same_subject_encounter_links_only': True,
                       'diagnosis_missingness_not_imputed_negative': True,
                       'leading_zero_icd_codes_preserved': True,
                       'all_selected_observed_codes_match_dictionary': True,
                       'derived_file_reread_matches_counts': True},
        'limitations': [
            'Diagnosis tables lack code-level timestamps and present-on-admission indicators.',
            'These phenotypes characterize the linked care encounter, not established baseline disease or incident post-ECG outcomes.',
            'Administrative coding is incomplete and not equivalent to clinical adjudication.',
            'The sepsis definition is an explicit code family, not Sepsis-3 or exhaustive organism-specific coding.',
            'Respiratory failure includes acute, chronic and unspecified disease; HF uses the principal 428/I50 families.',
            'Neither absence of a diagnosis row nor a conflicting encounter link is treated as phenotype-negative.',
            'Core records include repeated patients and four repeated raw ED stays; downstream inference requires patient clustering.',
        ],
        'outputs': {
            'derived_local_only': {'path': str(derived_path), 'sha256': sha256(derived_path)},
            'codebook': {'path': str(codebook_path), 'sha256': sha256(codebook_path)},
        },
    }
    (OUT / 'qa/phenotype_derivation.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'records': len(output), 'diagnosis_observed': int(diagnosis_observed.sum()),
                      'unknown': int((~diagnosis_observed).sum()), 'hadm_backfilled': int(backfilled.sum()),
                      'hadm_conflicts': int(conflict.sum()),
                      'positive_records': {key: value['positive_records'] for key, value in counts.items()}}, indent=2))


if __name__ == '__main__':
    main()
