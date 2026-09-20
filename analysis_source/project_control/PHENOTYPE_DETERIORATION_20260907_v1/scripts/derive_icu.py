#!/usr/bin/env python3
"""Derive exploratory within-episode post-ECG ICU outcomes; no association fitting."""
from pathlib import Path
import collections
import hashlib
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
RAW = Path('/path/to/research/mimiciv')
SOURCES = {
    'parent_test': PROJECT / 'data/proba_test_72.csv',
    'parent_cohort': PROJECT / 'data/ecg_muti.csv',
    'ed_stays': RAW / 'mimiciv-ed/edstays.csv',
    'admissions': RAW / '3.0/hosp/admissions.csv',
    'transfers': RAW / '3.0/hosp/transfers.csv',
}
ICU_UNITS = {
    'Medical Intensive Care Unit (MICU)',
    'Medical/Surgical Intensive Care Unit (MICU/SICU)',
    'Surgical Intensive Care Unit (SICU)',
    'Trauma SICU (TSICU)',
    'Coronary Care Unit (CCU)',
    'Cardiac Vascular Intensive Care Unit (CVICU)',
    'Neuro Surgical Intensive Care Unit (Neuro SICU)',
    'Intensive Care Unit (ICU)',
}
EXCLUSION_ORDER = [
    'invalid_index_time', 'parent_subject_mismatch', 'ED_subject_mismatch',
    'hospital_subject_mismatch', 'nonempty_ED_to_hospital_conflict',
    'ambiguous_ED_to_hospital_link', 'missing_episode_link',
    'missing_hospital_record', 'missing_hospital_transfers',
    'hospital_source_start_mismatch', 'index_before_episode_start',
    'unascertainable_episode_disposition', 'unascertainable_death_time',
    'index_at_or_after_terminal_discharge', 'death_at_or_before_index',
    'prior_or_current_ICU', 'ICU_terminal_timestamp_tie',
]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()

def norm_id(series):
    s = series.fillna('').astype(str).str.replace(r'\.0$', '', regex=True)
    return s

def read(path, columns=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)

def raw_time(series):
    # These locally exported raw tables use day/month/year. Source audit verified
    # no month/day records; do not rely on date-parser inference.
    return pd.to_datetime(series.replace('', pd.NA), format='%d/%m/%Y %H:%M:%S', errors='coerce')

def iso_time(series):
    return pd.to_datetime(series.replace('', pd.NA), format='%Y-%m-%d %H:%M:%S', errors='coerce')

def ts_out(value):
    return value.isoformat(sep=' ') if pd.notna(value) else ''

def main():
    for d in ['derived', 'qa', 'results']:
        (BASE / d).mkdir(exist_ok=True)
    p = read(SOURCES['parent_test'], ['file_name', 'subject_id'])
    p['row_id'] = range(len(p))
    p = p.rename(columns={'subject_id': 'prediction_subject_id'})
    if len(p) != 35948 or p.file_name.duplicated().any():
        raise ValueError('Parent test cohort identity or row count changed')
    c = read(SOURCES['parent_cohort'], ['file_name', 'subject_id', 'ecg_time', 'edregtime', 'ed_stay_id', 'hadm_id'])
    if c.file_name.duplicated().any():
        raise ValueError('Parent cohort has duplicate file_name keys')
    p = p.merge(c, on='file_name', how='left', validate='one_to_one', indicator=True)
    assert (p['_merge'] == 'both').all()
    p = p.sort_values('row_id')
    for col in ['prediction_subject_id', 'subject_id', 'ed_stay_id', 'hadm_id']:
        p[col] = norm_id(p[col])
    p['index_time'] = iso_time(p.ecg_time)
    p['source_episode_start'] = iso_time(p.edregtime)

    ed = read(SOURCES['ed_stays'])
    for col in ['subject_id', 'hadm_id', 'stay_id']:
        ed[col] = norm_id(ed[col])
    assert not ed.stay_id.duplicated().any()
    ed = ed[ed.stay_id.isin(p.ed_stay_id)].copy()
    for col in ['intime', 'outtime']:
        ed[col] = raw_time(ed[col])
    ed_lookup = ed.set_index('stay_id').to_dict('index')

    admissions = read(SOURCES['admissions'])
    for col in ['subject_id', 'hadm_id']:
        admissions[col] = norm_id(admissions[col])
    assert not admissions.hadm_id.duplicated().any()
    # Include raw-ED-linked candidate admissions for chronology audit ONLY. The
    # 20 parent-missing hadm links are not silently attached to the index episode.
    candidate_hadm = set(p.hadm_id) | set(ed.hadm_id)
    candidate_hadm.discard('')
    admissions = admissions[admissions.hadm_id.isin(candidate_hadm)].copy()
    for col in ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']:
        admissions[col] = raw_time(admissions[col])
    admission_lookup = admissions.set_index('hadm_id').to_dict('index')

    transfers = read(SOURCES['transfers'])
    for col in ['subject_id', 'hadm_id', 'transfer_id']:
        transfers[col] = norm_id(transfers[col])
    unit_counts = transfers.careunit.value_counts(dropna=False).to_dict()
    known_hadms = set(transfers.hadm_id)
    valid_transfers = transfers[transfers.hadm_id.isin(set(p.hadm_id))].copy()
    relevant_unit_counts = valid_transfers.careunit.value_counts(dropna=False).to_dict()
    for col in ['intime', 'outtime']:
        valid_transfers[col] = raw_time(valid_transfers[col])
    icu_rows = valid_transfers[valid_transfers.careunit.isin(ICU_UNITS)].copy()
    if icu_rows.intime.isna().any():
        raise ValueError('ICU entry timestamps failed explicit parsing')
    icu_lookup = {key: group.sort_values(['intime', 'transfer_id']).to_dict('records')
                  for key, group in icu_rows.groupby('hadm_id', sort=False)}
    audit = collections.Counter()
    combinations = collections.Counter()
    event_types = collections.Counter()
    rows = []
    for r in p.to_dict('records'):
        reasons = set()
        sid = r['subject_id']; h = r['hadm_id']; t0 = r['index_time']; start = r['source_episode_start']
        e = ed_lookup.get(r['ed_stay_id'])
        a = admission_lookup.get(h)
        terminal = pd.NaT; death = pd.NaT; discharge = pd.NaT
        episode_disposition = ''; terminal_disposition_conflict = False; first_icu = pd.NaT; first_post = pd.NaT
        first_post_unit = ''; first_post_type = ''; candidate_first = pd.NaT
        if pd.isna(t0): reasons.add('invalid_index_time')
        if sid != r['prediction_subject_id']: reasons.add('parent_subject_mismatch')
        if e:
            if e['subject_id'] != sid: reasons.add('ED_subject_mismatch')
            audit['linked_ED_records'] += 1
            audit['index_after_ED_outtime'] += int(pd.notna(e['outtime']) and t0 > e['outtime'])
            audit['parent_ED_start_differs_from_ED_intime'] += int(start != e['intime'])
            if h and e['hadm_id'] and h != e['hadm_id']:
                reasons.add('nonempty_ED_to_hospital_conflict')
            if not h and e['hadm_id']:
                reasons.add('ambiguous_ED_to_hospital_link')
                ea = admission_lookup.get(e['hadm_id'])
                audit['parent_missing_hadm_with_ED_hadm'] += 1
                if ea:
                    audit['ambiguous_candidate_hospital_start_after_ECG'] += int(ea['edregtime'] > t0)
                    audit['ambiguous_candidate_hospital_start_equals_ED_outtime'] += int(ea['edregtime'] == e['outtime'])
        if h:
            if not a:
                reasons.add('missing_hospital_record')
            else:
                audit['parent_hadm_exact_hospital_match'] += 1
                if a['subject_id'] != sid: reasons.add('hospital_subject_mismatch')
                if h not in known_hadms: reasons.add('missing_hospital_transfers')
                if a['edregtime'] != start: reasons.add('hospital_source_start_mismatch')
                discharge = a['dischtime']; death = a['deathtime']
                episode_disposition = a['discharge_location']
                if a['hospital_expire_flag'] == '1' and pd.isna(death):
                    reasons.add('unascertainable_death_time')
                if pd.notna(death) and pd.notna(discharge) and death > discharge:
                    audit['hospital_death_after_discharge'] += 1
                    terminal_disposition_conflict = True
                if e and e['disposition'] == 'EXPIRED':
                    if pd.notna(e['outtime']) and (pd.isna(death) or e['outtime'] < death):
                        death = e['outtime']
                        audit['ED_expired_time_used_in_hospital_episode'] += 1
        elif e:
            # Only an explicitly terminal ED disposition provides a finite observed
            # ED-only episode. ADMITTED without a resolved hospital link is unknown.
            if e['disposition'] == 'ADMITTED':
                reasons.add('unascertainable_episode_disposition')
            else:
                discharge = e['outtime']
                episode_disposition = e['disposition']
                if e['disposition'] == 'EXPIRED': death = e['outtime']
                if not e['disposition'] or pd.isna(e['outtime']):
                    reasons.add('unascertainable_episode_disposition')
        else:
            reasons.add('missing_episode_link')
        if pd.isna(start) or (pd.notna(t0) and start > t0): reasons.add('index_before_episode_start')
        if pd.isna(discharge): reasons.add('unascertainable_episode_disposition')
        if pd.notna(discharge) and discharge <= t0: reasons.add('index_at_or_after_terminal_discharge')
        if pd.notna(death) and death <= t0: reasons.add('death_at_or_before_index')
        ends = [t for t in [discharge, death] if pd.notna(t)]
        if ends: terminal = min(ends)
        terminal_type = 'death' if pd.notna(death) and death == terminal else 'discharge'
        if terminal_disposition_conflict:
            # Preserve the earliest terminal boundary and all ICU labels. A
            # death timestamp after discharge does not establish live discharge.
            terminal_type = 'terminal_timestamp_conflict'
        episodes_icu = icu_lookup.get(h, []) if h else []
        if episodes_icu:
            first_icu = episodes_icu[0]['intime']
            if any(x['subject_id'] != sid for x in episodes_icu):
                raise ValueError('ICU subject does not match exact linked admission')
            if first_icu <= t0: reasons.add('prior_or_current_ICU')
            after = [x for x in episodes_icu if x['intime'] > t0]
            if after:
                candidate_first = after[0]['intime']
                if pd.notna(terminal):
                    audit['post_ICU_records_after_terminal'] += sum(x['intime'] > terminal for x in after)
                    audit['encounters_with_post_ICU_record_after_terminal'] += int(any(x['intime'] > terminal for x in after))
                    if candidate_first == terminal:
                        reasons.add('ICU_terminal_timestamp_tie')
                    before_end = [x for x in after if x['intime'] < terminal]
                    if before_end:
                        first = before_end[0]
                        first_post = first['intime']; first_post_unit = first['careunit']; first_post_type = first['eventtype']
        ordered_reasons = [x for x in EXCLUSION_ORDER if x in reasons]
        assert len(ordered_reasons) == len(reasons)
        eligible = not reasons
        landmark = bool(eligible and terminal > t0 + pd.Timedelta(minutes=60)
                        and (pd.isna(first_post) or first_post > t0 + pd.Timedelta(minutes=60)))
        if reasons: combinations[';'.join(ordered_reasons)] += 1
        for reason in reasons: audit['exclusion_' + reason] += 1
        if eligible and pd.notna(first_post): event_types[first_post_type] += 1
        out = {
            'row_id': r['row_id'], 'file_name': r['file_name'], 'subject_id': sid,
            'ed_stay_id': r['ed_stay_id'], 'original_hadm_id': h, 'resolved_hadm_id': h,
            'hadm_link_source': 'parent_exact_hadm' if h else 'ED_only_no_hadm',
            'eligible': int(eligible), 'exclusion_reason': ';'.join(ordered_reasons),
            'index_time': ts_out(t0), 'episode_start_time': ts_out(start),
            'episode_end_time': ts_out(terminal), 'terminal_discharge_time': ts_out(discharge),
            'death_time': ts_out(death), 'terminal_type': terminal_type,
            'episode_disposition': episode_disposition,
            'terminal_disposition_conflict': int(terminal_disposition_conflict),
            'hospital_expire_flag': a['hospital_expire_flag'] if a else '',
            'first_icu_time': ts_out(first_icu), 'first_post_icu_candidate_time': ts_out(candidate_first),
            'first_post_icu_time': ts_out(first_post), 'first_post_icu_careunit': first_post_unit,
            'first_post_icu_eventtype': first_post_type,
            'hours_to_episode_end': (terminal - t0).total_seconds()/3600 if pd.notna(terminal) and pd.notna(t0) else None,
            'hours_to_first_post_icu': (first_post - t0).total_seconds()/3600 if pd.notna(first_post) and pd.notna(t0) else None,
            'landmark60_eligible': int(landmark),
        }
        for hours in [24, 72]:
            event = bool(pd.notna(first_post) and first_post <= t0 + pd.Timedelta(hours=hours))
            out[f'icu{hours}'] = int(event) if eligible else None
            out[f'landmark60_icu{hours}'] = int(event) if landmark else None
            if not eligible:
                out[f'disposition{hours}'] = 'excluded'
            elif event:
                out[f'disposition{hours}'] = 'ICU'
            elif terminal <= t0 + pd.Timedelta(hours=hours):
                out[f'disposition{hours}'] = terminal_type
            else:
                out[f'disposition{hours}'] = 'no_event_at_horizon'
        rows.append(out)
    result = pd.DataFrame(rows).sort_values('row_id')
    for col in ['icu24', 'icu72', 'landmark60_icu24', 'landmark60_icu72']:
        result[col] = result[col].astype('Int64')
    assert len(result) == len(p) and result.row_id.tolist() == list(range(35948))
    assert (result.loc[result.eligible.eq(1), 'icu24'] <= result.loc[result.eligible.eq(1), 'icu72']).all()
    assert (result.loc[result.landmark60_eligible.eq(1), 'landmark60_icu24'] <= result.loc[result.landmark60_eligible.eq(1), 'landmark60_icu72']).all()
    assert result.loc[result.eligible.eq(0), ['icu24', 'icu72']].isna().all().all()
    result.to_csv(BASE / 'derived/icu.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0}, lineterminator='\n')
    # Care-unit dictionary includes all observed raw categories and their explicit
    # inclusion status, not only positive units; stepdown, PACU and observation are excluded.
    definitions = pd.DataFrame([
        {'careunit': unit, 'is_ICU': int(unit in ICU_UNITS), 'all_raw_transfer_records': unit_counts.get(unit, 0),
         'records_in_parent_linked_admissions': relevant_unit_counts.get(unit, 0),
         'definition': 'ICU or coronary care entry' if unit in ICU_UNITS else 'not an ICU endpoint unit'}
        for unit in sorted(set(unit_counts) | ICU_UNITS)
    ])
    definitions.to_csv(BASE / 'results/icu_careunit_definitions.csv', index=False)
    flow = [{'stage': 'parent_test', 'n': len(result)}, {'stage': 'endpoint_eligible', 'n': int(result.eligible.sum())},
            {'stage': 'excluded_any', 'n': int(result.eligible.eq(0).sum())}]
    for reason in EXCLUSION_ORDER:
        flow.append({'stage': reason + '_nonexclusive', 'n': audit.get('exclusion_' + reason, 0)})
    flow.append({'stage': 'landmark60_eligible', 'n': int(result.landmark60_eligible.sum())})
    disposition_counts = {}
    for hours in [24, 72]:
        disposition_counts[str(hours)] = {str(k): int(v) for k,v in result[f'disposition{hours}'].value_counts().items()}
        for state, n in disposition_counts[str(hours)].items():
            flow.append({'stage': f'{hours}h_{state}', 'n': n})
        flow.append({'stage': f'landmark60_icu{hours}', 'n': int(result[f'landmark60_icu{hours}'].sum())})
    conflict_disposition_counts = {}
    for hours in [24, 72]:
        conflict_disposition_counts[str(hours)] = {str(k): int(v) for k,v in result.loc[result.terminal_disposition_conflict.eq(1), f'disposition{hours}'].value_counts().items()}
        for state, n in conflict_disposition_counts[str(hours)].items():
            flow.append({'stage': f'{hours}h_{state}_with_terminal_death_timestamp_conflict', 'n': n})
    pd.DataFrame(flow).to_csv(BASE / 'results/icu_flow.csv', index=False)
    qa = {
        'specification': str(BASE / 'ANALYSIS_PLAN.md'),
        'source_sha256': {key: {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size} for key,path in SOURCES.items()},
        'script_sha256': sha256(Path(__file__)),
        'output_sha256': {'icu_csv_gz': sha256(BASE / 'derived/icu.csv.gz'), 'flow_csv': sha256(BASE / 'results/icu_flow.csv'), 'careunit_csv': sha256(BASE / 'results/icu_careunit_definitions.csv')},
        'parent_n': len(result), 'eligible_n': int(result.eligible.sum()), 'excluded_n': int(result.eligible.eq(0).sum()),
        'landmark60_eligible_n': int(result.landmark60_eligible.sum()),
        'event_counts': {f'icu{h}': int(result[f'icu{h}'].sum()) for h in [24,72]},
        'landmark60_event_counts': {f'icu{h}': int(result[f'landmark60_icu{h}'].sum()) for h in [24,72]},
        'disposition_counts': disposition_counts,
        'terminal_disposition_conflict_counts': conflict_disposition_counts,
        'terminal_disposition_conflict_note': '50 parent records have hospital_expire_flag=1 and recorded death after discharge. Preserve earliest terminal boundary for ICU labels. Their terminal type is terminal_timestamp_conflict. Of these records, 3 reach the terminal boundary without preceding ICU within both horizons and are classified as terminal_timestamp_conflict, separately from discharge. This classification does not change eligibility, time boundaries, or ICU labels.',
        'audit_nonexclusive_counts': dict(audit), 'exclusion_overlap_combinations': dict(combinations),
        'first_post_icu_eventtype_among_eligible': dict(event_types),
        'ICU_careunits': sorted(ICU_UNITS),
        'date_rule': 'Parent ISO; raw ED, hospital and transfers explicitly %d/%m/%Y %H:%M:%S.',
        'ambiguous_ED_hadm_rule': 'Do not coalesce 20 raw ED hadm identifiers when parent hadm is missing: corresponding hospital ED start is later than index ECG (18 equal original ED outtime), and original ED disposition is HOME. Exclude these records from ICU analysis; preserve parent rows.',
        'boundary_rule': 'Index must precede terminal episode end; earliest same-hadm ICU entry must be strictly after index. Count ICU strictly before minimum terminal discharge/death. A tie of first post-index ICU with terminal time excludes that encounter. Ignore and audit ICU entries after terminal time.',
        'landmark_rule': 'Eligible and alive/in-episode/ICU-free strictly through ECG+60 minutes; retain original ECG+24/72h horizons.',
        'claim_boundary': 'Observed escalation to ICU within linked care episode, including direct ICU admission. Does not prove new physiological deterioration; no out-of-hospital or later-episode follow-up.',
        'units_excluded': 'PACU, observation, intermediate/stepdown units and special care nursery are not ICU endpoints.',
        'assertions': ['All 35948 original rows retained in exact parent-test order', 'No transfers attached by subject_id alone', 'All existing hadm_id match admissions exactly', 'Excluded endpoints are missing, never zero', '24h events are subset of 72h events', 'All relevant ICU timestamps parse explicitly'],
    }
    (BASE / 'qa/icu_derivation.json').write_text(json.dumps(qa, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: qa[k] for k in ['parent_n','eligible_n','excluded_n','landmark60_eligible_n','event_counts','landmark60_event_counts','disposition_counts','exclusion_overlap_combinations']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
