#!/usr/bin/env python3
"""Prepare recorded ICU organ-support endpoints. No association models.

Frozen specification: ../SUPPORT_ANALYSIS_PLAN.md.
Default execution requires all exports. --self-test uses synthetic data only.
--exports-verified records the operator's completed export-count/checksum QA;
without it all outputs remain provisional and must not enter a result lock.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
EXPORTS = BASE / 'data_exports'
SOURCE_DIR = BASE / 'qa/mimic_official_sources'
FILES = {'icustays': 'icustays.csv', 'charts': 'chartevents_resp.csv.gz',
         'inputs': 'inputevents_pressors.csv.gz', 'procedures': 'procedureevents_support.csv.gz',
         'existing_ventilation': 'ventilation.csv.gz'}
VENT_IDS = {224688,224689,224690,224687,224685,224684,224686,224696,220339,224700,223835,223849,229314,223848,224691}
OXYGEN_IDS = {223834,227582,227287,226732}
PRESSORS = {221906:'norepinephrine',221289:'epinephrine',221749:'phenylephrine',222315:'vasopressin',221662:'dopamine'}
PROCEDURE_IDS = {224385,225448,225468,225477,225792,225794,226237,227194}
BASELINE_AIRWAY_IDS = {224385,225792,225448,226237}
INVASIVE_MODES = {'(S) CMV','APRV','APRV/Biphasic+ApnPress','APRV/Biphasic+ApnVol','APV (cmv)','Ambient','Apnea Ventilation','CMV','CMV/ASSIST','CMV/ASSIST/AutoFlow','CMV/AutoFlow','CPAP/PPS','CPAP/PSV','CPAP/PSV+Apn TCPL','CPAP/PSV+ApnPres','CPAP/PSV+ApnVol','MMV','MMV/AutoFlow','MMV/PSV','MMV/PSV/AutoFlow','P-CMV','PCV+','PCV+/PSV','PCV+Assist','PRES/AC','PRVC/AC','PRVC/SIMV','PSV/SBT','SIMV','SIMV/AutoFlow','SIMV/PRES','SIMV/PSV','SIMV/PSV/AutoFlow','SIMV/VOL','SYNCHRON MASTER','SYNCHRON SLAVE','VOL/AC'}
HAMILTON_INVASIVE = {'APRV','APV (cmv)','Ambient','(S) CMV','P-CMV','SIMV','APV (simv)','P-SIMV','VS','ASV'}
SUPPLEMENTAL = {'Non-rebreather','Face tent','Aerosol-cool','Venti mask ','Medium conc mask ','Ultrasonic neb','Vapomist','Oxymizer','High flow neb','Nasal cannula'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def parse_time(series):
    s = series.fillna('').astype(str)
    out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    iso = s.str.match(r'^\d{4}-\d{2}-\d{2}[ T]')
    out.loc[iso] = pd.to_datetime(s.loc[iso], format='ISO8601', errors='coerce')
    slash = s.str.match(r'^\d{1,2}/\d{1,2}/\d{4} ')
    out.loc[slash] = pd.to_datetime(s.loc[slash], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    return out


def ids(df):
    for col in ['subject_id','hadm_id','stay_id','itemid','orderid','linkorderid']:
        if col in df: df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df


def filtered_csv(path, hadms=None, stays=None, columns=None):
    pieces = []; total = 0
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns, chunksize=200000):
        total += len(chunk)
        chunk = ids(chunk)
        if hadms is not None:
            chunk = chunk[chunk.hadm_id.isin(hadms)]
        elif stays is not None:
            chunk = chunk[chunk.stay_id.isin(stays)]
        if len(chunk): pieces.append(chunk)
    if pieces: return pd.concat(pieces, ignore_index=True), total
    header = pd.read_csv(path, nrows=0, usecols=columns)
    return ids(header), total


def clean_times(df, cols, qa, key):
    for col in cols:
        original = df[col].fillna('').astype(str)
        parsed = parse_time(original)
        qa[f'{key}_{col}_nonblank_parse_failures'] = int((original.ne('') & parsed.isna()).sum())
        df[col] = parsed
    return df


def check_links(df, stay_dim, qa, key):
    ref = stay_dim[['stay_id','subject_id','hadm_id']].rename(columns={'subject_id':'sid_ref','hadm_id':'hadm_ref'})
    x = df.merge(ref, on='stay_id', how='left', validate='many_to_one')
    invalid = x.sid_ref.isna() | x.subject_id.ne(x.sid_ref) | x.hadm_id.ne(x.hadm_ref)
    invalid = invalid.fillna(True)
    qa[key+'_invalid_ICU_links'] = int(invalid.sum())
    bad_hadms = set(x.loc[invalid,'hadm_id'].dropna().astype(int))
    return x.loc[~invalid].drop(columns=['sid_ref','hadm_ref']), bad_hadms


def oxygen_table(charts, require_flow=True):
    """Reproduce current oxygen_delivery ce.rn=1 behavior; optional direct-device QA."""
    keys = ['subject_id','charttime']
    flows = charts[charts.itemid.isin([223834,227582,227287]) & charts.value.ne('')].copy()
    flows['mapped_itemid'] = flows.itemid.replace({227582:223834})
    flows = flows.sort_values(keys+['mapped_itemid','storetime','valuenum'], ascending=[True,True,True,False,False], na_position='last')
    flows = flows.drop_duplicates(keys+['mapped_itemid'], keep='first')
    flow_base = flows.groupby(keys, dropna=False).stay_id.max().to_frame()
    devices = charts[charts.itemid.eq(226732)].copy()
    devices = devices.sort_values(keys+['storetime','value'], ascending=[True,True,False,False], na_position='last')
    devices['rn'] = devices.groupby(keys, dropna=False).cumcount()+1
    devices = devices[devices.rn.le(4)]
    dev_base = devices.groupby(keys, dropna=False).stay_id.max().to_frame()
    slots = devices.pivot(index=keys, columns='rn', values='value')
    slots = slots.rename(columns={n:f'o2_delivery_device_{n}' for n in slots.columns})
    if require_flow:
        # FULL OUTER JOIN followed by WHERE ce.rn=1 retains flow-side keys only.
        base = flow_base.join(dev_base.rename(columns={'stay_id':'device_stay'}), how='left')
        base['stay_id'] = base.stay_id.fillna(base.device_stay)
        base = base.drop(columns='device_stay').join(slots, how='left')
    else:
        base = dev_base.join(slots, how='left')
    for n in range(1,5):
        if f'o2_delivery_device_{n}' not in base: base[f'o2_delivery_device_{n}'] = ''
    return base.reset_index()


def ventilation_settings(charts):
    keys = ['subject_id','charttime']
    v = charts[charts.itemid.isin(VENT_IDS) & charts.value.ne('') & charts.stay_id.notna()].copy()
    base = v.groupby(keys, dropna=False).stay_id.max().to_frame()
    for item, name in [(223849,'ventilator_mode'),(229314,'ventilator_mode_hamilton')]:
        sub = v[v.itemid.eq(item)].groupby(keys, dropna=False).value.max().rename(name)
        base = base.join(sub, how='left')
    return base.reset_index()


def classify_states(settings, oxygen):
    keys = ['stay_id','charttime']
    a = settings.drop(columns='subject_id'); b = oxygen.drop(columns='subject_id')
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError('Upstream subject/time aggregation produced duplicate stay/time keys')
    x = a.merge(b, on=keys, how='outer', validate='one_to_one')
    for c in ['ventilator_mode','ventilator_mode_hamilton']+[f'o2_delivery_device_{n}' for n in range(1,5)]:
        if c not in x: x[c] = ''
        x[c] = x[c].fillna('')
    d = x.o2_delivery_device_1; v=x.ventilator_mode; h=x.ventilator_mode_hamilton
    conditions = [d.isin(['Tracheostomy tube','Trach mask ']),
                  d.eq('Endotracheal tube')|v.isin(INVASIVE_MODES)|h.isin(HAMILTON_INVASIVE),
                  x[[f'o2_delivery_device_{n}' for n in range(1,5)]].isin(['Bipap mask ','CPAP mask ']).any(axis=1)|h.isin(['DuoPaP','NIV','NIV-ST']),
                  d.eq('High flow nasal cannula'), d.isin(SUPPLEMENTAL), d.eq('None')]
    x['ventilation_status'] = np.select(conditions, ['Tracheostomy','InvasiveVent','NonInvasiveVent','HFNC','SupplementalOxygen','None'], default='')
    return x[x.ventilation_status.ne('') & x.stay_id.notna() & x.charttime.notna()].copy()


def ventilation_episodes(states):
    columns = ['stay_id','starttime','endtime','ventilation_status']
    if states.empty: return pd.DataFrame(columns=columns), {'classified_states':0,'single_charttime_sequences_removed':0}
    s = states.sort_values(['stay_id','charttime']).copy()
    if s.duplicated(['stay_id','charttime']).any(): raise ValueError('Duplicate classified state at same stay/time')
    s['same_state_lag'] = s.groupby(['stay_id','ventilation_status']).charttime.shift()
    s['next_time'] = s.groupby('stay_id').charttime.shift(-1)
    s['previous_state'] = s.groupby('stay_id').ventilation_status.shift()
    # Generated official PostgreSQL explicitly uses DATE_TRUNC hour boundaries.
    gap = (s.charttime.dt.floor('h')-s.same_state_lag.dt.floor('h')).dt.total_seconds()/3600
    new = s.previous_state.isna() | gap.ge(14) | s.previous_state.ne(s.ventilation_status)
    s['sequence'] = new.astype(int).groupby(s.stay_id).cumsum()
    next_gap = (s.next_time.dt.floor('h')-s.charttime.dt.floor('h')).dt.total_seconds()/3600
    s['row_end'] = s.next_time.where(s.next_time.notna() & next_gap.lt(14), s.charttime)
    e = s.groupby(['stay_id','sequence']).agg(starttime=('charttime','min'), last_charttime=('charttime','max'), endtime=('row_end','max'), ventilation_status=('ventilation_status','max')).reset_index()
    removed = int(e.starttime.eq(e.last_charttime).sum())
    e = e[e.starttime.ne(e.last_charttime)]
    return e[columns], {'classified_states':len(s),'single_charttime_sequences_removed':removed}


def merge_infusions(valid):
    rows = []
    for (hadm,drug), g in valid.sort_values(['hadm_id','itemid','starttime','endtime']).groupby(['hadm_id','itemid']):
        start=end=None; segments=0
        for r in g.itertuples():
            if start is None: start,end,segments=r.starttime,r.endtime,1
            elif r.starttime <= end: end=max(end,r.endtime);segments+=1
            else:
                rows.append((hadm,drug,start,end,segments));start,end,segments=r.starttime,r.endtime,1
        if start is not None: rows.append((hadm,drug,start,end,segments))
    return pd.DataFrame(rows,columns=['hadm_id','itemid','starttime','endtime','segments'])


def group_records(df, key='hadm_id'):
    return {int(h):g.to_dict('records') for h,g in df.groupby(key)} if len(df) else {}


def after_time(times,t0,terminal):
    times=sorted(t for t in times if pd.notna(t) and t>t0)
    tie=bool(times and times[0]==terminal)
    valid=[t for t in times if t<terminal]
    return (valid[0] if valid else pd.NaT),tie,sum(t>terminal for t in times)


def main(args):
    required = [EXPORTS/name for name in FILES.values()]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise SystemExit('Awaiting completed raw exports: '+', '.join(missing))
    export_manifest_path=BASE/'qa/navicat_export_manifest.json'
    export_manifest=json.loads(export_manifest_path.read_text())
    for name in FILES.values():
        if name not in export_manifest['files'] or sha(EXPORTS/name)!=export_manifest['files'][name]['sha256']:
            raise ValueError('Export checksum mismatch or absent Navicat manifest entry: '+name)
    cohort=pd.read_csv(BASE/'derived/icu.csv.gz',keep_default_na=False)
    if len(cohort)!=35948 or cohort.row_id.tolist()!=list(range(35948)): raise ValueError('Parent order mismatch')
    cohort['hadm_id']=pd.to_numeric(cohort.resolved_hadm_id,errors='coerce').astype('Int64')
    cohort['subject_id']=pd.to_numeric(cohort.subject_id,errors='raise').astype('Int64')
    for c in ['index_time','episode_start_time','episode_end_time']: cohort[c]=parse_time(cohort[c])
    hadms=set(cohort.hadm_id.dropna().astype(int));qa={};counts={}
    icu,counts['icustays']=filtered_csv(EXPORTS/FILES['icustays'],hadms=hadms)
    icu=clean_times(icu,['intime','outtime'],qa,'icustays')
    if icu.stay_id.duplicated().any(): raise ValueError('ICU stay_id is not unique')
    stays=set(icu.stay_id.dropna().astype(int))
    charts,counts['charts']=filtered_csv(EXPORTS/FILES['charts'],hadms=hadms,columns=['subject_id','hadm_id','stay_id','charttime','storetime','itemid','value','valuenum','valueuom'])
    charts=clean_times(charts,['charttime','storetime'],qa,'charts')
    charts['valuenum']=pd.to_numeric(charts.valuenum,errors='coerce')
    charts,bad_chart_hadms=check_links(charts,icu,qa,'charts')
    bad_chart_hadms |= set(charts.loc[charts.charttime.isna(),'hadm_id'].dropna().astype(int))
    if not set(charts.itemid.dropna().astype(int)).issubset(VENT_IDS|OXYGEN_IDS): raise ValueError('Unexpected respiratory item IDs')
    settings=ventilation_settings(charts)
    oxygen=oxygen_table(charts,require_flow=True)
    direct_oxygen=oxygen_table(charts,require_flow=False)
    states=classify_states(settings,oxygen)
    direct_states=classify_states(settings,direct_oxygen)
    episodes,qa['ventilation_sequences']=ventilation_episodes(states)
    qa['device_only_subject_timestamps_omitted_by_official_flow_join']=int(len(direct_oxygen.merge(oxygen[['subject_id','charttime']],on=['subject_id','charttime'],how='left',indicator=True).query('_merge == "left_only"')))
    episodes=episodes.merge(icu[['stay_id','hadm_id','subject_id']],on='stay_id',validate='many_to_one')
    direct_states=direct_states.merge(icu[['stay_id','hadm_id']],on='stay_id',validate='many_to_one')
    existing,counts['existing_ventilation']=filtered_csv(EXPORTS/FILES['existing_ventilation'],stays=stays)
    existing=clean_times(existing,['starttime','endtime'],qa,'existing_ventilation')
    compare_keys=['stay_id','starttime','endtime','ventilation_status']
    qa['existing_ventilation_duplicate_tuples']=int(existing.duplicated(compare_keys).sum())
    compare=episodes[compare_keys].drop_duplicates().merge(existing[compare_keys].drop_duplicates(),on=compare_keys,how='outer',indicator=True)
    compare_summary=compare.groupby(['ventilation_status','_merge'],observed=True).size().rename('n').reset_index()
    inputs,counts['inputs']=filtered_csv(EXPORTS/FILES['inputs'],hadms=hadms)
    inputs=clean_times(inputs,['starttime','endtime','storetime'],qa,'inputs')
    inputs,bad_input_hadms=check_links(inputs,icu,qa,'inputs')
    inputs['rate']=pd.to_numeric(inputs.rate,errors='coerce')
    qa['input_status_counts']=inputs.statusdescription.value_counts(dropna=False).to_dict()
    qa['input_rate_units']=inputs.groupby(['itemid','rateuom'],dropna=False).size().rename('n').reset_index().to_dict('records')
    rewritten=inputs.statusdescription.str.strip().str.lower().eq('rewritten')
    bad_duration=inputs.starttime.isna()|inputs.endtime.isna()|inputs.endtime.le(inputs.starttime)
    bad_rate=inputs.rate.isna()|inputs.rate.le(0)
    qa['input_nonexclusive_exclusions']={'Rewritten':int(rewritten.sum()),'invalid_duration':int(bad_duration.sum()),'nonpositive_or_missing_rate':int(bad_rate.sum())}
    bad_input_hadms |= set(inputs.loc[~rewritten & ~bad_rate & bad_duration,'hadm_id'].dropna().astype(int))
    valid=inputs[inputs.itemid.isin(PRESSORS)&~rewritten&~bad_duration&~bad_rate].copy()
    infusions=merge_infusions(valid)
    no_bolus_infusions=merge_infusions(valid[~valid.statusdescription.str.strip().str.lower().eq('bolus')])
    qa['positive_rate_segments_with_Bolus_status']=int(valid.statusdescription.str.strip().str.lower().eq('bolus').sum())
    qa['valid_pressor_segments']=len(valid);qa['merged_pressor_episodes']=len(infusions)
    procedures,counts['procedures']=filtered_csv(EXPORTS/FILES['procedures'],hadms=hadms)
    procedures=clean_times(procedures,['starttime','endtime','storetime'],qa,'procedures')
    procedures,bad_procedure_hadms=check_links(procedures,icu,qa,'procedures')
    bad_procedure_hadms |= set(procedures.loc[procedures.itemid.isin(BASELINE_AIRWAY_IDS) & procedures.starttime.isna(),'hadm_id'].dropna().astype(int))
    proc_valid=procedures[procedures.itemid.isin(PROCEDURE_IDS)&procedures.starttime.notna()].copy()
    # Coverage of physical ICU entries is independently checked with the parent
    # raw transfers; an ICU event missing from icustays must not become a negative.
    meta=json.loads((BASE/'qa/icu_derivation.json').read_text())
    transfer_path=Path(meta['source_sha256']['transfers']['path'])
    transfers,_=filtered_csv(transfer_path,hadms=hadms,columns=['subject_id','hadm_id','careunit','intime','outtime'])
    transfers=clean_times(transfers,['intime','outtime'],qa,'transfers')
    transfers=transfers[transfers.careunit.isin(meta['ICU_careunits'])]
    td=group_records(transfers);sd=group_records(icu)
    vd=group_records(episodes[episodes.ventilation_status.eq('InvasiveVent')]); rd=group_records(direct_states)
    existing_linked=existing.merge(icu[['stay_id','hadm_id']],on='stay_id',validate='many_to_one')
    evd=group_records(existing_linked[existing_linked.ventilation_status.eq('InvasiveVent')])
    pdict=group_records(proc_valid);fd=group_records(infusions);bfd=group_records(no_bolus_infusions)
    rows=[];flow=collections.Counter()
    for r in cohort.to_dict('records'):
        h=int(r['hadm_id']) if pd.notna(r['hadm_id']) else None
        t0=r['index_time']; terminal=r['episode_end_time']; base=int(r['eligible'])==1
        source_gap=False
        if base and h:
            for tr in td.get(h,[]):
                if tr['intime']>t0+pd.Timedelta(hours=72) or tr['intime']>=terminal: continue
                if tr['intime']<r['episode_start_time']: continue
                if not any(s['intime']<=tr['intime']<=s['outtime'] for s in sd.get(h,[]) if pd.notna(s['intime']) and pd.notna(s['outtime'])):
                    source_gap=True
        raw_baseline=any(z['charttime']<=t0 and z['ventilation_status'] in ['InvasiveVent','Tracheostomy'] for z in rd.get(h,[]))
        proc_baseline=any(z['starttime']<=t0 and z['itemid'] in BASELINE_AIRWAY_IDS for z in pdict.get(h,[]))
        pressor_baseline=any(z['starttime']<=t0 for z in fd.get(h,[]))
        imv_landmark_airway=any(z['charttime']<=t0+pd.Timedelta(minutes=60) and z['ventilation_status'] in ['InvasiveVent','Tracheostomy'] for z in rd.get(h,[])) or any(z['starttime']<=t0+pd.Timedelta(minutes=60) and z['itemid'] in BASELINE_AIRWAY_IDS for z in pdict.get(h,[]))
        imv,imv_tie,imv_late=after_time([z['starttime'] for z in vd.get(h,[])],t0,terminal)
        pressor,pressor_tie,pressor_late=after_time([z['starttime'] for z in fd.get(h,[])],t0,terminal)
        no_bolus_first,_,_=after_time([z['starttime'] for z in bfd.get(h,[])],t0,terminal)
        existing_imv_first,_,_=after_time([z['starttime'] for z in evd.get(h,[])],t0,terminal)
        non_dopamine_first,_,_=after_time([z['starttime'] for z in fd.get(h,[]) if z['itemid']!=221662],t0,terminal)
        direct_first,_,_=after_time([z['charttime'] for z in rd.get(h,[]) if z['ventilation_status']=='InvasiveVent'],t0,terminal)
        proc_first,_,_=after_time([z['starttime'] for z in pdict.get(h,[]) if z['itemid'] in [224385,225792]],t0,terminal)
        out={'row_id':r['row_id'],'file_name':r['file_name'],'subject_id':r['subject_id'],'parent_ICU_eligible':int(base),'hadm_id':h,
             'ICU_source_coverage_gap':int(source_gap),'baseline_imv_or_trach_raw_record':int(raw_baseline),'baseline_invasive_airway_procedure':int(proc_baseline),
             'baseline_pressor_record':int(pressor_baseline),'imv_raw_airway_support_by_landmark60':int(imv_landmark_airway),'first_imv_record_time':imv,'first_pressor_record_time':pressor,
             'first_raw_invasive_state_time':direct_first,'first_invasive_procedure_time':proc_first,
             'baseline_absence_interpretation':'not recorded; absence of ED or prehospital support is not established'}
        for name,first,prior,tie,bad in [('imv',imv,raw_baseline or proc_baseline,imv_tie,bad_chart_hadms|bad_procedure_hadms),('pressor',pressor,pressor_baseline,pressor_tie,bad_input_hadms)]:
            reasons=[]
            if not base: reasons.append('parent_ICU_risk_set_excluded')
            if source_gap: reasons.append('ICU_transfer_not_covered_by_icustays')
            if h in bad: reasons.append('invalid_raw_ICU_link')
            if prior: reasons.append('baseline_support_or_airway_record')
            if tie: reasons.append('support_terminal_timestamp_tie')
            eligible=not reasons
            landmark=bool(eligible and terminal>t0+pd.Timedelta(minutes=60) and (pd.isna(first) or first>t0+pd.Timedelta(minutes=60)) and not (name=='imv' and imv_landmark_airway))
            out[name+'_observable']=int(base and not source_gap and h not in bad)
            out[name+'_eligible']=int(eligible);out[name+'_exclusion_reason']=';'.join(reasons);out[name+'_landmark60_eligible']=int(landmark)
            flow[name+'_eligible']+=eligible;flow[name+'_landmark60_eligible']+=landmark
            for reason in reasons: flow[name+'_exclude_'+reason]+=1
            for hours in [24,72]:
                event=bool(pd.notna(first) and first<=t0+pd.Timedelta(hours=hours))
                out[f'{name}{hours}']=int(event) if eligible else None
                out[f'{name}_landmark60_{hours}']=int(event) if landmark else None
                flow[f'{name}{hours}']+=int(eligible and event)
                flow[f'{name}_landmark60_{hours}']+=int(landmark and event)
        for hours in [24,72]:
            out[f'pressor_no_bolus{hours}']=int(pd.notna(no_bolus_first) and no_bolus_first<=t0+pd.Timedelta(hours=hours)) if out['pressor_eligible'] else None
            out[f'imv_existing{hours}']=int(pd.notna(existing_imv_first) and existing_imv_first<=t0+pd.Timedelta(hours=hours)) if out['imv_eligible'] else None
            out[f'pressor_no_dopamine{hours}']=int(pd.notna(non_dopamine_first) and non_dopamine_first<=t0+pd.Timedelta(hours=hours)) if out['pressor_eligible'] else None
            out[f'imv_raw_state{hours}']=int(pd.notna(direct_first) and direct_first<=t0+pd.Timedelta(hours=hours)) if out['imv_eligible'] else None
            out[f'imv_procedure{hours}']=int(pd.notna(proc_first) and proc_first<=t0+pd.Timedelta(hours=hours)) if out['imv_eligible'] else None
        flow['IMV_episode_starts_after_terminal']+=imv_late;flow['pressor_episode_starts_after_terminal']+=pressor_late
        rows.append(out)
    result=pd.DataFrame(rows)
    for name in ['imv','pressor']:
        for col in [f'{name}24',f'{name}72',f'{name}_landmark60_24',f'{name}_landmark60_72']:
            result[col]=result[col].astype('Int64')
        eligible=result[name+'_eligible'].eq(1)
        assert (result.loc[eligible,name+'24']<=result.loc[eligible,name+'72']).all()
    qa['fixed_risk_set_sensitivity_counts']={}
    for hours in [24,72]:
        imv_mask=result.imv_eligible.eq(1); pressor_mask=result.pressor_eligible.eq(1)
        qa['fixed_risk_set_sensitivity_counts'][str(hours)]={
            'IMV_existing_derived_events':int(result[f'imv_existing{hours}'].sum()),
            'IMV_existing_vs_rebuilt_row_discordance':int((result.loc[imv_mask,f'imv_existing{hours}']!=result.loc[imv_mask,f'imv{hours}']).sum()),
            'IMV_raw_state_events':int(result[f'imv_raw_state{hours}'].sum()),
            'IMV_procedure_events':int(result[f'imv_procedure{hours}'].sum()),
            'pressor_no_dopamine_events':int(result[f'pressor_no_dopamine{hours}'].sum()),
            'pressor_no_Bolus_events':int(result[f'pressor_no_bolus{hours}'].sum()),
            'pressor_no_Bolus_row_discordance':int((result.loc[pressor_mask,f'pressor_no_bolus{hours}']!=result.loc[pressor_mask,f'pressor{hours}']).sum())}
    assert result.row_id.tolist()==list(range(35948))
    for sub in ['derived','qa','results']: (BASE/sub).mkdir(exist_ok=True)
    result.to_csv(BASE/'derived/support.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    episodes.to_csv(BASE/'derived/ventilation_rebuilt.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    compare_summary.to_csv(BASE/'results/ventilation_reconstruction_comparison.csv',index=False)
    pd.DataFrame([{'stage':k,'n':v} for k,v in sorted(flow.items())]).to_csv(BASE/'results/support_flow.csv',index=False)
    expected_counts={'icustays':export_manifest['navicat_ui_counts']['icustays'], 'charts':export_manifest['raw_source_rows']['chartevents_resp'], 'inputs':export_manifest['raw_source_rows']['inputevents_pressors'], 'procedures':export_manifest['raw_source_rows']['procedureevents_support']}
    for key,expected in expected_counts.items():
        if counts[key]!=expected: raise ValueError('Navicat export row count mismatch: '+key)
    qa['export_manifest_sha256']=sha(export_manifest_path)
    qa['export_manifest_checks']='All five source file hashes match; icustays and three raw support source row counts match recorded Navicat completed-export counts.'
    qa['exclusion_overlap_patterns']={name:result.loc[result[name+'_eligible'].eq(0),name+'_exclusion_reason'].value_counts().to_dict() for name in ['imv','pressor']}
    qa['derived_icu_input_sha256']=sha(BASE/'derived/icu.csv.gz')
    qa.update({'status':'derived pending statistical analysis' if args.exports_verified else 'PROVISIONAL - EXPORT COMPLETENESS NOT VERIFIED',
               'exports_verified':bool(args.exports_verified),'export_total_rows':counts,'flow_nonexclusive':dict(flow),
               'sources':{k:{'path':str(EXPORTS/n),'sha256':sha(EXPORTS/n)} for k,n in FILES.items()},
               'script_sha256':sha(__file__),'support_plan_sha256':sha(BASE/'SUPPORT_ANALYSIS_PLAN.md'),
               'official_source_manifest_sha256':sha(SOURCE_DIR/'source_manifest.json'),
               'input_unit_rule':'Binary valid infusion requires positive actual rate; unknown units are tabulated and never used for standardized dose claims.',
               'claim_boundary':'First ICU-system-recorded support, not established clinical initiation; no record does not prove absence of ED, prehospital or ward treatment.'})
    (BASE/'qa/support_derivation.json').write_text(json.dumps(qa,indent=2,default=str)+'\n')
    print(json.dumps({'status':qa['status'],'rows':len(result),'flow':dict(flow)},indent=2))


def self_test():
    # Scientific edge cases: priority, boundary-count 14h rule, singleton omission,
    # infusion rate changes, and strict terminal-event ordering.
    settings=pd.DataFrame({'subject_id':[1,1,1,1],'stay_id':[10]*4,'charttime':pd.to_datetime(['2020-01-01 00:00','2020-01-01 01:00','2020-01-01 02:00','2020-01-01 03:00']),'ventilator_mode':['CMV']*4,'ventilator_mode_hamilton':['']*4})
    oxygen=pd.DataFrame({'subject_id':[1]*4,'stay_id':[10]*4,'charttime':settings.charttime,'o2_delivery_device_1':['Tracheostomy tube','Endotracheal tube','','None']})
    classified=classify_states(settings,oxygen)
    assert classified.ventilation_status.tolist()==['Tracheostomy','InvasiveVent','InvasiveVent','InvasiveVent']
    states=pd.DataFrame({'stay_id':[1]*5,'charttime':pd.to_datetime(['2020-01-01 00:59','2020-01-01 01:01','2020-01-01 15:00','2020-01-01 15:05','2020-01-01 16:00']),'ventilation_status':['InvasiveVent']*4+['None']})
    episodes,q=ventilation_episodes(states)
    assert len(episodes)==2 and q['single_charttime_sequences_removed']==1
    assert episodes.starttime.tolist()==[pd.Timestamp('2020-01-01 00:59'),pd.Timestamp('2020-01-01 15:00')]
    assert episodes.endtime.tolist()==[pd.Timestamp('2020-01-01 01:01'),pd.Timestamp('2020-01-01 16:00')]
    inf=pd.DataFrame({'hadm_id':[1,1,1],'itemid':[221906]*3,'starttime':pd.to_datetime(['2020-01-01 00:00','2020-01-01 00:10','2020-01-01 01:00']),'endtime':pd.to_datetime(['2020-01-01 00:10','2020-01-01 00:20','2020-01-01 02:00'])})
    merged=merge_infusions(inf);assert len(merged)==2 and merged.segments.tolist()==[2,1]
    t0=pd.Timestamp('2020-01-01');terminal=t0+pd.Timedelta(hours=24)
    first,tie,late=after_time([terminal,terminal+pd.Timedelta(hours=1)],t0,terminal)
    assert pd.isna(first) and tie and late==1
    chart=pd.DataFrame({'subject_id':[1,1,1,1], 'hadm_id':[2]*4, 'stay_id':[3]*4, 'charttime':pd.to_datetime(['2020-01-01 00:00','2020-01-01 00:00','2020-01-01 01:00','2020-01-01 02:00']), 'storetime':pd.to_datetime(['2020-01-01 00:01','2020-01-01 00:01','2020-01-01 01:01','2020-01-01 02:01']), 'itemid':[223834,226732,226732,223849], 'value':['2','Endotracheal tube','Endotracheal tube','CMV'], 'valuenum':[2,np.nan,np.nan,np.nan]})
    ox=oxygen_table(chart,True); direct=oxygen_table(chart,False); st=ventilation_settings(chart)
    assert len(ox)==1 and len(direct)==2
    assert len(classify_states(st,ox))==2 and len(classify_states(st,direct))==3
    print('Synthetic priority, hour-boundary gap, singleton, rate-change, terminal-tie and oxygen-flow join checks passed. No patient exports read.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--exports-verified',action='store_true')
    args=ap.parse_args()
    if args.self_test: self_test()
    else: main(args)
