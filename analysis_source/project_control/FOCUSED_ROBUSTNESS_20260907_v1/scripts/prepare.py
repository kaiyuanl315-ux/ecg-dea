"""Verify frozen inputs and build a local analysis frame without modifying parent data."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
PARENT = ROOT / 'project_control/PHENOTYPE_DETERIORATION_20260907_v1'
CARDIAC = ['ami', 'heart_failure', 'af_flutter']
NONCARDIAC = ['sepsis', 'respiratory_failure', 'aki']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()

def native(x):
    if isinstance(x, dict): return {str(k): native(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [native(v) for v in x]
    if isinstance(x, np.ndarray): return native(x.tolist())
    if isinstance(x, (np.integer,np.bool_)): return x.item()
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, Path): return str(x)
    return x

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def prepare(out):
    for name in ['derived','results','qa','figures']: (out/name).mkdir(parents=True, exist_ok=True)
    freeze = json.loads((PACKAGE/'qa/PLAN_FREEZE.json').read_text())
    assert freeze['status'] == 'FROZEN_BEFORE_NEW_ESTIMATES'
    assert sha(PACKAGE/'ANALYSIS_PLAN.md') == freeze['plan_sha256']
    for name, expected in freeze['source_hashes'].items(): assert sha(ROOT/name) == expected, name
    for name, expected in freeze['manuscript_hashes'].items(): assert sha(name) == expected, name
    lock = json.loads((PARENT/'RESULT_LOCK.json').read_text())
    checked = []
    for name in ['derived/test_base.csv.gz','derived/phenotypes.csv.gz','derived/icu.csv.gz',
                 'derived/support.csv.gz','qa/input_qa.json','results/associations.csv']:
        assert sha(PARENT/name) == lock['files'][name], name
        checked.append(name)
    dfs = {name: pd.read_csv(PARENT/'derived'/f'{name}.csv.gz',
        dtype={'subject_id':str,'file_name':str}) for name in ['test_base','phenotypes','icu','support']}
    base = dfs['test_base']
    assert len(base)==35948 and int(base.death_72h.sum())==302
    assert base.row_id.is_unique and base.file_name.is_unique
    assert base.death_72h.isin([0,1]).all()
    for name, frame in dfs.items():
        assert frame.row_id.is_unique
        for key in ['row_id','file_name','subject_id']: assert np.array_equal(base[key],frame[key]), (name,key)
    d = base.copy()
    for name in ['phenotypes','icu','support']:
        frame = dfs[name]
        # Consume new endpoint columns; overlapping metadata remain the locked base values.
        columns = [c for c in frame if c not in d or c=='row_id']
        d = d.merge(frame[columns], on='row_id', validate='one_to_one', sort=False)
    for col in base: assert d[col].equals(base[col]), col
    score_scales = lock['score_scaling']
    for name in ['ecg','fusion','ecg_single']:
        p = np.clip(d[name].to_numpy(),1e-8,1-1e-8)
        z = (np.log(p/(1-p))-score_scales[name]['mean'])/score_scales[name]['sd']
        assert np.max(np.abs(z-d[name+'_z']))<1e-11
    strata = np.select([d.fusion<.456303122639656,d.fusion<.6162797212600708],
                        ['Low','Intermediate'],default='High')
    assert np.array_equal(strata,d.risk_stratum)
    first = d.assign(check_time=pd.to_datetime(d.index_time)).sort_values(['check_time','row_id']).drop_duplicates('subject_id').row_id
    assert np.array_equal(d.row_id.isin(first).astype(int),d.first_parent_encounter)
    observed = d.diagnosis_observed.eq(1)
    assert d.loc[observed,CARDIAC+NONCARDIAC].isin([0,1]).all().all()
    assert int(observed.sum())==35906
    # Do not turn missing ascertainment into absence of a cardiac diagnosis.
    d['no_recorded_cardiac'] = (observed & d[CARDIAC].eq(0).all(axis=1)).astype(int)
    assert d.no_recorded_cardiac.sum()==26239
    flow = []
    def add(domain, stage, f):
        flow.append({'domain':domain,'stage':stage,'n':len(f),'patients':f.subject_id.nunique(),
                     'mortality_positive':int(f.death_72h.sum())})
    add('A','parent',d)
    add('A','diagnoses_ascertainable',d[observed])
    add('A','no_AMI_HF_AF',d[d.no_recorded_cardiac.eq(1)])
    add('A','no_AMI_HF_AF_adjusted',d[d.no_recorded_cardiac.eq(1)&d.covariates_observed.eq(1)])
    add('B','parent',d)
    negative=d[d.death_72h.eq(0)]
    add('B','original_mortality_negative',negative)
    for endpoint,eligible in [('icu','eligible'),('imv','imv_eligible'),('pressor','pressor_eligible')]:
        f=d[d[eligible].eq(1)]
        assert f[[endpoint+'24',endpoint+'72']].isin([0,1]).all().all()
        assert (f[endpoint+'24']<=f[endpoint+'72']).all()
        add('B',endpoint+'_parent_eligible',f)
        f=f[f.death_72h.eq(0)]
        add('B',endpoint+'_mortality_negative_eligible',f)
        add('B',endpoint+'_mortality_negative_first_parent',f[f.first_parent_encounter.eq(1)])
        landmark='landmark60_eligible' if endpoint=='icu' else endpoint+'_landmark60_eligible'
        add('B',endpoint+'_mortality_negative_landmark60',f[f[landmark].eq(1)])
    assert d.loc[d.imv_eligible.eq(1)|d.pressor_eligible.eq(1),'ICU_source_coverage_gap'].eq(False).all()
    measures=['age','temperature_c','resprate','o2sat','sbp']
    missing=[]
    for col in measures:
        assert np.array_equal(d[col],d[col+'_imp'])
        assert np.isfinite(d[col+'_imp']).all()
        missing.append({'variable':col,'nonfinite_processed':int((~np.isfinite(d[col+'_imp'])).sum()),
                        'missing_indicator_sum':int(d[col+'_missing'].sum()),'original_values_preserved':True})
    assert int(d.covariates_observed.eq(0).sum())==1
    # Label audit is descriptive; no exclusions or replacement of locked mortality labels.
    death=pd.to_datetime(d.death_time,errors='coerce')
    arrival=pd.to_datetime(d.edregtime,errors='raise'); index=pd.to_datetime(d.index_time,errors='raise')
    audit=[]
    for eligibility in ['all_parent','eligible','imv_eligible','pressor_eligible']:
        mask=d.death_72h.eq(0)
        if eligibility!='all_parent': mask &= d[eligibility].eq(1)
        x={'population':eligibility,'mortality_negative_n':int(mask.sum()),
           'recorded_death_any_time':int((mask&death.notna()).sum())}
        for origin,anchor in [('arrival',arrival),('ECG',index)]:
            hours=(death-anchor).dt.total_seconds()/3600
            for h in [24,72]:x[f'recorded_death_{origin}_{h}h']=int((mask&hours.ge(0)&hours.le(h)).sum())
        audit.append(x)
    pd.DataFrame(flow).to_csv(out/'results/cohort_flow.csv',index=False)
    pd.DataFrame(missing).to_csv(out/'qa/processed_covariate_audit.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'qa/mortality_label_time_audit.csv',index=False)
    d.to_csv(out/'derived/analysis_local.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    ascertain=d[observed]
    pattern=ascertain.groupby(CARDIAC,dropna=False).agg(n=('row_id','size'),patients=('subject_id','nunique'),
        sepsis=('sepsis','sum'),respiratory_failure=('respiratory_failure','sum'),aki=('aki','sum')).reset_index()
    pattern.to_csv(out/'results/cardiac_overlap_counts.csv',index=False)
    exclusion=[]
    for col in ['exclusion_reason','imv_exclusion_reason','pressor_exclusion_reason']:
        for label,n in negative[col].fillna('eligible_or_no_reason').value_counts().items():
            exclusion.append({'reason_column':col,'reason':label,'n':int(n)})
    pd.DataFrame(exclusion).to_csv(out/'results/mortality_negative_exclusion_counts.csv',index=False)
    dump(out/'qa/preparation_qa.json',{'status':'PASS','parent_lock_files_verified':checked,
        'parent_rows':len(d),'parent_patients':d.subject_id.nunique(),'parent_mortality_events':int(d.death_72h.sum()),
        'one_to_one_identities_verified':True,'scores_and_original_covariates_unchanged':True,
        'original_first_parent_selection_verified':True,'new_missing_data_processing':False,
        'parent_test_sha256':sha(PARENT/'derived/test_base.csv.gz'),
        'analytic_sha256':sha(out/'derived/analysis_local.csv.gz'),'plan_sha256':freeze['plan_sha256']})
    return d,lock['covariate_transforms']

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=PACKAGE)
    prepare(ap.parse_args().out)
