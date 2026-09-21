"""Rebuild event populations and verify every draw by direct sampled-cluster summation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
Q=P.parent/'PHENOTYPE_DETERIORATION_20260907_v1'
base=pd.read_csv(Q/'derived/test_base.csv.gz',dtype={'subject_id':str,'file_name':str})
icu=pd.read_csv(Q/'derived/icu.csv.gz',dtype={'subject_id':str,'file_name':str})
support=pd.read_csv(Q/'derived/support.csv.gz',dtype={'subject_id':str,'file_name':str})
ref=pd.read_csv(P/'results/mortality_negative_care_escalation.csv')
checks=[]
for subset,endpoint in ref[['subset','endpoint']].drop_duplicates().itertuples(index=False):
    kind='icu' if endpoint.startswith('icu') else 'imv' if endpoint.startswith('imv') else 'pressor'
    source=icu if kind=='icu' else support
    eligibility='eligible' if kind=='icu' else kind+'_eligible'
    keep=(base.death_72h==0)&(source[eligibility]==1)
    event=source[endpoint].copy()
    if subset=='first_parent24':keep &= base.first_parent_encounter==1
    if subset=='landmark60_24':
        eligibility='landmark60_eligible' if kind=='icu' else kind+'_landmark60_eligible'
        event_column='landmark60_icu24' if kind=='icu' else kind+'_landmark60_24'
        keep &= source[eligibility]==1;event=source[event_column]
    d=base.loc[keep,['subject_id','fusion']].copy();d['event']=event[keep].astype(int)
    d['group']=pd.cut(d.fusion,bins=[-np.inf,.456303122639656,.6162797212600708,np.inf],
                      labels=['Low','Intermediate','High'],right=False)
    counts=pd.DataFrame({'subject_id':d.subject_id})
    for label in ['Low','Intermediate','High']:
        counts[label+'_n']=(d.group==label).astype(int)
        counts[label+'_e']=((d.group==label)&d.event.eq(1)).astype(int)
    aggregate=counts.groupby('subject_id',sort=True).sum().to_numpy()
    stored=np.load(P/'qa'/f'bootstrap_{subset}_{endpoint}.npz',allow_pickle=False)
    original=stored['draws'];nboot=int(stored['nboot']);seed=int(stored['seed'])
    rng=np.random.default_rng(seed);check=np.empty_like(original)
    for b in range(nboot):
        sampled=rng.choice(len(aggregate),len(aggregate),replace=True)
        z=aggregate[sampled].sum(axis=0)
        rates=z[[1,3,5]]/z[[0,2,4]]
        check[b]=[rates[0],rates[1],rates[2],rates[2]-rates[0],rates[1]-rates[0],z[[1,3,5]].sum()/z[[0,2,4]].sum()]
    maxdiff=float(np.max(np.abs(original-check)))
    assert maxdiff<1e-14
    for j,label in enumerate(['Low','Intermediate','High','High_minus_Low','Intermediate_minus_Low','Total']):
        r=ref[(ref.subset==subset)&(ref.endpoint==endpoint)&(ref.stratum_or_contrast==label)].iloc[0]
        ci=np.quantile(check[:,j],[.025,.975])
        assert np.max(np.abs(ci-[r.ci_low,r.ci_high]))<1e-14
    checks.append({'subset':subset,'endpoint':endpoint,'n':len(d),'patients':len(aggregate),
                   'replicates_checked':nboot,'max_absolute_difference':maxdiff,'seed':seed})
    print('Verified every draw:',subset,endpoint,flush=True)
(P/'qa/independent_bootstrap_check.json').write_text(json.dumps({'status':'PASS',
    'populations':len(checks),'replicates_checked':sum(x['replicates_checked'] for x in checks),
    'method':'Independent parent-input eligibility; pandas cluster sums; direct summation of sampled cluster rows',
    'checks':checks},indent=2)+'\n')
