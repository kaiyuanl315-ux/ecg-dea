"""Prespecified exploratory fits and clustered descriptive proportions."""
from pathlib import Path
import argparse,json,warnings,platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from prepare import PACKAGE,PARENT,NONCARDIAC,prepare,dump,sha

SUBSETS=['principal24','secondary72','first_parent24','landmark60_24']
ENDPOINTS=['icu','imv','pressor']
STRATA=['Low','Intermediate','High']

def design(d,score,stage,transform):
    columns={'intercept':np.ones(len(d)),'score_z':d[score+'_z'].to_numpy(),
             'male':d.gender_clean.to_numpy()}
    covariates=['age']+(['temperature_c','resprate','o2sat','sbp'] if stage=='M2' else [])
    for name in covariates:
        t=transform[name];x=d[name+'_imp'].to_numpy()
        columns[name]=(x-t['mean'])/t['sd']
        if t['nonlinear']:
            a,b,c=t['knots']
            spline=(np.maximum(x-a,0)**3-np.maximum(x-b,0)**3*(c-a)/(c-b)+
                    np.maximum(x-c,0)**3*(b-a)/(c-b))/(c-a)**2
            columns[name+'_nonlinear']=spline/t['sd']
    X=pd.DataFrame(columns,index=d.index)
    assert np.isfinite(X).all().all()
    return X

def fit_model(frame,outcome,score,stage,subset,transform):
    d=frame[frame.covariates_observed.eq(1)&frame[outcome].notna()]
    y=d[outcome].astype(int).to_numpy();x=design(d,score,stage,transform)
    r={'subset':subset,'endpoint':outcome,'score':score,'adjustment':stage,
       'n':len(d),'patients':d.subject_id.nunique(),'events':int(y.sum()),
       'design_columns':';'.join(x.columns),'n_parameters':x.shape[1],
       'rank':int(np.linalg.matrix_rank(x)), 'condition_number':float(np.linalg.cond(x)),
       'score_min':float(d[score+'_z'].min()),'score_max':float(d[score+'_z'].max())}
    if min(y.sum(),len(y)-y.sum())<30:return {**r,'status':'insufficient_events'}
    if r['rank']!=x.shape[1]:return {**r,'status':'rank_deficient'}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warnings.simplefilter('error',PerfectSeparationWarning)
            fit=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10,
                cov_type='cluster',cov_kwds={'groups':d.subject_id.to_numpy(),'use_correction':True})
        r['warnings']=' | '.join(str(w.message) for w in caught)
    except PerfectSeparationWarning:
        return {**r,'status':'separation_warning'}
    if not fit.converged:return {**r,'status':'not_converged'}
    beta=float(fit.params['score_z']);se=float(fit.bse['score_z'])
    if not np.isfinite(fit.params).all() or not np.isfinite(fit.bse).all() or se<=0:
        return {**r,'status':'nonfinite_estimate'}
    prob=np.asarray(fit.fittedvalues)
    return {**r,'status':'ok','beta':beta,'se':se,'or':np.exp(beta),
        'ci_low':np.exp(beta-norm.ppf(.975)*se),'ci_high':np.exp(beta+norm.ppf(.975)*se),
        'p':2*norm.sf(abs(beta/se)),'q':np.nan,'iterations':fit.fit_history['iteration'],
        'min_fitted_probability':float(prob.min()),'max_fitted_probability':float(prob.max()),
        'extreme_fitted_n':int(((prob<1e-8)|(prob>1-1e-8)).sum())}

def restricted_cohorts(d):
    p=d[d.diagnosis_observed.eq(1)].copy();a=p[p.no_recorded_cardiac.eq(1)]
    return [('full_reference',p,'ecg','M2'),('exclude_any_cardiac',a,'ecg','M2'),
        ('exclude_any_cardiac_M1',a,'ecg','M1'),('exclude_AMI',p[p.ami.eq(0)],'ecg','M2'),
        ('exclude_HF',p[p.heart_failure.eq(0)],'ecg','M2'),
        ('exclude_AF',p[p.af_flutter.eq(0)],'ecg','M2'),
        ('exclude_any_cardiac_first_parent',a[a.first_parent_encounter.eq(1)],'ecg','M2'),
        ('exclude_any_cardiac_single_run',a,'ecg_single','M2')]

def eligible_negative(d,subset,kind):
    eligible='eligible' if kind=='icu' else kind+'_eligible'
    q=d[d.death_72h.eq(0)&d[eligible].eq(1)].copy()
    hours=72 if subset=='secondary72' else 24
    endpoint=kind+str(hours)
    if subset=='first_parent24':q=q[q.first_parent_encounter.eq(1)]
    if subset=='landmark60_24':
        flag='landmark60_eligible' if kind=='icu' else kind+'_landmark60_eligible'
        column='landmark60_icu24' if kind=='icu' else kind+'_landmark60_24'
        q=q[q[flag].eq(1)].copy();q[endpoint]=q[column]
    assert q[endpoint].isin([0,1]).all()
    return q,endpoint

def proportions(d,endpoint,subset,seed,nboot,out):
    ids,patients=pd.factorize(d.subject_id,sort=True)
    y=d[endpoint].to_numpy(dtype=int)
    cats=pd.Categorical(d.risk_stratum,categories=STRATA).codes
    assert np.all(cats>=0)
    sums=np.zeros((len(patients),6),dtype=np.int64)
    np.add.at(sums,(ids,cats*2),1);np.add.at(sums,(ids,cats*2+1),y)
    total=sums.sum(axis=0);point=total[1::2]/total[::2]
    rng=np.random.default_rng(seed)
    draws=np.empty((nboot,6))
    for b in range(nboot):
        weights=np.bincount(rng.integers(len(patients),size=len(patients)),minlength=len(patients))
        v=weights@sums;p=np.divide(v[1::2],v[::2],out=np.full(3,np.nan),where=v[::2]>0)
        draws[b,:3]=p;draws[b,3]=p[2]-p[0];draws[b,4]=p[1]-p[0]
        draws[b,5]=v[1::2].sum()/v[::2].sum()
    names=STRATA+['High_minus_Low','Intermediate_minus_Low','Total']
    np.savez_compressed(out/'qa'/f'bootstrap_{subset}_{endpoint}.npz',draws=draws,seed=seed,nboot=nboot)
    rows=[]
    for i,cat in enumerate(names):
        values=draws[:,i];valid=np.isfinite(values)
        lo,hi=np.quantile(values[valid],[.025,.975])
        if i<3:
            n,e=total[2*i:2*i+2];estimate=point[i];g=d.loc[cats==i,'subject_id'].nunique()
            effect='proportion'
        elif cat=='Total':
            n=len(d);e=int(y.sum());estimate=y.mean();g=len(patients);effect='proportion'
        else:
            n=len(d);e=int(y.sum());estimate=point[2 if i==3 else 1]-point[0];g=len(patients)
            effect='absolute_difference'
        rows.append({'subset':subset,'endpoint':endpoint,'stratum_or_contrast':cat,'effect':effect,
            'n':int(n),'patients':int(g),'events':int(e),'estimate':estimate,'ci_low':lo,'ci_high':hi,
            'n_boot':nboot,'n_boot_valid':int(valid.sum()),'seed':seed})
    return rows

def run(out,nboot=2000):
    assert nboot==2000,'The frozen plan requires 2000 draws.'
    d,transform=prepare(out)
    rows=[]
    for subset,frame,score,stage in restricted_cohorts(d):
        for endpoint in NONCARDIAC:rows.append(fit_model(frame,endpoint,score,stage,subset,transform))
        print('A complete:',subset,flush=True)
    a=pd.DataFrame(rows)
    principal=a.index[a.subset.eq('exclude_any_cardiac')]
    if a.loc[principal,'status'].eq('ok').all():
        a.loc[principal,'q']=multipletests(a.loc[principal,'p'],method='fdr_bh')[1]
    else:
        # Keep the prespecified family of three by assigning p=1 to non-estimable tests.
        ps=a.loc[principal,'p'].fillna(1)
        corrected=multipletests(ps,method='fdr_bh')[1]
        for idx,q in zip(principal,corrected):
            if a.loc[idx,'status']=='ok':a.loc[idx,'q']=q
    parent=pd.read_csv(PARENT/'results/associations.csv')
    parent=parent[parent.domain.eq('phenotype')&parent.score.eq('ecg')&parent.adjustment.eq('M2')&parent.subset.eq('all')]
    refs=[]
    for idx,r in a[a.subset.eq('full_reference')].iterrows():
        old=parent[parent.endpoint.eq(r.endpoint)].iloc[0]
        assert r.status=='ok'
        for col in ['n','patients','events']:assert r[col]==old[col]
        for col in ['beta','se','or','ci_low','ci_high']:assert abs(r[col]-old[col])<1e-8,(r.endpoint,col)
        a.loc[idx,'q']=old.q
        refs.append({'endpoint':r.endpoint,'beta_difference':r.beta-old.beta,'se_difference':r.se-old.se})
    a.to_csv(out/'results/phenotype_restriction_associations.csv',index=False)
    pd.DataFrame(refs).to_csv(out/'qa/parent_reference_reproduction.csv',index=False)
    b=[]
    for si,subset in enumerate(SUBSETS):
        for ei,kind in enumerate(ENDPOINTS):
            q,endpoint=eligible_negative(d,subset,kind)
            b.extend(proportions(q,endpoint,subset,202609070+10*si+ei,nboot,out))
            print('B complete:',subset,endpoint,'eligible',len(q),flush=True)
    pd.DataFrame(b).to_csv(out/'results/mortality_negative_care_escalation.csv',index=False)
    dump(out/'qa/statistical_qa.json',{'status':'PASS' if a.status.eq('ok').all() else 'PARTIAL_ESTIMABILITY',
        'new_models':21,'parent_reference_models':3,'all_fits_ok':bool(a.status.eq('ok').all()),
        'bootstrap_populations':12,'bootstrap_draws_each':2000,
        'all_bootstrap_draws_valid':bool(pd.DataFrame(b).n_boot_valid.eq(2000).all()),
        'model_status_counts':a.status.value_counts().to_dict(),
        'warning_rows':a[a.warnings.fillna('').ne('')][['subset','endpoint','warnings']].to_dict('records'),
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'scipy':scipy.__version__,'statsmodels':sm.__version__},
        'plan_sha256':sha(PACKAGE/'ANALYSIS_PLAN.md'),
        'script_hashes':{str(p.relative_to(PACKAGE)):sha(p) for p in [Path(__file__),PACKAGE/'scripts/prepare.py']}})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=PACKAGE)
    run(ap.parse_args().out)
