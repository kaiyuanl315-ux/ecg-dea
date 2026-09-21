"""Exploratory fixed-score associations with recorded ICU organ support."""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from analyze import fit_assoc, sha, dump, PACKAGE, gzwrite

ENDPOINTS = ['imv24','imv72','pressor24','pressor72']

def support_strata(d, endpoint, seed, nboot=2000):
    y = d[endpoint].to_numpy(dtype=int)
    groups, unique = pd.factorize(d.subject_id, sort=True)
    columns = []
    for s in ['Low','Intermediate','High']:
        k = d.risk_stratum.eq(s).to_numpy(dtype=float)
        columns.extend([k,k*y])
    mat = np.column_stack(columns)
    sums = np.zeros((len(unique),6))
    np.add.at(sums,groups,mat)
    total = sums.sum(axis=0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(nboot):
        count = np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
        a = count@sums
        draws.append([a[2*i+1]/a[2*i] if a[2*i]>0 else np.nan for i in range(3)])
    draws = np.asarray(draws)
    rows = []
    for i,s in enumerate(['Low','Intermediate','High']):
        ci = np.nanquantile(draws[:,i],[.025,.975])
        n,e = total[2*i],total[2*i+1]
        rows.append({'endpoint':endpoint,'stratum':s,'n':int(n),'events':int(e),
                     'proportion':e/n,'ci_low':ci[0],'ci_high':ci[1],
                     'n_boot':nboot,'n_boot_valid':int(np.isfinite(draws[:,i]).sum()),'seed':seed})
    return rows

def run(out):
    for name in ['results','qa']: (out/name).mkdir(parents=True,exist_ok=True)
    quality = json.loads((PACKAGE/'qa/support_derivation.json').read_text())
    assert quality['exports_verified']
    base = pd.read_csv(PACKAGE/'derived/test_base.csv.gz',dtype={'file_name':str,'subject_id':str})
    support = pd.read_csv(PACKAGE/'derived/support.csv.gz',dtype={'file_name':str,'subject_id':str})
    for c in ['row_id','file_name','subject_id']: assert np.array_equal(base[c],support[c])
    data = base.merge(support[[c for c in support if c not in base or c=='row_id']],on='row_id',validate='one_to_one')
    transform = json.loads((PACKAGE/'qa/input_qa.json').read_text())['covariate_transform']
    rows, proportions = [], []
    for i,endpoint in enumerate(ENDPOINTS):
        kind = 'imv' if endpoint.startswith('imv') else 'pressor'
        hours = endpoint.replace(kind,'')
        d = data[data[kind+'_eligible'].eq(1)].copy()
        assert d[endpoint].notna().all()
        specs = [('ecg','M1','all'),('ecg','M2','all'),('fusion','M2','all'),
                 ('ecg','M2','first_encounter'),('ecg_single','M2','single_run'),
                 ('ecg','M2','landmark60')]
        sensitivities = [('raw_state','imv_raw_state'+hours)] if kind=='imv' else [
                        ('no_dopamine','pressor_no_dopamine'+hours),
                        ('no_bolus','pressor_no_bolus'+hours)]
        for score,stage,subset in specs:
            x = d.copy()
            if subset=='first_encounter': x=x[x.first_parent_encounter.eq(1)]
            if subset=='landmark60':
                x=x[x[kind+'_landmark60_eligible'].eq(1)].copy()
                x[endpoint]=x[kind+'_landmark60_'+hours]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('error',PerfectSeparationWarning)
                result=fit_assoc(x,endpoint,score,stage,transform,'support',subset)
                result['warning_count']=len(caught)
            rows.append(result)
        for subset,col in sensitivities:
            assert col in d, col
            x=d.copy(); x[endpoint]=x[col]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('error',PerfectSeparationWarning)
                result=fit_assoc(x,endpoint,'ecg','M2',transform,'support',subset)
                result['warning_count']=len(caught)
            rows.append(result)
        proportions.extend(support_strata(d,endpoint,20261001+i))
        print('Complete:',endpoint,flush=True)
    assoc=pd.DataFrame(rows)
    idx=assoc.index[assoc.score.eq('ecg')&assoc.adjustment.eq('M2')&assoc.subset.eq('all')&assoc.status.eq('ok')]
    assert len(idx)==4
    assoc.loc[idx,'q']=multipletests(assoc.loc[idx,'p'],method='fdr_bh')[1]
    assoc.to_csv(out/'results/support_associations.csv',index=False)
    pd.DataFrame(proportions).to_csv(out/'results/support_strata.csv',index=False)
    gzwrite(data,out/'qa/support_analytic_local.csv.gz')
    dump(out/'qa/support_statistical_qa.json',{'status':'PASS','n_parent':len(data),
        'all_models_converged':bool(assoc.status.eq('ok').all()),'warning_count':int(assoc.warning_count.sum()),
        'bootstrap_replicates':2000,'original_processed_covariates_preserved':True,
        'source_hashes':{str(p.relative_to(PACKAGE)):sha(p) for p in [
            PACKAGE/'derived/test_base.csv.gz',PACKAGE/'derived/support.csv.gz',
            PACKAGE/'qa/input_qa.json',PACKAGE/'SUPPORT_STATISTICAL_PLAN.md',
            PACKAGE/'scripts/analyze.py',Path(__file__)]},
        'output_hashes':{str(p.relative_to(out)):sha(p) for p in [out/'results/support_associations.csv',out/'results/support_strata.csv']}})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=PACKAGE)
    args=ap.parse_args();run(args.out)
