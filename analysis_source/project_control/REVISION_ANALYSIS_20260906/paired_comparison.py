"""Paired patient bootstrap of frozen predictions; no model fitting or selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import ROOT,PACKAGE,SEED,N_BOOT,write_json,point_discrimination,log


def prepare(y,p,codes):
    order=np.argsort(-p,kind='stable');s=p[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    return y[order],codes[order],ends


def weighted_metrics(prepared,counts):
    y,cluster,ends=prepared
    w=counts[cluster]
    tp=np.cumsum(w*y)[ends];fp=np.cumsum(w*(1-y))[ends]
    recall=tp/tp[-1];fpr=fp/fp[-1]
    precision=np.divide(tp,tp+fp,out=np.ones_like(tp,dtype=float),where=(tp+fp)>0)
    return np.array([np.trapezoid(np.r_[0,recall],np.r_[0,fpr]),
                     np.dot(np.diff(np.r_[0,recall]),precision)])


def main():
    out=PACKAGE/'paired_comparison';out.mkdir(exist_ok=True)
    metrics=[];diffs=[];qa=[]
    for site in ['internal','external']:
        if site=='internal':
            ids=pd.read_csv(ROOT/'data/proba_test_72.csv',usecols=['subject_id'],dtype=str).subject_id
            fusion_path=PACKAGE/'batch1_verified/internal_test/evaluation_scores.npz'
            suffix='internal_test'
        else:
            ids=pd.read_csv(ROOT/'data/zs/中山结果_v2.csv',usecols=['subject_id'],dtype=str).subject_id
            fusion_path=PACKAGE/'batch1_verified/external_article_saved/evaluation_scores.npz'
            suffix='external_same_records'
        fusion=np.load(fusion_path);y=fusion['y']
        raw={'fusion':fusion['raw']};cal={'fusion':fusion['calibrated']}
        for model in ['clinical_logistic','clinical_xgboost','clinical_ecg_xgboost']:
            a=np.load(PACKAGE/'batch2'/model/suffix/'evaluation_scores.npz')
            assert np.array_equal(a['y'],y)
            raw[model]=a['raw'];cal[model]=a['calibrated']
        if site=='external':
            a=np.load(PACKAGE/'batch1_verified/external_original_score_sensitivity/evaluation_scores.npz')
            assert np.array_equal(a['y'],y)
            raw['fusion_original_score_sensitivity']=a['raw'];cal['fusion_original_score_sensitivity']=a['calibrated']
        codes,unique=pd.factorize(ids,sort=True);g=len(unique);names=list(raw)
        prepared={m:prepare(y,raw[m],codes) for m in names}
        ones=np.ones(g,dtype=int);point={m:weighted_metrics(prepared[m],ones) for m in names}
        components=np.zeros((g,len(names)+1));np.add.at(components[:,0],codes,1)
        for j,m in enumerate(names):
            np.add.at(components[:,j+1],codes,(cal[m]-y)**2)
            direct=point_discrimination(y,raw[m]);assert np.allclose(point[m],[direct['auroc'],direct['average_precision']],atol=1e-12,rtol=0)
            point[m]=np.r_[point[m],np.mean((cal[m]-y)**2)]
        rng=np.random.default_rng(SEED+200+(site=='external'))
        draws=np.empty((N_BOOT,len(names),3))
        for b in range(N_BOOT):
            counts=np.bincount(rng.integers(g,size=g),minlength=g)
            sums=counts@components
            for j,m in enumerate(names):
                draws[b,j,:2]=weighted_metrics(prepared[m],counts)
                draws[b,j,2]=sums[j+1]/sums[0]
            if (b+1)%250==0:
                log(f'{site}: paired discrimination bootstrap {b+1}/{N_BOOT}')
        assert np.isfinite(draws).all()
        for j,m in enumerate(names):
            for k,metric in enumerate(['auroc','average_precision','calibrated_brier']):
                lo,hi=np.quantile(draws[:,j,k],[.025,.975])
                metrics.append({'cohort':site,'model':m,'metric':metric,'estimate':point[m][k],
                                'ci_lower':lo,'ci_upper':hi,'n':len(y),'events':int(y.sum()),'patients':g})
                if m!='fusion':
                    delta=draws[:,0,k]-draws[:,j,k];lo,hi=np.quantile(delta,[.025,.975])
                    diffs.append({'cohort':site,'reference':'fusion','comparator':m,'metric':metric,
                                  'difference_fusion_minus_comparator':point['fusion'][k]-point[m][k],
                                  'ci_lower':lo,'ci_upper':hi,'interval_type':'unadjusted paired patient bootstrap 95%',
                                  'interpretation':'source sensitivity, not model comparison' if m.startswith('fusion_original') else 'secondary model comparison'})
        qa.append({'cohort':site,'replicates':N_BOOT,'patients':g,'point_metrics_match_sklearn':True,
                   'all_bootstrap_values_finite':True,'all_labels_same_order':True,'no_fitting_or_tuning':True})
    pd.DataFrame(metrics).to_csv(out/'model_metrics_with_ci.csv',index=False)
    pd.DataFrame(diffs).to_csv(out/'paired_differences.csv',index=False)
    write_json(out/'QA.json',{'analyses':qa,'ci_scope':'conditional on frozen fits, calibrators and thresholds',
                              'multiplicity':'secondary descriptive unadjusted intervals, no confirmatory superiority tests'})


if __name__=='__main__':
    main()
