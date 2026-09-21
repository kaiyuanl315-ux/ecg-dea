"""Versioned score diagnostics; frozen calibration is fitted on validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
import sklearn
from sklearn.metrics import (average_precision_score, auc, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
import statsmodels.api as sm

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
EPS = 1e-8
SEED = 20260906
N_BOOT = 2000


def log(s):
    print(s, flush=True)


def native(o):
    if isinstance(o, dict):
        return {str(k): native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [native(x) for x in o]
    if isinstance(o, np.ndarray):
        return native(o.tolist())
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(native(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity_hash(d, cols):
    h = hashlib.sha256()
    for row in d[cols].astype(str).itertuples(index=False, name=None):
        h.update(('\t'.join(row) + '\n').encode())
    return h.hexdigest()


def pclip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def check_data(y, p, ids):
    assert len(y) == len(p) == len(ids) and len(y) > 0
    assert np.isin(y, [0, 1]).all() and 0 < np.sum(y) < len(y)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert not pd.isna(ids).any()


def stable_cluster_ci(fit, design, y, ids):
    """Expected-information sandwich avoids observed-Hessian overflow at p≈1."""
    mu = np.asarray(fit.fittedvalues)
    residual = np.asarray(y)-mu
    info = design.T @ ((mu*(1-mu))[:, None]*design)
    bread = np.linalg.inv(info)
    codes, unique = pd.factorize(ids, sort=True)
    scores = np.zeros((len(unique), design.shape[1]))
    np.add.at(scores, codes, design*residual[:,None])
    n,k = design.shape
    correction = len(unique)/(len(unique)-1)*(n-1)/(n-k)
    cov = correction*bread @ (scores.T@scores) @ bread
    se = np.sqrt(np.diag(cov))
    ci = np.column_stack([fit.params-1.959963984540054*se,fit.params+1.959963984540054*se])
    assert np.isfinite(ci).all()
    return ci


def diagnostic_fit(y, p, ids):
    """Evaluation-only regression. Its coefficients must NEVER update scores."""
    x = logit(pclip(p))
    design = sm.add_constant(x, has_constant='add')
    fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(maxiter=100, tol=1e-10)
    ones = np.ones((len(y), 1))
    citl = sm.GLM(y, ones, offset=x,
                  family=sm.families.Binomial()).fit(maxiter=100, tol=1e-10)
    assert fit.converged and citl.converged
    ci, ci0 = stable_cluster_ci(fit,design,y,ids), stable_cluster_ci(citl,ones,y,ids)
    return {
        'calibration_intercept_joint': (fit.params[0], ci[0, 0], ci[0, 1]),
        'calibration_slope': (fit.params[1], ci[1, 0], ci[1, 1]),
        'calibration_in_the_large': (citl.params[0], ci0[0, 0], ci0[0, 1]),
    }


def fit_calibrator(y, p, ids):
    check_data(y, p, ids)
    x = sm.add_constant(logit(pclip(p)), has_constant='add')
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100, tol=1e-10)
    assert fit.converged and fit.params[1] > 0
    return {'intercept': float(fit.params[0]), 'slope': float(fit.params[1]),
            'coefficient_ci95_cluster_robust': stable_cluster_ci(fit,x,y,ids).tolist(),
            'fit_n': len(y), 'fit_events': int(np.sum(y)),
            'fit_patients': int(pd.Series(ids).nunique()), 'clip_epsilon': EPS,
            'formula': 'expit(intercept + slope * logit(clip(raw_score)))',
            'fitted_on': 'internal_validation_only', 'fit_weights': 'none',
            'regularization': 'none', 'converged': True}


def calibrate(p, model):
    return expit(model['intercept'] + model['slope'] * logit(pclip(p)))


def select_thresholds(y, p):
    fpr, sens, thresholds = roc_curve(y, p)
    valid = np.flatnonzero(sens >= .90)
    assert len(valid)
    low = float(thresholds[valid[np.argmax(1 - fpr[valid])]])
    precision, recall, thresholds = precision_recall_curve(y, p)
    valid = np.flatnonzero(recall[:-1] >= .50)
    assert len(valid)
    high = float(thresholds[valid[np.argmax(precision[:-1][valid])]])
    assert np.isfinite(low) and np.isfinite(high) and low < high
    return {'low': low, 'high': high}


def point_discrimination(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return {'auroc': roc_auc_score(y, p), 'average_precision': average_precision_score(y, p),
            'auprc_trapezoidal': auc(recall, precision)}


def ratio(a, b):
    return a / b if b > 0 else np.nan


def metrics_from_sums(t):
    n, e = t[0], t[1]
    prev = e / n
    out = {'prevalence': prev}
    for state, k in [('raw', 2), ('calibrated', 5)]:
        pred, se, ll = t[k:k+3]
        bs = se / n
        ref = prev * (1-prev)
        out.update({state + '/' + name: v for name, v in {
            'mean_predicted': pred/n, 'brier': bs, 'null_brier': ref,
            'brier_skill': 1-bs/ref, 'observed_expected_ratio': ratio(e, pred),
            'log_loss': ll/n}.items()})
    for name, k in [('low', 8), ('high', 10)]:
        tp, fp = t[k:k+2]
        fn, tn = e-tp, n-e-fp
        alerts = tp+fp
        d = {'sensitivity': ratio(tp,e), 'specificity': ratio(tn,n-e),
             'false_negative_rate': ratio(fn,e), 'false_positive_rate': ratio(fp,n-e),
             'ppv': ratio(tp,alerts), 'npv': ratio(tn,tn+fn),
             'false_discovery_rate': ratio(fp,alerts), 'alert_fraction': alerts/n,
             'alerts_per_1000': alerts/n*1000, 'false_alerts_per_1000': fp/n*1000,
             'deaths_detected_per_1000': tp/n*1000, 'deaths_missed_per_1000': fn/n*1000,
             'deaths_per_100_alerts': ratio(tp,alerts)*100,
             'alerts_per_death_detected': ratio(alerts,tp)}
        out.update({name+'/'+m: v for m,v in d.items()})
    out['paired/brier_calibrated_minus_raw'] = (t[6]-t[3])/n
    return out


def aggregate_components(y, p, pc, ids, thresholds):
    y = np.asarray(y, dtype=float)
    a, b = pclip(p), pclip(pc)
    lp = -(y*np.log(a)+(1-y)*np.log1p(-a))
    lc = -(y*np.log(b)+(1-y)*np.log1p(-b))
    q0, q1 = p >= thresholds['low'], p >= thresholds['high']
    obs = np.column_stack([np.ones(len(y)), y, p, (p-y)**2, lp,
                           pc, (pc-y)**2, lc, y*q0, (1-y)*q0, y*q1, (1-y)*q1])
    codes, unique = pd.factorize(ids, sort=True)
    sums = np.zeros((len(unique), obs.shape[1]))
    np.add.at(sums, codes, obs)
    return sums, obs.sum(axis=0)


def cluster_bootstrap(sums, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    g = len(sums)
    draws = []
    for b in range(n_boot):
        counts = np.bincount(rng.integers(g, size=g), minlength=g)
        draws.append(metrics_from_sums(counts @ sums))
        if (b+1) % 500 == 0:
            log(f'  patient bootstrap {b+1}/{n_boot}')
    df = pd.DataFrame(draws)
    intervals = {c: np.nanquantile(df[c], [.025,.975]).tolist() for c in df}
    return intervals, {'replicates': n_boot, 'seed': seed, 'patient_clusters': g,
                       'invalid_replicates_by_metric': df.isna().sum().to_dict()}


def evaluate(y, p, ids, name, model, thresholds, bin_edges, out, n_boot=N_BOOT, seed=SEED):
    log(f'Evaluating {name}: n={len(y)}, events={int(np.sum(y))}')
    y, p, ids = np.asarray(y,dtype=int), np.asarray(p,dtype=float), np.asarray(ids)
    check_data(y,p,ids)
    pc = calibrate(p,model)
    patients = pd.Series(ids).nunique()
    invariant = {k: bool(np.array_equal(p>=v, pc>=calibrate([v],model)[0]))
                 for k,v in thresholds.items()}
    assert all(invariant.values())
    disc_raw, disc_cal = point_discrimination(y,p), point_discrimination(y,pc)
    assert max(abs(disc_raw[k]-disc_cal[k]) for k in disc_raw) < 1e-12
    sums, totals = aggregate_components(y,p,pc,ids,thresholds)
    point = metrics_from_sums(totals)
    assert abs(point['raw/brier']-brier_score_loss(y,p)) < 1e-10
    assert abs(point['calibrated/brier']-brier_score_loss(y,pc)) < 1e-10
    intervals, bootqa = cluster_bootstrap(sums,n_boot,seed)
    metrics, threshold_rows, bins, strata = [], [], [], []
    common = {'cohort': name, 'n': len(y), 'events': int(y.sum()), 'patients': patients}
    for state, ps in [('raw',p), ('calibrated',pc)]:
        for key,v in point.items():
            if key.startswith(state+'/'):
                lo,hi=intervals[key]
                metrics.append({**common,'score_state':state,'metric':key.split('/')[1],
                                'estimate':v,'ci_lower':lo,'ci_upper':hi,
                                'ci_method':'patient_cluster_bootstrap_conditional'})
        for key,v in point_discrimination(y,ps).items():
            metrics.append({**common,'score_state':state,'metric':key,'estimate':v,
                            'ci_lower':np.nan,'ci_upper':np.nan,'ci_method':'point_estimate_only'})
        for key,(v,lo,hi) in diagnostic_fit(y,ps,ids).items():
            metrics.append({**common,'score_state':state,'metric':key,'estimate':v,
                            'ci_lower':lo,'ci_upper':hi,'ci_method':'patient_cluster_robust_wald'})
    for key in ['prevalence','paired/brier_calibrated_minus_raw']:
        lo,hi=intervals[key]
        metrics.append({**common,'score_state':'paired' if key.startswith('paired') else 'observed',
                        'metric':key.split('/')[-1],'estimate':point[key],
                        'ci_lower':lo,'ci_upper':hi,'ci_method':'patient_cluster_bootstrap_conditional'})
    for k,t in thresholds.items():
        q=p>=t
        counts={'tp':int(np.sum(q&(y==1))), 'fp':int(np.sum(q&(y==0))),
                'tn':int(np.sum(~q&(y==0))), 'fn':int(np.sum(~q&(y==1)))}
        assert sum(counts.values())==len(y)
        row={**common,'threshold_name':k,'threshold_raw':t,
             'threshold_calibrated':float(calibrate([t],model)[0]),**counts,
             'alerts_n':counts['tp']+counts['fp'], 'nonalerts_n':counts['tn']+counts['fn']}
        for key,v in point.items():
            if key.startswith(k+'/'):
                metric=key.split('/')[1]
                row.update({metric:v,metric+'_ci_lower':intervals[key][0],metric+'_ci_upper':intervals[key][1]})
        threshold_rows.append(row)
    groups=np.where(p<thresholds['low'],'low',np.where(p>=thresholds['high'],'high','intermediate'))
    for g in ['low','intermediate','high']:
        s=groups==g
        strata.append({**common,'risk_group':g,'group_n':int(s.sum()),
                       'group_events':int(y[s].sum()),'group_fraction':s.mean(),
                       'group_event_rate':float(y[s].mean()) if s.any() else None})
    bin_ids=np.searchsorted(bin_edges[1:-1],p,side='right')
    for b in range(len(bin_edges)-1):
        s=bin_ids==b
        if not s.any():
            continue
        for state,ps in [('raw',p),('calibrated',pc)]:
            bins.append({**common,'score_state':state,'bin':b+1,'bin_n':int(s.sum()),
                         'bin_events':int(y[s].sum()),'observed_rate':float(y[s].mean()),
                         'mean_predicted':float(ps[s].mean()),
                         'raw_edge_low':float(bin_edges[b]),'raw_edge_high':float(bin_edges[b+1])})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(out/'calibration_metrics.csv',index=False)
    pd.DataFrame(threshold_rows).to_csv(out/'threshold_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'risk_strata.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    # These arrays contain no names, patient IDs, dates, or free-text records.
    np.savez_compressed(out/'evaluation_scores.npz',y=y,raw=p,calibrated=pc)
    qa={**common,'threshold_classification_invariant':invariant,
        'auroc_ap_pr_area_invariant':True,'calibration_diagnostic_fit_never_applied':True,
        'bootstrap':bootqa,'both_brier_independently_checked_with_sklearn':True,
        'raw_score_at_clip_limits':int(((p<EPS)|(p>1-EPS)).sum()),
        'calibrated_at_clip_limits':int(((pc<EPS)|(pc>1-EPS)).sum()),
        'numeric_arrays_only_no_patient_identifiers':True}
    write_json(out/'QA.json',qa)
    return metrics,threshold_rows,bins,strata,qa


def load_internal_splits():
    cols=['file_name','subject_id','death_72h','general_strat_fold']
    main=pd.read_csv(ROOT/'data/ecg_muti.csv',usecols=cols,dtype={'file_name':str,'subject_id':str})
    assert not main.file_name.duplicated().any()
    parts={}
    for split,folds in [('train',range(7)),('val',[7]),('test',[8,9])]:
        p=pd.read_csv(ROOT/f'data/proba_{split}_72.csv',dtype={'file_name':str,'subject_id':str})
        assert not p.file_name.duplicated().any()
        m=p.merge(main,on='file_name',validate='one_to_one',how='left',suffixes=('','_source'))
        assert m.general_strat_fold.isin(folds).all()
        assert np.array_equal(m.subject_id,m.subject_id_source)
        assert np.array_equal(m.death_72h,m.death_72h_source)
        runs=[f'proba_run{i}' for i in range(10)]
        assert np.max(np.abs(p[runs].mean(axis=1)-p.proba_mean))<1e-6
        parts[split]=p
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not set(parts[a].subject_id)&set(parts[b].subject_id)
    assert sum(map(len,parts.values()))==len(main)
    return parts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=PACKAGE/'batch1')
    parser.add_argument('--n-boot',type=int,default=N_BOOT)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/f'data/proba_{s}_72.csv' for s in ['train','val','test']]+[
        ROOT/'data/ecg_muti.csv',ROOT/'data/zs/中山结果_v2.csv',PACKAGE/'ANALYSIS_PLAN.md',Path(__file__)]
    before={str(p):sha(p) for p in paths}
    write_json(out/'source_manifest.json',before)
    parts=load_internal_splits();v=parts['val']
    y=v.death_72h.to_numpy();p=v.proba_mean.to_numpy();ids=v.subject_id.to_numpy()
    model=fit_calibrator(y,p,ids);thresholds=select_thresholds(y,p)
    assert abs(thresholds['low']-.45630312)<1e-8 and abs(thresholds['high']-.61627972)<1e-8
    edges=np.unique(np.r_[0,np.quantile(p,np.arange(.1,1,.1)),1])
    lock={'created_utc':datetime.now(timezone.utc).isoformat(),'model':model,
          'thresholds_raw':thresholds,'thresholds_calibrated':{k:float(calibrate([t],model)[0]) for k,t in thresholds.items()},
          'validation_bin_edges':edges,'fit_source_hash':before[str(ROOT/'data/proba_val_72.csv')],
          'validation_record_order_hash':identity_hash(v,['file_name','subject_id']),
          'method_predefined_before_new_test_evaluation':True,
          'external_or_internal_test_used_to_fit_or_select':False}
    write_json(out/'CALIBRATOR_AND_THRESHOLDS_FROZEN.json',lock)
    lock_hash=sha(out/'CALIBRATOR_AND_THRESHOLDS_FROZEN.json')
    log('Calibration and validation thresholds frozen; starting evaluation, no further fitting of deployment parameters.')
    metrics=[];threshold_rows=[];bins=[];strata=[];qas=[]
    test=parts['test']
    r=evaluate(test.death_72h.to_numpy(),test.proba_mean.to_numpy(),test.subject_id.to_numpy(),
               'internal_test',model,thresholds,edges,out/'internal_test',args.n_boot,SEED)
    for bucket,new in zip([metrics,threshold_rows,bins,strata,qas],r):
        bucket.extend(new if isinstance(new,list) else [new])
    e=pd.read_csv(ROOT/'data/zs/中山结果_v2.csv',usecols=['subject_id','study_id','visit_id',
                   'proba_mean_1','proba_mean_2','death_72h_or_triage'],
                  dtype={'subject_id':str,'study_id':str,'visit_id':str})
    assert not e.visit_id.duplicated().any()
    for name,score in [('external_article_saved','proba_mean_2'),('external_original_score_sensitivity','proba_mean_1')]:
        r=evaluate(e.death_72h_or_triage.to_numpy(),e[score].to_numpy(),e.subject_id.to_numpy(),
                   name,model,thresholds,edges,out/name,args.n_boot,SEED+1)
        for bucket,new in zip([metrics,threshold_rows,bins,strata,qas],r):
            bucket.extend(new if isinstance(new,list) else [new])
    for filename,rows in [('calibration_metrics',metrics),('threshold_metrics',threshold_rows),
                          ('calibration_bins',bins),('risk_strata',strata)]:
        pd.DataFrame(rows).to_csv(out/f'{filename}.csv',index=False)
    # Show validation calibration only as an apparent fitting check, not validation performance.
    pc=calibrate(p,model)
    write_json(out/'validation_fit_apparent_only.json',{'n':len(y),'events':int(y.sum()),
                'mean_observed':float(y.mean()),'mean_calibrated':float(pc.mean()),
                'raw_brier':brier_score_loss(y,p),'calibrated_brier':brier_score_loss(y,pc),
                'purpose':'fitting_sanity_check_not_heldout_validation'})
    after={str(p):sha(p) for p in paths}
    assert before==after and sha(out/'CALIBRATOR_AND_THRESHOLDS_FROZEN.json')==lock_hash
    changed=np.abs(e.proba_mean_2-e.proba_mean_1)>1e-12
    qa={'status':'completed_saved_score_diagnostics_not_full_study_lock',
        'source_hashes_unchanged':before==after,'deployment_lock_unchanged_after_evaluation':True,
        'split_patient_overlap_zero':True,'all_internal_labels_and_ids_match_source':True,
        'thresholds_match_article_at_reported_precision':True,
        'external_record_order_hash':identity_hash(e,['subject_id','study_id','visit_id']),
        'external_score_variants_differ_rows':int(changed.sum()),
        'external_score_variants_differ_events':int(e.loc[changed,'death_72h_or_triage'].sum()),
        'external_provenance_caveat':'Historical article scores include source-dependent backfill. Both predefined variants reported, neither used for selection.',
        'outcome_note':'Main death_72h and existing external death_72h_or_triage retained without relabeling.',
        'ci_scope':'Frozen model, calibrator and thresholds; fitting uncertainty excluded.',
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'scipy':scipy.__version__,'sklearn':sklearn.__version__,'statsmodels':sm.__version__},
        'cohorts':qas}
    write_json(out/'QA_SUMMARY.json',qa)
    log('BATCH1 COMPLETE: all frozen-parameter and source-preservation checks passed.')


if __name__=='__main__':
    main()
