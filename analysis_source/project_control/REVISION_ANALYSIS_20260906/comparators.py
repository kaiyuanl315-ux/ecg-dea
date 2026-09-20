"""Train mortality comparators; tuning only inside training, calibration on VAL."""
from __future__ import annotations
import argparse
import itertools
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost
from analysis import (ROOT, PACKAGE, SEED, N_BOOT, load_internal_splits, fit_calibrator,
                      select_thresholds, calibrate, evaluate, write_json, sha, identity_hash,log)

CLINICAL=['temperature_c','resprate','o2sat','sbp','age','gender_clean']
ECG=['pr_interval','qrs_duration','qt_interval','qrs_axis']
GRIDS=[{'max_depth':d,'n_estimators':n} for d,n in itertools.product([2,3],[200,500])]


def numeric_inputs(d,features):
    """Undo earlier imputation by the saved observation masks before CV."""
    x=d[features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float,copy=True)
    for j,f in enumerate(features):
        if f in CLINICAL:
            mask=pd.to_numeric(d[f+'_m'],errors='raise').to_numpy()
            assert np.isin(mask,[0,1]).all()
            x[mask==0,j]=np.nan
    x[~np.isfinite(x)]=np.nan
    return x


def fit_preprocess(x):
    med=np.nanmedian(x,axis=0)
    assert np.isfinite(med).all()
    return med


def transform(x,med):
    observed=np.isfinite(x)
    return np.column_stack([np.where(observed,x,med),observed.astype(float)])


def xgb(config,seed):
    return XGBClassifier(**config, learning_rate=.05,min_child_weight=10,reg_lambda=10,
         subsample=.8,colsample_bytree=1,scale_pos_weight=1,tree_method='hist',
         objective='binary:logistic',eval_metric='logloss',random_state=seed,n_jobs=4)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=PACKAGE/'batch2')
    parser.add_argument('--n-boot',type=int,default=N_BOOT)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    manifest_paths=[ROOT/'data/ecg_muti.csv',ROOT/'data/zs/中山结果_v2.csv',PACKAGE/'ANALYSIS_PLAN.md',
                    PACKAGE/'analysis.py',Path(__file__)]+[ROOT/f'data/proba_{s}_72.csv' for s in ['train','val','test']]
    before={str(p):sha(p) for p in manifest_paths};write_json(out/'source_manifest.json',before)
    parts=load_internal_splits()
    features=CLINICAL+ECG;mask_cols=[f+'_m' for f in CLINICAL]
    main_data=pd.read_csv(ROOT/'data/ecg_muti.csv',usecols=['file_name']+features+mask_cols,dtype={'file_name':str})
    frames={s:d[['file_name','subject_id','death_72h']].merge(main_data,on='file_name',how='left',validate='one_to_one') for s,d in parts.items()}
    ytrain=frames['train'].death_72h.to_numpy();groups=frames['train'].subject_id.to_numpy()
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    splits=list(cv.split(np.zeros(len(ytrain)),ytrain,groups))
    cv_description=[]
    for fold,(itr,iva) in enumerate(splits):
        assert not set(groups[itr]) & set(groups[iva])
        cv_description.append({'fold':fold,'train_n':len(itr),'train_events':int(ytrain[itr].sum()),
                               'holdout_n':len(iva),'holdout_events':int(ytrain[iva].sum()),'patient_overlap':0})
    write_json(out/'training_cv_splits_summary.json',cv_description)
    models={};cv_rows=[];missing_rows=[]
    specs=[('clinical_logistic',CLINICAL,'logistic'),('clinical_xgboost',CLINICAL,'xgb'),
           ('clinical_ecg_xgboost',features,'xgb')]
    for model_name,fs,kind in specs:
        log(f'Fitting {model_name}: {len(fs)} variables + fixed observation indicators')
        x=numeric_inputs(frames['train'],fs)
        for split,d in frames.items():
            z=numeric_inputs(d,fs)
            for j,f in enumerate(fs):
                missing_rows.append({'model':model_name,'cohort':split,'feature':f,'n':len(z),
                                     'missing_n':int(np.isnan(z[:,j]).sum()),'missing_fraction':float(np.isnan(z[:,j]).mean())})
        if kind=='xgb':
            config_results=[]
            for ci,config in enumerate(GRIDS):
                aps=[]
                for fold,(itr,iva) in enumerate(splits):
                    med=fit_preprocess(x[itr]);est=xgb(config,SEED+fold)
                    est.fit(transform(x[itr],med),ytrain[itr])
                    pr=est.predict_proba(transform(x[iva],med))[:,1]
                    ap=float(average_precision_score(ytrain[iva],pr));aps.append(ap)
                    cv_rows.append({'model':model_name,'config_id':ci,**config,'fold':fold,
                        'average_precision':ap,'auroc':roc_auc_score(ytrain[iva],pr),
                        'n':len(iva),'events':int(ytrain[iva].sum())})
                    log(f'  config {ci+1}/4 fold {fold+1}/3: AP={ap:.6f}')
                config_results.append({'config_id':ci,**config,'mean_cv_ap':float(np.mean(aps))})
            chosen=sorted(config_results,key=lambda d:(-d['mean_cv_ap'],d['max_depth'],d['n_estimators']))[0]
            selected={k:chosen[k] for k in ['max_depth','n_estimators']}
            estimator=xgb(selected,SEED)
            scaler=None
        else:
            config_results=[];selected={'penalty':None,'solver':'lbfgs','max_iter':3000,'tol':1e-8}
            estimator=LogisticRegression(**selected)
            scaler=StandardScaler()
        med=fit_preprocess(x);xt=transform(x,med)
        if scaler is not None:
            xt=scaler.fit_transform(xt)
        estimator.fit(xt,ytrain)
        if kind=='logistic':
            assert max(estimator.n_iter_) < selected['max_iter']
        def predict(d):
            xx=transform(numeric_inputs(d,fs),med)
            if scaler is not None:
                xx=scaler.transform(xx)
            return estimator.predict_proba(xx)[:,1].astype(float)
        v=frames['val'];pval=predict(v);yv=v.death_72h.to_numpy();vid=v.subject_id.to_numpy()
        cal=fit_calibrator(yv,pval,vid);thresholds=select_thresholds(yv,pval)
        edges=np.unique(np.r_[0,np.quantile(pval,np.arange(.1,1,.1)),1])
        mdir=out/model_name;mdir.mkdir(exist_ok=True)
        joblib.dump({'estimator':estimator,'scaler':scaler,'medians':med,'features':fs},mdir/'fitted_model.joblib')
        if kind=='xgb':
            estimator.save_model(mdir/'fitted_xgboost.json')
        lock={'model':model_name,'created_utc':datetime.now(timezone.utc).isoformat(),
              'features':fs,'missingness':'restore missing using original observation masks; training-only median; indicators appended',
              'training_medians':dict(zip(fs,med)),'selected_parameters':selected,'cv_grid_results':config_results,
              'selection_metric':'training-only 3-fold patient-grouped CV average precision' if kind=='xgb' else 'unpenalized logistic baseline',
              'calibrator':cal,'thresholds_raw':thresholds,
              'thresholds_calibrated':{k:float(calibrate([t],cal)[0]) for k,t in thresholds.items()},
              'validation_bin_edges':edges,'training_n':len(ytrain),'training_events':int(ytrain.sum()),
              'training_record_order_hash':identity_hash(frames['train'],['file_name','subject_id']),
              'validation_record_order_hash':identity_hash(v,['file_name','subject_id']),
              'test_or_external_used_in_selection':False,'target':'actual death label, not fusion score',
              'model_file_sha256':sha(mdir/'fitted_model.joblib'),'xgboost_version':xgboost.__version__}
        write_json(mdir/'MODEL_CALIBRATION_THRESHOLDS_FROZEN.json',lock)
        models[model_name]={'estimator':estimator,'scaler':scaler,'medians':med,'features':fs,
                           'calibrator':cal,'thresholds':thresholds,'edges':edges}
    pd.DataFrame(cv_rows).to_csv(out/'training_only_hyperparameter_search.csv',index=False)
    locks={m:sha(out/m/'MODEL_CALIBRATION_THRESHOLDS_FROZEN.json') for m in models}
    log('ALL comparator models, preprocessing, calibration and thresholds frozen. Starting tests.')
    # This is the first access to external feature values in the model pipeline.
    external=pd.read_csv(ROOT/'data/zs/中山结果_v2.csv',usecols=['subject_id','visit_id','death_72h_or_triage']+features+mask_cols,
                         dtype={'subject_id':str,'visit_id':str})
    allmetrics=[];allthresholds=[];allbins=[];allstrata=[];allqa=[]
    for mi,(model_name,model) in enumerate(models.items()):
        for name,d,ycol in [('internal_test',frames['test'],'death_72h'),('external_same_records',external,'death_72h_or_triage')]:
            xx=numeric_inputs(d,model['features'])
            if name.startswith('external'):
                for j,f in enumerate(model['features']):
                    missing_rows.append({'model':model_name,'cohort':'external','feature':f,'n':len(xx),
                                         'missing_n':int(np.isnan(xx[:,j]).sum()),'missing_fraction':float(np.isnan(xx[:,j]).mean())})
            xx=transform(xx,model['medians'])
            if model['scaler'] is not None:
                xx=model['scaler'].transform(xx)
            p=model['estimator'].predict_proba(xx)[:,1].astype(float)
            r=evaluate(d[ycol].to_numpy(),p,d.subject_id.to_numpy(),name,model['calibrator'],model['thresholds'],model['edges'],
                       out/model_name/name,args.n_boot,SEED+(0 if name=='internal_test' else 1))
            for bucket,new in zip([allmetrics,allthresholds,allbins,allstrata,allqa],r):
                for row in (new if isinstance(new,list) else [new]):
                    row['model']=model_name;bucket.append(row)
    for filename,rows in [('calibration_metrics',allmetrics),('threshold_metrics',allthresholds),
                          ('calibration_bins',allbins),('risk_strata',allstrata),('input_missingness',missing_rows)]:
        pd.DataFrame(rows).to_csv(out/f'{filename}.csv',index=False)
    after={str(p):sha(p) for p in manifest_paths}
    assert before==after
    assert all(sha(out/m/'MODEL_CALIBRATION_THRESHOLDS_FROZEN.json')==s for m,s in locks.items())
    write_json(out/'QA_SUMMARY.json',{'status':'completed_secondary_comparators',
        'source_hashes_unchanged':True,'all_models_locked_before_test_evaluation':True,
        'training_cv_patient_overlap_zero':True,'all_parameter_searches_inside_training_only':True,
        'no_external_optimization':True,'all_estimators_predict_actual_mortality':True,
        'n_cv_configurations_per_nonlinear_model':4,'n_grouped_cv_folds':3,
        'ecg_features_excluded_pending_mapping':['ventricular_rate/heartrate'],
        'external_endpoint_and_membership_unchanged':True,'cohorts':allqa})
    log('BATCH2 COMPLETE: no external or internal-test tuning performed.')


if __name__=='__main__':
    main()
