"""Fixed-score, exploratory phenotype and subsequent ICU-transfer analyses."""
from pathlib import Path
import argparse, hashlib, json, platform, sys
import numpy as np
import pandas as pd
import scipy
from scipy.special import logit
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]
SEED=20260907
PHENOS=["ami","heart_failure","af_flutter","sepsis","respiratory_failure","aki"]
LABELS=dict(zip(PHENOS,["Acute myocardial infarction","Heart failure","Atrial fibrillation/flutter",
                       "Coded sepsis/septic shock","Respiratory failure","Acute kidney injury"]))
VITALS=["temperature_c","resprate","o2sat","sbp"]
ALLCONT=["age"]+VITALS+["heartrate"]
SCORES=["ecg","fusion","ecg_single"]

def native(x):
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,Path):return str(x)
    return x
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(native(x),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
    return h.hexdigest()
def msg(x):print(x,flush=True)
def gzwrite(df,p):
    df.to_csv(p,index=False,compression={"method":"gzip","mtime":0})
def rcs(x,knots):
    a,b,c=knots
    nonlinear=(np.maximum(x-a,0)**3-
               np.maximum(x-b,0)**3*(c-a)/(c-b)+
               np.maximum(x-c,0)**3*(b-a)/(c-b))/(c-a)**2
    return np.column_stack([x,nonlinear])

def prepare(out):
    main=pd.read_csv(ROOT/"data/ecg_muti.csv",dtype={"file_name":str,"subject_id":str})
    assert main.file_name.is_unique
    parts={};scales={};paths=[ROOT/"data/ecg_muti.csv",PACKAGE/"ANALYSIS_PLAN.md",PACKAGE/"DEVIATIONS.md",Path(__file__)]
    for split,folds in [("train",range(7)),("val",[7]),("test",[8,9])]:
        pf=ROOT/f"data/proba_{split}_72.csv"
        pe=ROOT/f"ecgdea/data/ecg_only_{'val' if split=='val' else split}_predictions.csv"
        ps=ROOT/f"data/proba_{split}_ecg.csv"
        paths.extend([pf,pe,ps])
        f=pd.read_csv(pf,dtype={"file_name":str,"subject_id":str})
        e=pd.read_csv(pe,dtype={"file_name":str,"subject_id":str})
        s=pd.read_csv(ps,dtype={"file_name":str,"subject_id":str})
        for d in [f,e,s]:assert d.file_name.is_unique
        d=f[["file_name","subject_id","death_72h","proba_mean"]].rename(columns={"proba_mean":"fusion"})
        for source,name,ycol in [(e,"ecg","y_true"),(s,"ecg_single","death_72h")]:
            q=source[["file_name","subject_id",ycol,"proba_mean"]].rename(
                columns={"subject_id":"check_id",ycol:"check_y","proba_mean":name})
            d=d.merge(q,on="file_name",how="left",validate="one_to_one")
            assert (d.check_id==d.subject_id).all() and (d.check_y==d.death_72h).all()
            d=d.drop(columns=["check_id","check_y"])
        md=main.drop(columns=["death_72h","subject_id"])
        d=d.merge(md,on="file_name",how="left",validate="one_to_one")
        assert d.general_strat_fold.isin(folds).all()
        for source in [f,e]:
            runcols=[f"proba_run{i}" for i in range(10)]
            assert np.max(np.abs(source[runcols].mean(axis=1)-source.proba_mean))<1e-6
        for name in SCORES:
            x=logit(np.clip(d[name].to_numpy(),1e-8,1-1e-8))
            assert np.isfinite(x).all()
            if split=="train":scales[name]={"mean":x.mean(),"sd":x.std(ddof=0)}
            d[name+"_z"]=(x-scales[name]["mean"])/scales[name]["sd"]
        parts[split]=d
    for x,y in [("train","val"),("train","test"),("val","test")]:
        assert not set(parts[x].subject_id)&set(parts[y].subject_id)
    # User instruction: preserve parent-study missing-data processing.
    # Raw-table missingness audit remains separate and does not change M1/M2.
    for d in parts.values():
        for c in ALLCONT:d[c+"_raw"]=pd.to_numeric(d[c],errors="raise")
    tr=parts["train"];te=parts["test"]
    transform={}
    for c in ALLCONT:
        z=pd.to_numeric(tr[c+"_raw"],errors="coerce").to_numpy(dtype=float)
        obs=z[np.isfinite(z)]
        med=float(np.median(obs));knots=np.quantile(obs,[.1,.5,.9])
        transform[c]={"median":med,"mean":obs.mean(),"sd":obs.std(ddof=0),"knots":knots,
                      "nonlinear":len(np.unique(knots))==3}
    for d in parts.values():
        d["covariates_observed"]=(d.age_m.eq(1)&d.gender_unknown.eq(0)&d.gender_clean_m.eq(1)).astype(int)
        for c in ALLCONT:
            x=pd.to_numeric(d[c+"_raw"],errors="coerce").to_numpy(dtype=float)
            assert np.isfinite(x).all(), "Unexpected missing value: obtain author decision before changing processing."
            d[c+"_missing"]=False
            d[c+"_imp"]=x
    te.insert(0,"row_id",np.arange(len(te)))
    te["risk_stratum"]=np.select([te.fusion<.456303122639656,te.fusion<.6162797212600708],["Low","Intermediate"],default="High")
    te["index_time"]=pd.to_datetime(te.ecg_time,errors="raise")
    first=te.sort_values(["index_time","row_id"]).drop_duplicates("subject_id").row_id
    te["first_parent_encounter"]=te.row_id.isin(first).astype(int)
    # Regenerate frozen comparator scores to verify the saved score-array order.
    cp=ROOT/"project_control/REVISION_ANALYSIS_20260906/batch2/clinical_ecg_xgboost"
    model=joblib.load(cp/"fitted_model.joblib")
    features=model["features"];x=te[features].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float,copy=True)
    for j,c in enumerate(features):
        if c in ["temperature_c","resprate","o2sat","sbp","age","gender_clean"]:
            x[te[c+"_m"].to_numpy()==0,j]=np.nan
    x[~np.isfinite(x)]=np.nan
    xx=np.column_stack([np.where(np.isfinite(x),x,model["medians"]),np.isfinite(x).astype(float)])
    if model["scaler"] is not None:xx=model["scaler"].transform(xx)
    te["clinical_ecg"]=model["estimator"].predict_proba(xx)[:,1].astype(float)
    saved=np.load(cp/"internal_test/evaluation_scores.npz",allow_pickle=False)
    assert np.array_equal(saved["y"],te.death_72h)
    assert np.max(np.abs(saved["raw"]-te.clinical_ecg))<1e-12
    paths.extend([cp/"fitted_model.joblib",cp/"internal_test/evaluation_scores.npz"])
    (PACKAGE/"derived").mkdir(exist_ok=True)
    gzwrite(te,PACKAGE/"derived/test_base.csv.gz")
    dump(out/"qa/input_qa.json",{"n":len(te),"patients":te.subject_id.nunique(),"mortality_events":int(te.death_72h.sum()),
        "split_sizes":{k:len(v) for k,v in parts.items()},"patient_overlap_zero":True,"scores_aligned":True,
        "score_scales":scales,"covariate_transform":transform,"missing_demographics_n":int((te.covariates_observed==0).sum()),
        "comparator_regeneration_matches_saved_array":True,"source_hashes":{str(p):sha(p) for p in paths}})
    return te,transform

def design(d,score,stage,transform):
    cols={"intercept":np.ones(len(d)),"score_z":d[score+"_z"].to_numpy(),"male":d.gender_clean.to_numpy()}
    vars=["age"]+(VITALS if stage in ["M2","M2_HR"] else [])+(["heartrate"] if stage=="M2_HR" else [])
    for c in vars:
        t=transform[c];x=d[c+"_imp"].to_numpy()
        cols[c]=(x-t["mean"])/t["sd"]
        if t["nonlinear"]:
            b=rcs(x,t["knots"])[:,1]/t["sd"]
            cols[c+"_nonlinear"]=b
        miss=d[c+"_missing"].to_numpy(dtype=float)
        if miss.min()!=miss.max():cols[c+"_missing"]=miss
    frame=pd.DataFrame(cols,index=d.index)
    assert np.isfinite(frame.to_numpy()).all()
    assert np.linalg.matrix_rank(frame.to_numpy())==frame.shape[1]
    return frame

def fit_assoc(d,ycol,score,stage,transform,domain,subset):
    v=d.loc[d.covariates_observed.eq(1)&d[ycol].notna()].copy()
    y=v[ycol].astype(int).to_numpy()
    base={"domain":domain,"endpoint":ycol,"score":score,"adjustment":stage,"subset":subset,
          "n":len(v),"patients":v.subject_id.nunique(),"events":int(y.sum())}
    if min(y.sum(),len(y)-y.sum())<30:return {**base,"status":"insufficient_events"}
    x=design(v,score,stage,transform)
    fit=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10,
        cov_type="cluster",cov_kwds={"groups":v.subject_id.to_numpy(),"use_correction":True})
    assert fit.converged
    b=float(fit.params["score_z"]);se=float(fit.bse["score_z"])
    assert np.isfinite([b,se]).all() and se>0
    p=float(2*norm.sf(abs(b/se)))
    return {**base,"status":"ok","beta":b,"se":se,"or":np.exp(b),"ci_low":np.exp(b-1.959963984540054*se),
            "ci_high":np.exp(b+1.959963984540054*se),"p":p,"q":np.nan,
            "design_columns":";".join(x.columns),"converged":True}

def weighted_metrics(y,orders,ties,weights):
    w=weights[orders];ys=y[orders]
    tp=np.cumsum(w*ys)[ties];fp=np.cumsum(w*(1-ys))[ties]
    pos,neg=tp[-1],fp[-1]
    if pos<=0 or neg<=0:return [np.nan,np.nan]
    roc=np.trapezoid(np.r_[0,tp/pos],np.r_[0,fp/neg])
    ap=np.sum(np.diff(np.r_[0,tp/pos])*np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0))
    return [roc,ap]

def discrimination(d,ycol,nboot,seed,subset):
    y=d[ycol].to_numpy(dtype=int);groups,uniq=pd.factorize(d.subject_id,sort=True)
    rng=np.random.default_rng(seed);models=["ecg","fusion","clinical_ecg"];spec={};draws=[]
    for m in models:
        score=d[m].to_numpy();order=np.argsort(-score,kind="mergesort")
        tie=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(score)-1]
        spec[m]=(order,tie)
        a=weighted_metrics(y,order,tie,np.ones(len(d)))
        assert np.allclose(a,[roc_auc_score(y,score),average_precision_score(y,score)],atol=1e-12)
    for b in range(nboot):
        count=np.bincount(rng.integers(len(uniq),size=len(uniq)),minlength=len(uniq))
        draws.append([weighted_metrics(y,*spec[m],count[groups].astype(float)) for m in models])
    draws=np.asarray(draws);rows=[]
    for i,m in enumerate(models):
        pt=weighted_metrics(y,*spec[m],np.ones(len(d)))
        for j,metric in enumerate(["auroc","average_precision"]):
            ci=np.nanquantile(draws[:,i,j],[.025,.975])
            rows.append({"subset":subset,"endpoint":ycol,"model":m,"metric":metric,"estimate":pt[j],
                         "ci_low":ci[0],"ci_high":ci[1],"n":len(d),"events":int(y.sum()),
                         "prevalence":y.mean(),"n_boot":nboot,"seed":seed,
                         "n_boot_valid":int(np.isfinite(draws[:,i,j]).sum())})
    # Paired descriptive differences from the same patient resamples.
    for a,b in [(1,2),(0,2)]:
        for j,metric in enumerate(["auroc","average_precision"]):
            diff=draws[:,a,j]-draws[:,b,j];ci=np.nanquantile(diff,[.025,.975])
            pts=[weighted_metrics(y,*spec[m],np.ones(len(d)))[j] for m in models]
            rows.append({"subset":subset,"endpoint":ycol,"model":models[a]+"_minus_"+models[b],
                         "metric":metric,"estimate":pts[a]-pts[b],"ci_low":ci[0],"ci_high":ci[1],
                         "n":len(d),"events":int(y.sum()),"prevalence":y.mean(),"n_boot":nboot,
                         "seed":seed,"n_boot_valid":int(np.isfinite(diff).sum())})
    return rows

def strata(d,nboot,seed,subset):
    groups,uniq=pd.factorize(d.subject_id,sort=True);rng=np.random.default_rng(seed)
    obs=[]
    for s in ["Low","Intermediate","High"]:
        keep=d.risk_stratum.eq(s).to_numpy(dtype=float)
        obs.extend([keep,keep*d.icu24,keep*d.icu72])
    mat=np.column_stack(obs);sums=np.zeros((len(uniq),9));np.add.at(sums,groups,mat)
    total=sums.sum(axis=0);boot=[]
    for b in range(nboot):
        c=np.bincount(rng.integers(len(uniq),size=len(uniq)),minlength=len(uniq));a=c@sums
        boot.append([a[3*i+j]/a[3*i] if a[3*i]>0 else np.nan for i in range(3) for j in [1,2]])
    boot=np.asarray(boot);rows=[]
    for i,s in enumerate(["Low","Intermediate","High"]):
        for j,endpoint in enumerate(["icu24","icu72"]):
            n,e=total[3*i],total[3*i+j+1];ci=np.nanquantile(boot[:,2*i+j],[.025,.975])
            rows.append({"subset":subset,"stratum":s,"endpoint":endpoint,"n":int(n),"events":int(e),
                         "proportion":e/n,"ci_low":ci[0],"ci_high":ci[1],"n_boot":nboot,
                         "seed":seed,"n_boot_valid":int(np.isfinite(boot[:,2*i+j]).sum())})
    return rows

def run(out,nboot):
    for k in ["qa","results"]: (out/k).mkdir(parents=True,exist_ok=True)
    base,tr=prepare(out)
    # Endpoint derivations are produced independently, retaining every parent row.
    ph=pd.read_csv(PACKAGE/"derived/phenotypes.csv.gz",dtype={"file_name":str,"subject_id":str})
    ic=pd.read_csv(PACKAGE/"derived/icu.csv.gz",dtype={"file_name":str,"subject_id":str})
    for d in [ph,ic]:
        assert np.array_equal(d.row_id,base.row_id)
        assert np.array_equal(d.file_name,base.file_name)
        assert np.array_equal(d.subject_id,base.subject_id)
    # Consume endpoint columns only; preserve the verified baseline metadata.
    p=base.merge(ph[[c for c in ph if c not in base.columns or c=="row_id"]],on="row_id",validate="one_to_one")
    q=base.merge(ic[[c for c in ic if c not in base.columns or c=="row_id"]],on="row_id",validate="one_to_one")
    p=p.loc[p.diagnosis_observed.eq(1)].copy()
    q=q.loc[q.eligible.eq(1)].copy()
    assert p[PHENOS].notna().all().all()
    assert q[["icu24","icu72"]].notna().all().all() and (q.icu24<=q.icu72).all()
    rows=[]
    for endpoint in PHENOS:
        for score,stage,subset in [("ecg","M1","all"),("ecg","M2","all"),("fusion","M2","all"),
                                  ("ecg_single","M2","single_run"),("ecg","M2","first_encounter")]:
            d=p.loc[p.first_parent_encounter.eq(1)] if subset=="first_encounter" else p
            rows.append(fit_assoc(d,endpoint,score,stage,tr,"phenotype",subset))
        msg("Phenotype models complete: "+endpoint)
    for endpoint in ["icu24","icu72"]:
        for score,stage,subset in [("ecg","M1","all"),("ecg","M2","all"),("fusion","M2","all"),
                                  ("ecg_single","M2","single_run"),("ecg","M2","first_encounter"),
                                  ("ecg","M2","landmark60"),
                                  ("fusion","M2","landmark60")]:
            d=q.copy()
            if subset=="first_encounter":d=d.loc[d.first_parent_encounter.eq(1)]
            if subset=="landmark60":
                d=d.loc[d.landmark60_eligible.eq(1)].copy()
                d[endpoint]=d["landmark60_"+endpoint]
            rows.append(fit_assoc(d,endpoint,score,stage,tr,"icu",subset))
        msg("ICU models complete: "+endpoint)
    assoc=pd.DataFrame(rows)
    for domain in ["phenotype","icu"]:
        idx=assoc.index[(assoc.domain==domain)&(assoc.score=="ecg")&(assoc.adjustment=="M2")&(assoc.subset=="all")&(assoc.status=="ok")]
        assoc.loc[idx,"q"]=multipletests(assoc.loc[idx,"p"],method="fdr_bh")[1]
    assoc.to_csv(out/"results/associations.csv",index=False)
    strat=strata(q,nboot,SEED,"all")
    lm=q.loc[q.landmark60_eligible.eq(1)].copy()
    for ep in ["icu24","icu72"]:lm[ep]=lm["landmark60_"+ep]
    strat+=strata(lm,nboot,SEED+1,"landmark60")
    pd.DataFrame(strat).to_csv(out/"results/icu_strata.csv",index=False)
    discs=[]
    for i,ep in enumerate(["icu24","icu72"]):
        msg("Patient bootstrap discrimination: "+ep)
        discs.extend(discrimination(q,ep,nboot,SEED+10+i,"all"))
    pd.DataFrame(discs).to_csv(out/"results/icu_discrimination.csv",index=False)
    counts=[]
    for ep in ["icu24","icu72"]:
        col="disposition"+ep.replace("icu","")
        for status,n in q[col].value_counts(dropna=False).items():
            counts.append({"endpoint":ep,"disposition":status,"n":int(n),"denominator":len(q)})
    pd.DataFrame(counts).to_csv(out/"results/icu_dispositions.csv",index=False)
    # Local analytical rows support reproducibility and independent checking.
    gzwrite(p,out/"qa/phenotype_analytic_local.csv.gz")
    gzwrite(q,out/"qa/icu_analytic_local.csv.gz")
    primary=assoc[(assoc.score=="ecg")&(assoc.adjustment=="M2")&(assoc.subset=="all")]
    assert primary.status.eq("ok").all()
    dump(out/"qa/statistical_qa.json",{"status":"PASS","n_parent":len(base),"phenotype_n":len(p),"icu_n":len(q),
       "all_primary_models_converged":True,"patient_clustering":True,"first_encounter_from_parent":True,
       "score_scaling_training_only":True,"no_model_retraining":True,"bootstrap_replicates":nboot,
       "bootstrap_seeds":{"strata_all":SEED,"strata_landmark60":SEED+1,
                          "discrimination_icu24":SEED+10,"discrimination_icu72":SEED+11},
       "all_bootstrap_replicates_valid":all(r["n_boot_valid"]==nboot for r in strat+discs),
       "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
       "scipy":scipy.__version__,"statsmodels":sm.__version__,
       "endpoint_source_hashes":{str(x):sha(x) for x in [PACKAGE/"derived/phenotypes.csv.gz",PACKAGE/"derived/icu.csv.gz"]},
       "outputs":{str(x.relative_to(out)):sha(x) for x in sorted((out/"results").glob("*.csv"))}})
    msg("ANALYSES COMPLETE")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=PACKAGE)
    ap.add_argument("--n-boot",type=int,default=2000);ap.add_argument("--prepare-only",action="store_true")
    args=ap.parse_args()
    if args.prepare_only:prepare(args.out)
    else:run(args.out,args.n_boot)
