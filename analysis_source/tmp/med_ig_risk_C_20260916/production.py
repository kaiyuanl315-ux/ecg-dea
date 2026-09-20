"""High-risk C attribution; exact input-validated reuse and frozen numerical checks."""
from run_ig import q, BASE, OLD
import numpy as np
import json, time, gc, sys, platform

def save(name,obj):q.savejson(name,obj)

def main():
    a=np.load(BASE/'inputs.npz');x=a['ecg'];c=a['tab']
    assert len(x)==1536
    pred=[];info=[]
    for k in range(10):
        model,meta=q.load_model(k,'mps')
        pk=q.predict(model,x,c,'mps');pred.append(pk)
        info.append({**meta,**q.compare(pk,a['expected'][:,k])})
        q.emit('production_mps_prediction',**info[-1])
        del model;gc.collect();q.torch.mps.empty_cache()
    pred=np.stack(pred,axis=1);ens=pred.mean(1)
    gate={'members':info,'ensemble':q.compare(ens,a['expected_mean']),
          'threshold_changes':{str(t):int(((ens>=t)!=(a['expected_mean']>=t)).sum()) for t in [q.LOW,q.HIGH]},
          'historical_member_gate_pass':all(z['outside_prior_tolerance']==0 for z in info),
          'runtime':{'python':sys.version,'torch':q.torch.__version__,'numpy':np.__version__},
          'saved_scores_and_grouping_replaced':False}
    np.save(BASE/'production_predictions_mps.npy',pred);save('production_prediction_gate.json',gate)
    # Any new ensemble discrepancy or risk reassignment requires explicit review.
    assert gate['ensemble']['outside_prior_tolerance']==0, gate
    assert not any(gate['threshold_changes'].values()), gate
    reuse=json.loads((BASE/'overlap_reuse_private.json').read_text())
    mapping=reuse['high_risk_mapping'];old_a=np.load(OLD/'inputs.npz')
    old_sha=q.digest(OLD/'inputs.npz');source_sha=q.digest(q.SOURCE)
    for r in mapping:
        ni,oi=r['new_sample_row_0based'],r['old_sample_row_0based']
        assert np.array_equal(x[ni],old_a['ecg'][oi]) and np.array_equal(c[ni],old_a['tab'][oi])
    old_members=[]
    for k in range(10):
        path=OLD/'runs/zero_128'/f'member_{k:02d}.npz'
        z=np.load(path)
        assert int(z['steps'])==128 and np.array_equal(z['indices'],np.arange(191))
        assert json.loads(str(z['provenance']))=={'baseline':'zero','device':'mps','input_sha256':old_sha,'source_sha256':source_sha,'checkpoint_sha256':info[k]['sha256']}
        old_members.append({'run':k,'sha256':q.digest(path),'checkpoint_sha256':info[k]['sha256']})
    high=np.arange(512);reused={r['new_sample_row_0based'] for r in mapping}
    new=[int(i) for i in high if i not in reused]
    scores=a['expected_mean'][:512]
    sens=[int(np.argmin(np.abs(scores-np.quantile(scores,v)))) for v in [.25,.5,.75,.9]]
    assert len(set(sens))==4
    frozen={'new_indices':new,'reused_mapping':mapping,'old_members':old_members,'sensitivity_indices':sens,'cpu_indices':sens[:2],
            'integrator_sha256':q.digest(OLD/'ig_analysis.py'),'runner_sha256':q.digest(BASE/'run_ig.py'),
            'production_sha256':q.digest(BASE/'production.py'),'input_sha256':q.digest(BASE/'inputs.npz'),
            'target':'mean of ten raw sigmoid FP32 member outputs; observed clinical tensor fixed; zero ECG reference',
            'decision':'Proceed for retained-checkpoint FP32 function; historical member discrepancies retained, all new ensemble and risk-membership checks passed.'}
    save('production_plan_private.json',frozen)
    q.unit_check()
    q.integrate('mps',128,'zero','indices:'+','.join(map(str,new)),'new_high_zero_128',64)
    z=np.load(BASE/'runs/new_high_zero_128/ensemble.npz')
    ig=np.zeros((512,12,1000),np.float32);px=np.zeros(512);pb=np.zeros(512);steps=np.full(512,128)
    ig[new]=z['ig'];px[new]=z['prediction'];pb[new]=z['baseline_prediction']
    for r in mapping:
        ni,oi=r['new_sample_row_0based'],r['old_sample_row_0based']
        member_ig=[];preds=[];bases=[]
        for k in range(10):
            zz=np.load(OLD/'runs/zero_128'/f'member_{k:02d}.npz')
            member_ig.append(zz['ig'][oi].astype(np.float64));preds.append(zz['prediction'][oi]);bases.append(zz['baseline_prediction'][oi])
        ig[ni]=np.mean(member_ig,axis=0).astype(np.float32);px[ni]=np.mean(preds);pb[ni]=np.mean(bases)
    refinements=[]
    for nsteps in [256,512,1024,2048]:
        residual=ig.sum((1,2),dtype=np.float64)-(px-pb);tolerance=.0005+.02*np.abs(px-pb)
        failed=np.flatnonzero(np.abs(residual)>tolerance)
        if not len(failed):break
        tag=f'completeness_refinement_{nsteps}'
        q.integrate('mps',nsteps,'zero','indices:'+','.join(map(str,failed)),tag,64)
        zz=np.load(BASE/'runs'/tag/'ensemble.npz')
        ig[failed]=zz['ig'];px[failed]=zz['prediction'];pb[failed]=zz['baseline_prediction'];steps[failed]=nsteps
        refinements.append({'tag':tag,'indices':failed.tolist()})
    residual=ig.sum((1,2),dtype=np.float64)-(px-pb);tolerance=.0005+.02*np.abs(px-pb)
    np.savez_compressed(BASE/'high_risk_attributions.npz',ig=ig,prediction=px,baseline_prediction=pb,residual=residual,tolerance=tolerance,steps=steps,indices=high)
    report={'n':512,'n_new':len(new),'n_reused':len(reused),'n_failed':int((np.abs(residual)>tolerance).sum()),
            'max_absolute_residual':float(np.max(np.abs(residual))),'median_absolute_residual':float(np.median(np.abs(residual))),
            'refinements':refinements,'saved_float32_rechecked':True}
    save('primary_IG_QA.json',report);assert report['n_failed']==0
    q.emit('primary_high_complete',**report)
    q.integrate('mps',256,'zero','indices:'+','.join(map(str,sens)),'new_sensitivity_256',64)
    q.integrate('cpu',128,'zero','indices:'+','.join(map(str,sens[:2])),'new_cpu_128',64)
    checks={}
    from scipy.stats import spearmanr
    for name,inds in [('new_sensitivity_256',sens),('new_cpu_128',sens[:2])]:
        zz=np.load(BASE/'runs'/name/'ensemble.npz');other=zz['ig'].astype(float);ref=ig[inds].astype(float)
        residual2=other.sum((1,2))-(zz['prediction']-zz['baseline_prediction'])
        tol=.0005+.02*np.abs(zz['prediction']-zz['baseline_prediction'])
        shares=lambda v:np.abs(v).sum(2)/np.abs(v).sum((1,2))[:,None]*100
        s1,s2=shares(ref),shares(other)
        checks[name]={'n':len(inds),'completeness_failed':int((np.abs(residual2)>tol).sum()),
                     'signed_relative_L1':(np.abs(other-ref).sum((1,2))/np.abs(ref).sum((1,2))).tolist(),
                     'max_lead_share_difference_pp':float(np.max(np.abs(s1-s2))),
                     'lead_share_spearman':[float(spearmanr(aa,bb).statistic) for aa,bb in zip(s1,s2)]}
        assert checks[name]['completeness_failed']==0
    save('sensitivity_IG_QA.json',checks)
    q.emit('production_finished',checks=checks)

if __name__=='__main__':main()
