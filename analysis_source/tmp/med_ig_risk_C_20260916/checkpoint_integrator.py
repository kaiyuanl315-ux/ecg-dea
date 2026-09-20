"""Progress/checkpoint instrumentation of the unchanged verified IG arithmetic.

The mathematical lines and case/node ordering are copied from ig_analysis.ig_one.
Only logging and atomic host-float64 partial-integral snapshots are added.
"""
from run_ig import q, BASE, OLD
import numpy as np
import json, time, os

CONTEXT={}
original_ig_one=q.ig_one
original_load_model=q.load_model
original_integrate=q.integrate

def load_model(k,device):
    model,meta=original_load_model(k,device)
    CONTEXT.update({'run':k,'checkpoint_sha256':meta['sha256']})
    return model,meta

def ig_one(model,x,c,b,device,steps,batch=64):
    if 'tag' not in CONTEXT:return original_ig_one(model,x,c,b,device,steps,batch)
    nodes,w=np.polynomial.legendre.leggauss(steps);nodes=(nodes+1)/2;w=w/2
    n=len(x);diff=x-b;integral=np.zeros(x.shape,dtype=np.float64)
    folder=BASE/'partial_integrals'/CONTEXT['tag'];folder.mkdir(parents=True,exist_ok=True)
    state=folder/f"member_{CONTEXT['run']:02d}.npz"
    meta={'run':CONTEXT['run'],'checkpoint_sha256':CONTEXT['checkpoint_sha256'],
          'tag':CONTEXT['tag'],'device':device,'steps':steps,'batch':batch,
          'x_sha256':q.array_digest(x),'c_sha256':q.array_digest(c),'b_sha256':q.array_digest(b),
          'integrator_sha256':q.digest(OLD/'ig_analysis.py'),
          'instrumentation_sha256':q.digest(BASE/'checkpoint_integrator.py')}
    resume=0
    if state.exists():
        saved=np.load(state)
        assert json.loads(str(saved['metadata']))==meta
        integral=saved['integral'];resume=int(saved['next_start'])
        assert integral.dtype==np.float64 and integral.shape==x.shape and np.isfinite(integral).all()
        assert 0<=resume<=n*steps and (resume%batch==0 or resume==n*steps)
    start_time=time.monotonic();last_log=start_time
    q.emit('ig_member_begin',tag=CONTEXT['tag'],run=CONTEXT['run'],completed_path_points=resume,total_path_points=n*steps)
    for start in range(resume,n*steps,batch):
        ix=np.arange(start,min(start+batch,n*steps));case=ix//steps;node=ix%steps
        xx=q.torch.tensor(b[case]+diff[case]*nodes[node,None,None].astype(np.float32),device=device,requires_grad=True)
        cc=q.torch.tensor(c[case],device=device)
        p=q.torch.sigmoid(model(xx,cc)).sum();g=q.torch.autograd.grad(p,xx,create_graph=False)[0].detach().cpu().numpy()
        assert np.isfinite(g).all(),'Nonfinite gradient'
        np.add.at(integral,case,g.astype(np.float64)*w[node,None,None])
        del xx,cc,p,g
        next_start=min(start+batch,n*steps)
        if next_start%(128*batch)==0 or next_start==n*steps:
            temporary=state.with_suffix('.partial')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle,integral=integral,next_start=next_start,metadata=json.dumps(meta))
            os.replace(temporary,state)
        now=time.monotonic()
        if now-last_log>=30 or next_start==n*steps:
            q.emit('ig_batch_progress',tag=CONTEXT['tag'],run=CONTEXT['run'],completed_path_points=next_start,
                   total_path_points=n*steps,percent=100*next_start/(n*steps),seconds_since_resume=now-start_time)
            last_log=now
    return (diff*integral).astype(np.float32)

def integrate(device,steps,kind,subset,tag,batch):
    CONTEXT['tag']=tag
    try:return original_integrate(device,steps,kind,subset,tag,batch)
    finally:CONTEXT.pop('tag',None)

def install():
    q.load_model=load_model;q.ig_one=ig_one;q.integrate=integrate
