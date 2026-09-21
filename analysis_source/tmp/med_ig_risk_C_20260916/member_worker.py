"""One retained model per fresh process, with identical IG math and checkpoints."""
from run_ig import q, BASE, OLD
import checkpoint_integrator as traced
import argparse,json,time,os
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=int,required=True)
    p.add_argument('--device',required=True);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--indices',required=True)
    p.add_argument('--batch',type=int,default=64);args=p.parse_args()
    a=np.load(BASE/'inputs.npz');indices=np.array([int(v) for v in args.indices.split(',')])
    x=a['ecg'][indices];c=a['tab'][indices];b=np.zeros_like(x)
    gate=json.loads((BASE/'production_prediction_gate.json').read_text())
    assert q.torch.__version__==gate['runtime']['torch'] and np.__version__==gate['runtime']['numpy']
    reference=np.load(BASE/'runs/new_high_zero_128/member_00.npz')
    reference_provenance=json.loads(str(reference['provenance']))
    assert q.digest(q.SOURCE)==reference_provenance['source_sha256']
    assert q.digest(BASE/'inputs.npz')==reference_provenance['input_sha256']
    assert q.digest(OLD/'ig_analysis.py')==json.loads((BASE/'production_plan_private.json').read_text())['integrator_sha256']
    folder=BASE/'runs'/args.tag;folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'member_{args.run:02d}.npz'
    provenance={'baseline':'zero','device':args.device,'input_sha256':q.digest(BASE/'inputs.npz'),
                'source_sha256':q.digest(q.SOURCE),'checkpoint_sha256':q.digest(q.WEIGHTS/f'run_{args.run:02d}_best_by_val_auprc.ckpt')}
    if dest.exists():
        saved=np.load(dest)
        assert np.array_equal(saved['indices'],indices) and int(saved['steps'])==args.steps
        assert json.loads(str(saved['provenance']))==provenance
        q.emit('member_worker_existing',tag=args.tag,run=args.run)
    else:
        start=time.monotonic();model,meta=traced.load_model(args.run,args.device)
        traced.CONTEXT['tag']=args.tag
        attr=traced.ig_one(model,x,c,b,args.device,args.steps,args.batch)
        px=q.predict(model,x,c,args.device);pb=q.predict(model,b,c,args.device)
        assert np.isfinite(attr).all() and np.isfinite(px).all() and np.isfinite(pb).all()
        temporary=dest.with_suffix('.partial')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle,ig=attr,prediction=px,baseline_prediction=pb,indices=indices,steps=args.steps,provenance=json.dumps(provenance))
        os.replace(temporary,dest)
        report={**meta,'tag':args.tag,'device':args.device,'steps':args.steps,'batch':args.batch,
                'seconds':time.monotonic()-start,'max_completeness_error':float(np.abs(attr.sum((1,2),dtype=float)-(px-pb)).max()),
                'finite':True,'member_file_sha256':q.digest(dest),'worker_sha256':q.digest(BASE/'member_worker.py'),
                'instrumentation_sha256':q.digest(BASE/'checkpoint_integrator.py'),'runtime':gate['runtime']}
        q.savejson(f'runs/{args.tag}/member_{args.run:02d}_worker.json',report)
        q.emit('member_worker_finished',**report)
