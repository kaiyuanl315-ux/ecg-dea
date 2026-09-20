from pathlib import Path
import sys,os
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'runtime_deps'))
os.environ['MPLCONFIGDIR']=str(BASE/'mplconfig')
os.environ['XDG_CACHE_HOME']=str(BASE/'cache')
import argparse,ast,hashlib,json,time,struct,zipfile,gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tsai.models.InceptionTime import InceptionTime

SOURCE=Path('/path/to/research/aiecg/RECOVERED_20260906/cursor_code/ensemble_main__cursor_b5af97a1.py')
WEIGHTS=Path('/path/to/research/aiecg/external_validation_package/model/ensemble10')
ARCHIVE=Path('/path/to/research/aiecg/data/multi_ecg_x.npz')
LOW=.456303122639656;HIGH=.6162797212600708
torch.set_num_threads(4);torch.set_num_interop_threads(1)
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for s in iter(lambda:f.read(8*1024*1024),b''):h.update(s)
 return h.hexdigest()
def emit(event,**kwargs):print(json.dumps({'event':event,**kwargs}),flush=True)
def savejson(name,d):
 p=BASE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False))
def mmap_array(name):
 with zipfile.ZipFile(ARCHIVE) as z:inf=z.getinfo(name+'.npy')
 assert inf.compress_type==zipfile.ZIP_STORED
 with open(ARCHIVE,'rb') as f:
  f.seek(inf.header_offset);h=f.read(30);a,b=struct.unpack('<HH',h[26:30]);f.seek(a+b,1)
  v=np.lib.format.read_magic(f)
  shape,fort,dtype=np.lib.format.read_array_header_1_0(f) if v==(1,0) else np.lib.format.read_array_header_2_0(f)
  offset=f.tell()
 return np.memmap(ARCHIVE,mode='r',dtype=dtype,shape=shape,offset=offset,order='F' if fort else 'C')
def load_model(k,device):
 ns={'torch':torch,'nn':nn,'InceptionTime':InceptionTime}
 tree=ast.parse(SOURCE.read_text());cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'TabMLP','MultiModalInception'}]
 assert len(cls)==2
 exec(compile(ast.Module(body=cls,type_ignores=[]),str(SOURCE),'exec'),ns)
 model=ns['MultiModalInception'](c_in=12,seq_len=1000,d_tab_in=12)
 path=WEIGHTS/f'run_{k:02d}_best_by_val_auprc.ckpt'
 ck=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ck['model_state'],strict=True)
 model.eval().to(device)
 for p in model.parameters():p.requires_grad_(False)
 assert all(not m.training for m in model.modules())
 return model,{'run':k,'sha256':digest(path),'epoch':ck.get('epoch'),'validation_ap':ck.get('val_auprc')}
def array_digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def prepare():
 m=pd.read_csv(BASE/'sample_metadata.csv');assert np.array_equal(m.sample_row_0based,np.arange(len(m)))
 arc=np.load(ARCHIVE,allow_pickle=False)
 ecg=np.array(mmap_array('arr_0')[m.npz_ecg_row_0based],dtype=np.float32)
 tab=np.array(arc['x_test'][m.npz_tabular_test_row_0based],dtype=np.float32)
 y=arc['y_test_72_death'][m.npz_label_test_row_0based]
 assert ecg.shape==(191,12,1000) and tab.shape==(191,12)
 assert np.isfinite(ecg).all() and np.isfinite(tab).all()
 assert np.array_equal(y,m.death_72h.to_numpy())
 expected=m[[f'proba_run{k}' for k in range(10)]].to_numpy(dtype=float)
 np.savez(BASE/'inputs.npz',ecg=ecg,tab=tab,y=y,expected=expected,expected_mean=m.proba_mean.to_numpy())
 sensitivity=[]
 for _,g in m.groupby(['death_72h','risk_stratum'],sort=False):
  for quantile in [.25,.75]:
   goal=g.proba_mean.quantile(quantile);idx=(g.proba_mean-goal).abs().idxmin();sensitivity.append(int(m.loc[idx,'sample_row_0based']))
 sensitivity=list(dict.fromkeys(sensitivity));assert len(sensitivity)==12
 savejson('input_manifest.json',{'n':len(m),'ecg_shape':list(ecg.shape),'tab_shape':list(tab.shape),'ecg_sha256':array_digest(ecg),'tab_sha256':array_digest(tab),'labels_sha256':array_digest(y),'source_sha256':digest(SOURCE),'sample_metadata_sha256':digest(BASE/'sample_metadata.csv'),'sensitivity_indices':sensitivity,'sampling_before_attribution':True})
 emit('prepared',n=len(m),sensitivity_indices=sensitivity)
def predict(model,x,c,device,batch=64):
 out=[]
 with torch.no_grad():
  for i in range(0,len(x),batch):
   xx=torch.tensor(x[i:i+batch],device=device);cc=torch.tensor(c[i:i+batch],device=device)
   out.extend(torch.sigmoid(model(xx,cc)).reshape(-1).cpu().numpy().astype(float))
 return np.asarray(out)
def compare(a,b):
 dif=np.abs(a-b)
 return {'max_absolute_error':float(dif.max()),'mean_absolute_error':float(dif.mean()),'outside_prior_tolerance':int((dif>5e-5+1e-4*np.abs(b)).sum())}
def verify():
 a=np.load(BASE/'inputs.npz');report={'source_sha256':digest(SOURCE),'training_performed':False,'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},'devices':{}}
 for device in ['cpu','mps']:
  if device=='mps' and not torch.backends.mps.is_available():continue
  pred=[];info=[];start=time.monotonic()
  for k in range(10):
   model,meta=load_model(k,device);p=predict(model,a['ecg'],a['tab'],device);pred.append(p)
   info.append({**meta,**compare(p,a['expected'][:,k])})
   emit('prediction_member',device=device,**info[-1]);del model
  pred=np.stack(pred,axis=1);ens=pred.mean(1)
  entry={'members':info,'ensemble':compare(ens,a['expected_mean']),'threshold_changes':{str(t):int(((ens>=t)!=(a['expected_mean']>=t)).sum()) for t in [LOW,HIGH]},'seconds':time.monotonic()-start}
  report['devices'][device]=entry;np.save(BASE/f'predictions_{device}.npy',pred);savejson('prediction_verification.json',report)
  emit('prediction_device',device=device,ensemble=entry['ensemble'],threshold_changes=entry['threshold_changes'],seconds=entry['seconds'])
 if 'mps' in report['devices']:
  report['cpu_vs_mps']=compare(np.load(BASE/'predictions_cpu.npy'),np.load(BASE/'predictions_mps.npy'))
 report['cpu_prediction_gate_pass']=all(x['outside_prior_tolerance']==0 for x in report['devices']['cpu']['members']) and report['devices']['cpu']['ensemble']['outside_prior_tolerance']==0
 report['mps_ensemble_gate_pass']=report['devices'].get('mps',{}).get('ensemble',{}).get('outside_prior_tolerance',1)==0
 savejson('prediction_verification.json',report)
 emit('verification_finished',cpu_gate=report['cpu_prediction_gate_pass'],mps_gate=report['mps_ensemble_gate_pass'])
def ig_one(model,x,c,b,device,steps,batch=64):
 nodes,w=np.polynomial.legendre.leggauss(steps);nodes=(nodes+1)/2;w=w/2
 n=len(x);diff=x-b;integral=np.zeros(x.shape,dtype=np.float64)
 for start in range(0,n*steps,batch):
  ix=np.arange(start,min(start+batch,n*steps));case=ix//steps;node=ix%steps
  xx=torch.tensor(b[case]+diff[case]*nodes[node,None,None].astype(np.float32),device=device,requires_grad=True)
  cc=torch.tensor(c[case],device=device)
  p=torch.sigmoid(model(xx,cc)).sum();g=torch.autograd.grad(p,xx,create_graph=False)[0].detach().cpu().numpy()
  np.add.at(integral,case,g.astype(np.float64)*w[node,None,None])
  del xx,cc,p,g
 return (diff*integral).astype(np.float32)
def baseline(x,kind):return np.zeros_like(x) if kind=='zero' else np.broadcast_to(x.mean(-1,keepdims=True),x.shape).copy()
def integrate(device,steps,kind,subset,tag,batch):
 a=np.load(BASE/'inputs.npz');indices=np.arange(len(a['ecg']))
 if subset=='sensitivity':indices=np.array(json.loads((BASE/'input_manifest.json').read_text())['sensitivity_indices'])
 elif subset.startswith('indices:'):indices=np.array([int(i) for i in subset.split(':',1)[1].split(',')])
 x=a['ecg'][indices];c=a['tab'][indices];b=baseline(x,kind);directory=BASE/'runs'/tag;directory.mkdir(parents=True,exist_ok=True)
 start=time.monotonic();manifest={'tag':tag,'device':device,'steps':steps,'baseline':kind,'sample_indices':indices.tolist(),'target':'mean of ten raw sigmoid member outputs; tabular inputs fixed','members':[]}
 for k in range(10):
  dest=directory/f'member_{k:02d}.npz'
  provenance={'baseline':kind,'device':device,'input_sha256':digest(BASE/'inputs.npz'),'source_sha256':digest(SOURCE),'checkpoint_sha256':digest(WEIGHTS/f'run_{k:02d}_best_by_val_auprc.ckpt')}
  if dest.exists():
   saved=np.load(dest);assert np.array_equal(saved['indices'],indices) and int(saved['steps'])==steps
   assert json.loads(str(saved['provenance']))==provenance
   manifest['members'].append({'run':k,'resumed':True});continue
  st=time.monotonic();model,meta=load_model(k,device)
  attr=ig_one(model,x,c,b,device,steps,batch=batch)
  px=predict(model,x,c,device);pb=predict(model,b,c,device)
  np.savez_compressed(dest,ig=attr,prediction=px,baseline_prediction=pb,indices=indices,steps=steps,provenance=json.dumps(provenance))
  result={**meta,'seconds':time.monotonic()-st,'max_completeness_error':float(np.max(np.abs(attr.sum((1,2),dtype=np.float64)-(px-pb))))}
  manifest['members'].append(result);savejson(f'runs/{tag}/manifest.json',manifest)
  emit('ig_member_finished',tag=tag,**result);del model,attr;gc.collect()
  if device=='mps':torch.mps.empty_cache()
 ig=np.zeros(x.shape,dtype=np.float64);ps=[];bs=[]
 for k in range(10):
  r=np.load(directory/f'member_{k:02d}.npz');ig+=r['ig'].astype(np.float64)/10;ps.append(r['prediction']);bs.append(r['baseline_prediction'])
 pred=np.mean(ps,axis=0);base=np.mean(bs,axis=0);difference=pred-base;residual=ig.sum((1,2))-difference;tol=.0005+.02*np.abs(difference)
 np.savez_compressed(directory/'ensemble.npz',ig=ig.astype(np.float32),prediction=pred,baseline_prediction=base,indices=indices,residual=residual,tolerance=tol)
 manifest['seconds']=time.monotonic()-start;manifest['ensemble_completeness']={'max_absolute_residual':float(np.abs(residual).max()),'median_absolute_residual':float(np.median(np.abs(residual))),'n_failed':int((np.abs(residual)>tol).sum()),'n':len(indices)}
 savejson(f'runs/{tag}/manifest.json',manifest);emit('ig_ensemble_finished',tag=tag,**manifest['ensemble_completeness'],seconds=manifest['seconds'])
def unit_check():
 # A differentiable model with an analytic IG reference, independent of the neural model.
 class LinearProbability(nn.Module):
  def forward(self,x,c):
   p=.4+.05*x.sum((1,2))+.01*c.sum(1)
   return torch.logit(p).unsqueeze(1)
 x=np.arange(12,dtype=np.float32).reshape(2,2,3)/100;c=np.ones((2,2),dtype=np.float32);b=np.zeros_like(x)
 got=ig_one(LinearProbability(),x,c,b,'cpu',16,16);err=float(np.abs(got-.05*x).max());assert err<1e-8
 savejson('integrator_unit_check.json',{'status':'PASS','analytic_linear_max_error':err})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','verify','unit','integrate']);p.add_argument('--device',default='mps');p.add_argument('--steps',type=int,default=128);p.add_argument('--baseline',default='zero',choices=['zero','lead_mean']);p.add_argument('--subset',default='all');p.add_argument('--tag',default='zero_128');p.add_argument('--batch',type=int,default=64);args=p.parse_args()
 if args.mode=='prepare':prepare()
 elif args.mode=='verify':verify()
 elif args.mode=='unit':unit_check()
 else:integrate(args.device,args.steps,args.baseline,args.subset,args.tag,args.batch)
