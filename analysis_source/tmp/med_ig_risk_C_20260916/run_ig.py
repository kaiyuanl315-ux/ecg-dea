"""Reuse the verified frozen-model integrator in an isolated new output directory."""
from pathlib import Path
import sys, os, importlib.util, argparse, json
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'med_ig_20260915'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('verified_ig',OLD/'ig_analysis.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.BASE=BASE
os.environ['MPLCONFIGDIR']=str(BASE/'mplconfig')
os.environ['XDG_CACHE_HOME']=str(BASE/'cache')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['verify','integrate','unit'])
    p.add_argument('--indices',default='')
    p.add_argument('--tag',default='new_high_zero_128')
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--device',default='mps')
    args=p.parse_args()
    if args.mode=='verify':q.verify()
    elif args.mode=='unit':q.unit_check()
    else:
        assert args.indices
        q.integrate(args.device,args.steps,'zero','indices:'+args.indices,args.tag,64)
