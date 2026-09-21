from pathlib import Path
import os,sys,json,hashlib,warnings
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
os.environ['MPLCONFIGDIR']=str(BASE/'mplconfig')
sys.path.insert(0,str(BASE/'deps'))
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
SRC=ROOT/'tmp/med_ig_20260915'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
npz=np.load(SRC/'sample_arrays.npz',allow_pickle=False)
ecg=npz['ecg'].astype(float);ig=npz['ig_zero'].astype(float)
assert ecg.shape==ig.shape==(191,12,1000) and np.isfinite(ecg).all() and np.isfinite(ig).all()
metadata=pd.read_csv(SRC/'sample_metadata.csv')
assert np.array_equal(metadata.sample_row_0based,np.arange(191))
perm=np.random.default_rng(20260916).permutation(191)
alias_by_index={int(idx):f'R{j+1:03d}' for j,idx in enumerate(perm)}
records=[];peaks_by_index={};window_counts=[]
waves=np.full((191,12,100),np.nan);attrs=np.full_like(waves,np.nan)
for i in range(191):
 signal=ecg[i,1,:]
 caught=[];error='';inverted=None;peaks=np.array([],dtype=int)
 try:
  with warnings.catch_warnings(record=True) as wrn:
   warnings.simplefilter('always')
   oriented,inverted=nk.ecg_invert(signal,sampling_rate=100,force=False,show=False)
   clean=nk.ecg_clean(oriented,sampling_rate=100,method='neurokit')
   _,info=nk.ecg_peaks(clean,sampling_rate=100,method='neurokit',correct_artifacts=False)
   peaks=np.asarray(info['ECG_R_Peaks'],dtype=int)
   caught=[str(w.message) for w in wrn]
 except Exception as e:error=type(e).__name__+': '+str(e)
 assert np.all(np.diff(peaks)>0) and np.all((peaks>=0)&(peaks<1000))
 valid=peaks[(peaks>=40)&(peaks+60<=1000)]
 rr=np.diff(peaks)/100
 flags=[]
 if len(valid)<3:flags.append('fewer_than_3_complete_windows')
 if len(rr) and rr.min()<.25:flags.append('RR_below_0.25_s')
 if len(rr) and rr.max()>2.5:flags.append('RR_above_2.5_s')
 if error:flags.append('detector_error')
 if len(valid)>=3:
  w=np.stack([ecg[i,:,p-40:p+60] for p in valid])
  a=np.stack([np.abs(ig[i,:,p-40:p+60]) for p in valid])
  assert w.shape==a.shape==(len(valid),12,100)
  waves[i]=np.median(w,axis=0);attrs[i]=np.mean(a,axis=0)
 rec={'alias':alias_by_index[i],'sample_row':i,'polarity_inverted_for_detection_only':bool(inverted),
  'detected_peaks':peaks.tolist(),'complete_window_peaks':valid.tolist(),'n_detected':len(peaks),'n_complete_windows':len(valid),
  'RR_min_seconds':float(rr.min()) if len(rr) else None,'RR_max_seconds':float(rr.max()) if len(rr) else None,
  'flags':flags,'warnings':caught,'error':error,'automatic_eligibility':len(valid)>=3}
 records.append(rec);peaks_by_index[i]=peaks;window_counts.append(len(valid))

(BASE/'detection_records_private.json').write_text(json.dumps(records,indent=2))
np.savez_compressed(BASE/'encounter_aligned_arrays.npz',waveform_median=waves,absolute_ig_mean=attrs,n_windows=np.array(window_counts))
qa=BASE/'qc_blinded';qa.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.linewidth':.4})
t=np.arange(1000)/100
for page in range(12):
 indices=perm[page*16:(page+1)*16]
 fig,axes=plt.subplots(16,1,figsize=(12,16),sharex=True)
 fig.subplots_adjust(left=.12,right=.985,bottom=.03,top=.977,hspace=.40)
 fig.suptitle(f'Blinded QRS detection review — sheet {page+1:02d} (red=complete window, gray=boundary)',fontsize=10,y=.995)
 for ax,i in zip(axes,indices):
  rec=records[int(i)];s=ecg[i,1]
  ax.plot(t,s,color='#222222',lw=.65)
  for pk in rec['detected_peaks']:
   col='#bb3333' if pk in rec['complete_window_peaks'] else '#aaaaaa'
   ax.axvline(pk/100,color=col,lw=.6,alpha=.65)
   ax.scatter([pk/100],[s[pk]],s=7,color=col,zorder=3)
  ax.set_ylabel(rec['alias'],rotation=0,ha='right',va='center',labelpad=17,fontsize=9,weight='bold')
  ax.set_xlim(0,10);ax.tick_params(axis='y',labelsize=6,length=2,pad=2)
  ax.text(.995,.92,f"windows={rec['n_complete_windows']} · invert={int(rec['polarity_inverted_for_detection_only'])}"+(" · REVIEW" if rec['flags'] else ''),transform=ax.transAxes,ha='right',va='top',fontsize=6,bbox={'facecolor':'white','edgecolor':'none','alpha':.7,'pad':1})
  ax.spines[['top','right']].set_visible(False)
 for ax in axes[len(indices):]:ax.set_visible(False)
 axes[min(len(indices),16)-1].set_xlabel('Time (s)')
 fig.savefig(qa/f'sheet_{page+1:02d}.png',dpi=160)
 plt.close(fig)

summary={'n':191,'fs_hz':100,'window_samples':[-40,60],'neurokit_version':nk.__version__,
 'automatic_eligible':sum(r['automatic_eligibility'] for r in records),
 'flagged_aliases':[{k:r[k] for k in ['alias','n_complete_windows','flags','error']} for r in records if r['flags']],
 'source_arrays_sha256':sha(SRC/'sample_arrays.npz'),'output_sha256':sha(BASE/'encounter_aligned_arrays.npz'),
 'detection_polarity_inversion_n':sum(r['polarity_inverted_for_detection_only'] for r in records),'input_ECG_unchanged':True,
 'waveform_or_IG_smoothing':False,'review_state':'blinded_visual_review_pending'}
(BASE/'alignment_QA.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
