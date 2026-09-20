"""Apply blinded QRS decisions, summarize encounters, and assess sampling stability."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
BASE=Path(__file__).resolve().parent
ALIGN=BASE/'alignment'
LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
DISPLAY=np.array([0,1,2,3,5,4,6,7,8,9,10,11])
TIME=np.arange(-40,60)/100

def js(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def apply_qc():
    x=np.load(BASE/'inputs.npz')['ecg']
    records=json.loads((ALIGN/'detection_records_private.json').read_text())
    mapping=json.loads((ALIGN/'private_alias_index_map.json').read_text())['rows']
    by_alias={d['alias']:d for d in records};row_by_alias={d['alias']:d['input_row_0based'] for d in mapping}
    decisions=[];sources={}
    for name in ['review_01_32.json','review_33_64.json','review_65_96.json']:
        p=ALIGN/name;review=json.loads(p.read_text());sources[name]=sha(p)
        for original in review['records']:
            d=dict(original)
            if 'status' not in d:
                translation={'retain_all_complete_window_anchors':'accept',
                    'retain_record_delete_false_anchors':'remove_anchors',
                    'retain_all_anchors':'accept','remove_clearly_non_qrs_anchors':'remove_anchors'}
                assert d.get('decision') in translation,d
                d['status']=translation[d['decision']]
            if 'remove_peaks' not in d:d['remove_peaks']=d.get('delete_peaks',[])
            assert not d.get('add_peaks',[]) and not d.get('shift_peaks',[]),d
            decisions.append(d)
    assert len(decisions)==1536 and len({d['alias'] for d in decisions})==1536
    assert set(d['alias'] for d in decisions)==set(by_alias)
    wave=np.full((1536,12,100),np.nan);nbeats=np.zeros(1536,int);retained=[]
    for d in decisions:
        assert d['reviewed'] and d['status'] in ['accept','remove_anchors','exclude'],d
        r=by_alias[d['alias']];row=row_by_alias[d['alias']]
        removed=d.get('remove_peaks',[])
        assert set(removed)<=set(r['detected_peaks']),d
        keep=[int(p) for p in r['complete_window_peaks'] if p not in removed]
        if d['status']=='exclude':
            assert d.get('reason') and len(keep)<3,d
        eligible=len(keep)>=3
        nbeats[row]=len(keep)
        if eligible:
            wave[row]=np.median(np.stack([x[row,:,p-40:p+60].astype(float) for p in keep]),axis=0)
        retained.append({'alias':d['alias'],'row':row,'peaks':keep,'eligible':eligible,'removed_peaks':removed,
                         'reason':d.get('reason',''),'status':d['status']})
    retained.sort(key=lambda d:d['row'])
    assert [d['row'] for d in retained]==list(range(1536))
    js(ALIGN/'retained_peaks_private.json',retained)
    np.savez_compressed(ALIGN/'encounter_waveforms_qc.npz',waveforms=wave,eligible=nbeats>=3,nbeats=nbeats)
    m=pd.read_csv(BASE/'sample_metadata.csv',dtype={'subject_id':str})
    summaries=[]
    for risk in ['high','low']:
        mask=(m.risk_stratum==risk).to_numpy();selected=m[mask];kept=m[mask&(nbeats>=3)]
        summaries.append({'risk':risk,'selected':int(mask.sum()),'retained':len(kept),'excluded':len(selected)-len(kept),
                          'selected_patients':int(selected.subject_id.nunique()),'retained_patients':int(kept.subject_id.nunique()),
                          'retained_endpoint_events':int(kept.death_72h.sum()),
                          'retained_beats_range':[int(nbeats[mask&(nbeats>=3)].min()),int(nbeats[mask&(nbeats>=3)].max())]})
    report={'n_reviewed':1536,'n_retained':int((nbeats>=3).sum()),'n_excluded':int((nbeats<3).sum()),
            'removed_anchor_count':sum(len(d['removed_peaks']) for d in retained),
            'records_with_anchor_deletions':sum(bool(d['removed_peaks']) for d in retained),
            'groups':summaries,'review_source_sha256':sources,'AI_assisted_not_clinician_adjudicated':True,
            'all_window_values_unchanged':True,'minimum_complete_windows':3}
    js(BASE/'Alignment_QA.json',report);print(json.dumps(report),flush=True)

def summarize():
    z=np.load(ALIGN/'encounter_waveforms_qc.npz');wave=z['waveforms'];eligible=z['eligible']
    igfile=np.load(BASE/'high_risk_attributions.npz');rawig=igfile['ig']
    assert np.all(np.abs(rawig.sum((1,2),dtype=np.float64)-(igfile['prediction']-igfile['baseline_prediction']))<=igfile['tolerance'])
    peaks=json.loads((ALIGN/'retained_peaks_private.json').read_text())
    alignedig=np.full((512,12,100),np.nan)
    for d in peaks[:512]:
        if d['eligible']:
            i=d['row'];alignedig[i]=np.mean(np.stack([np.abs(rawig[i,:,p-40:p+60]).astype(float) for p in d['peaks']]),axis=0)
    m=pd.read_csv(BASE/'sample_metadata.csv',dtype={'subject_id':str})
    hi=np.flatnonzero(eligible&(m.risk_stratum=='high').to_numpy());lo=np.flatnonzero(eligible&(m.risk_stratum=='low').to_numpy())
    wh=np.median(wave[hi],axis=0);wl=np.median(wave[lo],axis=0);igh=np.mean(alignedig[hi],axis=0)
    np.savez_compressed(BASE/'aligned_summaries_private.npz',encounter_waveforms=wave,encounter_IG_high=alignedig,
                        high_indices=hi,low_indices=lo,high_median=wh,low_median=wl,high_mean_absolute_IG=igh)
    rows=[]
    for lead,j in zip(LEADS,DISPLAY):
        for t in range(100):rows.append({'lead':lead,'time_from_QRS_s':TIME[t],
            'high_risk_median_mV':wh[j,t],'low_risk_median_mV':wl[j,t],
            'high_risk_mean_absolute_IG':igh[j,t],'high_risk_n':len(hi),'low_risk_n':len(lo)})
    pd.DataFrame(rows).to_csv(BASE/'Panel_C_aligned_waveforms_and_IG.csv',index=False)
    diagnostics=[]
    for risk,full,group,limits in [('high',wh,hi,[128,256,512]),('low',wl,lo,[256,512,1024])]:
        for limit in limits:
            ii=group[m.iloc[group].sample_rank_within_risk.to_numpy()<=limit]
            candidate=np.median(wave[ii],axis=0);dif=candidate-full
            for lead,j in zip(LEADS,DISPLAY):
                rec={'risk':risk,'prefix_selected_n':limit,'retained_n':len(ii),'lead':lead,
                     'waveform_max_absolute_difference_mV':float(np.abs(dif[j]).max()),
                     'waveform_RMS_difference_mV':float(np.sqrt(np.mean(dif[j]**2))),
                     'full_waveform_peak_to_peak_mV':float(np.ptp(full[j]))}
                if risk=='high':
                    a=np.mean(alignedig[ii],axis=0)
                    rec.update({'IG_relative_L1':float(np.abs(a[j]-igh[j]).sum()/igh[j].sum()),
                                'IG_temporal_spearman':float(spearmanr(a[j],igh[j]).statistic),
                                'IG_peak_time_difference_ms':int((np.argmax(a[j])-np.argmax(igh[j]))*10)})
                diagnostics.append(rec)
    pd.DataFrame(diagnostics).to_csv(BASE/'Sampling_stability_by_lead.csv',index=False)
    # Resample the union of patients, so repeat encounters and patients spanning
    # high/low strata receive the same multiplicity in both groups.
    patients=m.subject_id.to_numpy();eligible_rows=np.flatnonzero(eligible)
    unique=np.unique(patients[eligible_rows]);patient_rows=[eligible_rows[patients[eligible_rows]==p] for p in unique]
    rng=np.random.default_rng(20260918);bh=[];bl=[];bi=[]
    for b in range(300):
        rows=np.concatenate([patient_rows[i] for i in rng.integers(0,len(unique),len(unique))])
        h=rows[rows<512];l=rows[rows>=512]
        assert len(h)>0 and len(l)>0
        bh.append(np.median(wave[h],axis=0));bl.append(np.median(wave[l],axis=0));bi.append(np.mean(alignedig[h],axis=0))
    bh=np.stack(bh);bl=np.stack(bl);bi=np.stack(bi)
    quantiles={name:np.quantile(array,[.025,.975],axis=0) for name,array in [('high',bh),('low',bl),('difference',bh-bl),('IG',bi)]}
    bootrows=[]
    for lead,j in zip(LEADS,DISPLAY):
        for t in range(100):
            row={'lead':lead,'time_from_QRS_s':TIME[t]}
            for name,a in quantiles.items():row.update({name+'_pointwise_2.5pct':a[0,j,t],name+'_pointwise_97.5pct':a[1,j,t]})
            bootrows.append(row)
    pd.DataFrame(bootrows).to_csv(BASE/'Patient_cluster_bootstrap_pointwise_intervals.csv',index=False)
    global_checks=[]
    for limit in [128,256,512]:
        ix=hi[m.iloc[hi].sample_rank_within_risk.to_numpy()<=limit];v=np.mean(alignedig[ix],axis=0)
        shares=lambda a:a.sum(1)/a.sum()*100
        global_checks.append({'prefix_selected_n':limit,'retained_n':len(ix),'IG_relative_L1':float(np.abs(v-igh).sum()/igh.sum()),
                              'IG_flattened_spearman':float(spearmanr(v.ravel(),igh.ravel()).statistic),
                              'lead_share_spearman':float(spearmanr(shares(v),shares(igh)).statistic),
                              'max_lead_share_difference_pp':float(np.abs(shares(v)-shares(igh)).max())})
    color={'max':float(igh.max()),'P98':float(np.quantile(igh,.98)),'P99':float(np.quantile(igh,.99)),
           'max_over_P98':float(igh.max()/np.quantile(igh,.98)),'max_over_P99':float(igh.max()/np.quantile(igh,.99)),
           'normalization':'One common linear scale, 0 to actual maximum; no clipping or IG smoothing'}
    report={'high_n':len(hi),'low_n':len(lo),'IG_global_prefix_comparisons':global_checks,'color_distribution':color,
            'patient_cluster_bootstrap':{'replicates':300,'seed':20260918,'patients':len(unique),
              'design':'Resample eligible patient union with replacement; all eligible encounters of each selected patient retain multiplicity in both risk groups.',
              'intervals':'Exploratory pointwise 2.5/97.5 percentiles, not simultaneous bands; conditional on sampled QC-eligible encounters; no finite population correction.'},
            'waveform_vs_IG':'Median group waveform is not the input on which averaged IG was computed.',
            'source_hashes':{p:sha(BASE/p) for p in ['high_risk_attributions.npz','Alignment_QA.json','sample_metadata.csv']}}
    js(BASE/'Sampling_stability_QA.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qc','summarize']);args=p.parse_args()
    apply_qc() if args.mode=='qc' else summarize()
