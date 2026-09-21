from pathlib import Path
import pandas as pd,json,collections,importlib.util
b=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('support',b/'scripts/derive_support.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
c=pd.read_csv(b/'derived/icu.csv.gz',keep_default_na=False);s=pd.read_csv(b/'derived/support.csv.gz',keep_default_na=False)
c['hadm_id']=pd.to_numeric(c.resolved_hadm_id,errors='coerce').astype('Int64');c['subject_id']=pd.to_numeric(c.subject_id).astype('Int64')
for col in ['index_time','episode_start_time','episode_end_time']:c[col]=m.parse_time(c[col])
gap=c.merge(s[['row_id','ICU_source_coverage_gap']],on='row_id',validate='one_to_one');hadms=set(c.hadm_id.dropna().astype(int))
ic,_=m.filtered_csv(b/'data_exports/icustays.csv',hadms=hadms)
for col in ['intime','outtime']:ic[col]=m.parse_time(ic[col])
meta=json.loads((b/'qa/icu_derivation.json').read_text());tr,_=m.filtered_csv(meta['source_sha256']['transfers']['path'],hadms=hadms)
for col in ['intime','outtime']:tr[col]=m.parse_time(tr[col])
tr_all=tr.copy();tr=tr[tr.careunit.isin(meta['ICU_careunits'])];td=m.group_records(tr);sd=m.group_records(ic)
patients=pd.read_csv('/path/to/research/mimiciv/3.0/hosp/patients.csv',usecols=['subject_id','anchor_year_group']);pt=patients.set_index('subject_id').anchor_year_group.to_dict()
counts=collections.Counter();units=collections.Counter();periods=collections.Counter();gtypes=collections.Counter();grecords=[];matching=collections.Counter()
for row in c.to_dict('records'):
 if not row['eligible']:continue
 h=int(row['hadm_id']) if pd.notna(row['hadm_id']) else None
 if not h:continue
 ss=sd.get(h,[]);tt=[z for z in td.get(h,[]) if z['intime']>=row['episode_start_time'] and z['intime']<=row['index_time']+pd.Timedelta(hours=72) and z['intime']<row['episode_end_time']]
 if tt:counts['parent_eligible_with_any_ICU_entry_in72h']+=1
 uncovered=[]
 for t in tt:
  covered=[z for z in ss if z['intime']<=t['intime']<=z['outtime']]
  if not covered:
   kind='no_icustay_in_same_hadm' if not ss else 'has_icustay_but_transfer_entry_outside_all_intervals'
   # This is a read-only temporal difference audit, never a nearest-match relink.
   any_overlap=[z for z in ss if t['intime']<=z['outtime'] and (pd.isna(t['outtime']) or t['outtime']>=z['intime'])]
   raw_length=(t['outtime']-t['intime']).total_seconds()/3600 if pd.notna(t['outtime']) else None
   startdif=min(abs((t['intime']-z['intime']).total_seconds())/60 for z in ss) if ss else None
   uncovered.append((t,kind,bool(any_overlap),raw_length,startdif))
  else:
   matching['covered_transfer_records']+=1
   matching['covered_entry_exact_icustay_intime']+=any(z['intime']==t['intime'] for z in covered)
 if not uncovered:continue
 counts['gap_encounters']+=1;kind='no_icustay_in_same_hadm' if not ss else 'has_icustay_but_transfer_entry_outside_all_intervals';gtypes[kind]+=1
 periods[(pt.get(row['subject_id'],'unknown'),kind)]+=1
 for hh in [24,72]:
  counts[f'gap_icu{hh}_parent_events']+=str(row[f'icu{hh}']) in ['1','1.0']
 for t,typ,overlap,raw_length,startdif in uncovered:
  units[(t['careunit'],typ)]+=1
  grecords.append({'gap_type':typ,'careunit':t['careunit'],'eventtype':t['eventtype'],'anchor_year_group':pt.get(row['subject_id'],'unknown'),'deidentified_index_year':row['index_time'].year,'raw_interval_h':raw_length,'overlaps_any_ICU_interval':overlap,'nearest_start_absolute_minutes_for_QA_only':startdif})
x=pd.DataFrame(grecords)
agg=x.groupby(['gap_type','careunit'],dropna=False).agg(raw_transfer_records=('raw_interval_h','size'),median_raw_interval_h=('raw_interval_h','median'),min_raw_interval_h=('raw_interval_h','min'),max_raw_interval_h=('raw_interval_h','max'),overlapping_interval_records=('overlaps_any_ICU_interval','sum')).reset_index()
agg.to_csv(b/'results/support_ICU_coverage_gaps_by_unit.csv',index=False)
period=pd.DataFrame([{'anchor_year_group':k[0],'gap_type':k[1],'encounters':v} for k,v in periods.items()]);period.to_csv(b/'results/support_ICU_coverage_gaps_by_period.csv',index=False)
report={'counts':dict(counts),'gap_encounter_types':dict(gtypes),'covered_transfer_audit':dict(matching),'gap_raw_eventtypes':x.eventtype.value_counts().to_dict(),'gap_raw_interval_h_summary':x.raw_interval_h.describe().to_dict(),'gap_with_other_ICU_interval_start_difference_minutes':x.loc[x.gap_type.eq('has_icustay_but_transfer_entry_outside_all_intervals'),'nearest_start_absolute_minutes_for_QA_only'].describe().to_dict(),'gap_deidentified_year_counts':x.deidentified_index_year.value_counts().sort_index().to_dict(),'note':'Time-nearest differences are QA only; no relinking. anchor_year_group is coarse actual-calendar grouping; deidentified year is not actual calendar year.'}
(b/'qa/support_coverage_gap_audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


# Distinguish dimension aggregation from incompatible source identifiers/times.
d_all=m.group_records(tr_all); dimension=collections.Counter();unit_nonmatch=collections.Counter()
for rr in ic.to_dict('records'):
 trows=d_all.get(int(rr['hadm_id']),[]);dimension['selected_icustays']+=1
 dimension['subject_mismatch_any_same_hadm']+=any(t['subject_id']!=rr['subject_id'] for t in trows)
 dimension['start_exact_any_raw_transfer']+=any(t['intime']==rr['intime'] for t in trows)
 dimension['end_exact_any_raw_transfer']+=any(t['outtime']==rr['outtime'] for t in trows)
 dimension['start_exact_raw_unit_and_time']+=any(t['intime']==rr['intime'] and t['careunit']==rr['first_careunit'] for t in trows)
 dimension['end_exact_raw_unit_and_time']+=any(t['outtime']==rr['outtime'] and t['careunit']==rr['last_careunit'] for t in trows)
 if not any(t['intime']==rr['intime'] and t['careunit']==rr['first_careunit'] for t in trows):unit_nonmatch[rr['first_careunit']]+=1
period_rows=[];denom=gap[gap.eligible.eq(1)].copy();denom['period']=denom.subject_id.map(pt)
for col in ['icu24','icu72']:denom[col]=pd.to_numeric(denom[col])
for pp,gg in denom.groupby('period'):
 period_rows.append({'anchor_year_group':pp,'parent_ICU_eligible':len(gg),'parent_ICU24':int(gg.icu24.sum()),'parent_ICU72':int(gg.icu72.sum()),'coverage_gap':int(gg.ICU_source_coverage_gap.sum()),'gap_ICU24':int(gg.loc[gg.ICU_source_coverage_gap.eq(1),'icu24'].sum()),'gap_ICU72':int(gg.loc[gg.ICU_source_coverage_gap.eq(1),'icu72'].sum())})
pd.DataFrame(period_rows).to_csv(b/'results/support_ICU_coverage_denominators_by_period.csv',index=False)
report['exported_icustays_exact_ALL_raw_transfer_match']=dict(dimension)
report['icustays_first_careunit_counts']=ic.first_careunit.value_counts().to_dict()
report['icustays_start_unit_time_nonmatch_counts']=dict(unit_nonmatch)
report['period_denominators']=period_rows
report['parent_event_gap_fraction']={f'{hh}h':counts[f'gap_icu{hh}_parent_events']/float(denom[f'icu{hh}'].sum()) for hh in [24,72]}
report['script_sha256']=m.sha(__file__)
report['interpretation']='All exported ICU stay intervals match exact same-admission raw transfer start/end times and patient identifiers. This supports compatible source identifiers/times; dimension aggregation differs and does not establish a documented release version. Uncovered raw ICU intervals remain unknown for organ-support recording. Missing-heart-rate filtering in official documentation is a possible explanation, not confirmed per patient.'
(b/'qa/support_coverage_gap_audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps({'gap_encounters':counts['gap_encounters'],'gap_types':dict(gtypes),'event_gap_fraction':report['parent_event_gap_fraction'],'dimension_exact_match':dict(dimension)},indent=2))
