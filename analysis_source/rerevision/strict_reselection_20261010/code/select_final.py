"""Select only on development predictions, freeze, refit, then open evaluation labels."""
from engine import *

def key(r):return '__'.join(str(r[x]) for x in ['condition','collection','group','encoder','family','config_index'])
def pooled():
 rows=[]
 for cond in CONDITIONS:
  for c in ['BAYC','MAYC']:
   for g in ['metadata','image','early']:
    for enc in (['TF_IDF'] if g=='metadata' else ENCODERS):
     for fam in FAMILIES:
      for num,config in enumerate(C.GRIDS[fam]):
       parts=[readj(ckpath(cond,c,g,enc,o,fam,num)) for o,e in FOLDS]
       valid=all(p['valid'] for p in parts)
       n=sum(p['n'] for p in parts);sse=sum(p['sse'] for p in parts) if valid else None
       r=dict(condition=cond,collection=c,group=g,encoder=enc,family=fam,config_index=num,config=config,valid=valid,n=n,mse_lrp=None if sse is None else sse/n,rmse=None if sse is None else float(np.sqrt(sse/n)))
       rows.append(r)
 return rows

def dev_arrays(r):
 ps=[];ys=[];ids=[]
 for o,e in FOLDS:
  p=ckpath(r['condition'],r['collection'],r['group'],r['encoder'],o,r['family'],r['config_index'])
  ck=readj(p);assert ck['valid'] and sha(p.with_suffix('.npy'))==ck['prediction_sha256']
  fold=pathprep(r['condition'],r['collection'],r['group'],r['encoder'],o)
  ps.append(np.load(p.with_suffix('.npy')));ys.append(np.load(fold/'valid_y.npy'));ids.append(np.load(fold/'valid_rows.npy'))
 return np.concatenate(ps),np.concatenate(ys),np.concatenate(ids)

def selections(rows,failed):
 reps=[];late_scores=[];family_choices=[]
 for cond in CONDITIONS:
  for c in ['BAYC','MAYC']:
   winners=[]
   for g in ['metadata','image','early']:
    for enc in (['TF_IDF'] if g=='metadata' else ENCODERS):
     for fam in FAMILIES:
      scores=[r for r in rows if (r['condition'],r['collection'],r['group'],r['encoder'],r['family'])==(cond,c,g,enc,fam) and key(r) not in failed]
      ordered=C.ordered_choices(fam,scores)
      if ordered:
       winners.append(ordered[0]);family_choices.append(ordered[0])
   def groupbest(g):return min([r for r in winners if r['group']==g],key=lambda r:(r['mse_lrp'],(['TF_IDF']+ENCODERS).index(r['encoder']),FAMILIES.index(r['family'])))
   chosen={g:groupbest(g) for g in ['metadata','image','early']}
   pm,y,ids=dev_arrays(chosen['metadata'])
   local=[]
   for r in [r for r in winners if r['group']=='image']:
    pi,yi,idi=dev_arrays(r);assert np.array_equal(yi,y) and np.array_equal(idi,ids)
    for w in np.arange(11)/10:
     pred=(1-w)*pm+w*pi;mse=float(np.mean((y-pred)**2))
     z=dict(condition=cond,collection=c,group='late',encoder=r['encoder'],family=r['family'],config_index=r['config_index'],config=r['config'],metadata_family=chosen['metadata']['family'],metadata_config=chosen['metadata']['config'],metadata_key=key(chosen['metadata']),image_key=key(r),image_weight=float(w),n=len(y),mse_lrp=mse,rmse=float(np.sqrt(mse)))
     local.append(z)
   best=min(local,key=lambda r:(r['mse_lrp'],ENCODERS.index(r['encoder']),FAMILIES.index(r['family']),r['image_weight']))
   reps.extend(chosen.values());reps.append(best);late_scores.extend(local)
 return reps,late_scores,family_choices

def fit_final(r):
 savej(OUT/'progress.json',dict(updated_utc=now(),stage='final_refit',candidate=key(r),completed_checkpoints=15930,expected_checkpoints=15930))
 dst=OUT/'final_models'/key(r);dst.mkdir(parents=True,exist_ok=True)
 if (dst/'manifest.json').exists():return readj(dst/'manifest.json')
 with threadpool_limits(limits=3):
  cond,c,g,enc=r['condition'],r['collection'],r['group'],r['encoder']
  prepare_task(cond,c,enc,'2025-01-01','2026-04-14',groups=[g])
  folder=pathprep(cond,c,g,enc,'2025-01-01');man,a=loadprep(folder,r['family'])
  before=ash(a['z']);model,offset,pz,status=fit(a,r['family'],r['config'],man['rank']);assert ash(a['z'])==before
  predictions=None if pz is None else C.target_inverse(pz,man['target_parameters'])
  valid=bool(status['valid'] and predictions is not None and np.isfinite(predictions).all())
  rec=dict(candidate=r,status=status,valid=valid,prepared=str(folder.relative_to(OUT)),prepared_sha256=sha(folder/'manifest.json'),completed_utc=now(),evaluation_labels_loaded=False)
  if valid:
   joblib.dump(dict(model=model,offset=offset,candidate=r,target_parameters=man['target_parameters'],preprocessing=str(folder.relative_to(OUT)),projection=man['projection']),dst/'model.joblib',compress=3)
   np.savez_compressed(dst/'token_predictions.npz',tokens=np.load(folder/'predict_tokens.npy'),predictions=predictions)
   rec.update(model_sha256=sha(dst/'model.joblib'),predictions_sha256=sha(dst/'token_predictions.npz'))
  savej(dst/'manifest.json',rec)
  print('FINAL_FIT',key(r),status,flush=True)
 return rec

def main():
 assert (OUT/'development_complete.json').exists()
 savej(OUT/'progress.json',dict(updated_utc=now(),stage='development_selection',completed_checkpoints=15930,expected_checkpoints=15930))
 rows=pooled();assert len(rows)==5310
 writecsv(OUT/'candidate_development_scores.csv',[dict(r,config=json.dumps(r['config'],sort_keys=True)) for r in rows])
 failed=set();iterations=[]
 reps,late,families=selections(rows,failed)
 freeze=dict(created_utc=now(),protocol_sha256=sha(OUT/'protocol_freeze.json'),evaluation_labels_loaded=False,prior_evaluation_inspection=True,representatives=reps,family_choices=families,candidate_scores_sha256=sha(OUT/'candidate_development_scores.csv'))
 p=OUT/'selection_freeze.json'
 if p.exists():assert readj(p)['representatives']==reps
 else:savej(p,freeze)
 while True:
  reps,late,families=selections(rows,failed)
  bykey={key(r):r for r in families};needed={}
  for r in reps:
   if r['group']=='late':needed[r['image_key']]=bykey[r['image_key']]
   else:needed[key(r)]=r
  newfails=[]
  for r in needed.values():
   result=fit_final(r)
   if not result['valid']:newfails.append(key(r))
  iterations.append(dict(completed_utc=now(),required=list(needed),failed=newfails))
  if not newfails:break
  failed.update(newfails)
  savej(OUT/'final_fit_fallbacks.json',dict(iterations=iterations,failed=sorted(failed),evaluation_labels_loaded=False))
 writecsv(OUT/'late_weight_development_scores.csv',[dict(r,config=json.dumps(r['config'],sort_keys=True),metadata_config=json.dumps(r['metadata_config'],sort_keys=True)) for r in late])
 writecsv(OUT/'family_selections.csv',[dict(r,config=json.dumps(r['config'],sort_keys=True)) for r in families])
 writecsv(OUT/'selected_models.csv',[dict(r,config=json.dumps(r['config'],sort_keys=True),metadata_config=json.dumps(r.get('metadata_config'),sort_keys=True)) for r in reps])
 savej(OUT/'final_selection_freeze.json',dict(created_utc=now(),initial_selection_sha256=sha(OUT/'selection_freeze.json'),representatives=reps,failed=sorted(failed),iterations=iterations,evaluation_labels_loaded=False,final_models={k:sha(OUT/'final_models'/k/'manifest.json') for k in needed}))
 savej(OUT/'progress.json',dict(updated_utc=now(),stage='final_evaluation',completed_checkpoints=15930,expected_checkpoints=15930))
 # Only now are retrospective evaluation targets read by this execution.
 savej(OUT/'evaluation_started.json',dict(created_utc=now(),final_selection_sha256=sha(OUT/'final_selection_freeze.json')))
 for c in ['BAYC','MAYC']:
  d=cohort(c,'base',phase='evaluation');assert np.all(d['times']>='2025-01-01') and np.all(d['times']<'2026-04-14')
  arrays=dict(source_rows=d['rowids'],tokens=d['tokens'],y_lrp=d['y'],times=d['times'].astype(str),strict=d['strict'],strict_no_wei=d['strict']&(d['prices']>1e-18*(1+1e-9)))
  for r in [r for r in reps if r['collection']==c]:
   def aligned(k):
    with np.load(OUT/'final_models'/k/'token_predictions.npz') as z:lookup=dict(zip(z['tokens'],z['predictions']))
    return np.array([lookup[t] for t in d['tokens']])
   p=aligned(key(r)) if r['group']!='late' else (1-r['image_weight'])*aligned(r['metadata_key'])+r['image_weight']*aligned(r['image_key'])
   assert np.isfinite(p).all();arrays[r['condition']+'__'+r['group']]=p
  np.savez_compressed(OUT/(c+'_evaluation_predictions.npz'),**arrays)
 savej(OUT/'evaluation_complete.json',dict(completed_utc=now(),selection_sha256=sha(OUT/'final_selection_freeze.json'),collections=['BAYC','MAYC']))
if __name__=='__main__':main()
