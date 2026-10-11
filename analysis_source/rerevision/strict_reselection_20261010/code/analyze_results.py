from engine import *
sys.path.insert(0,str(ROOT/'rerevision/seen_unseen_20261010/code'))
import run_seen_unseen as stats

def main():
 savej(OUT/'progress.json',dict(updated_utc=now(),stage='paired_bootstrap_analysis',completed_checkpoints=15930,expected_checkpoints=15930))
 selected=readj(OUT/'final_selection_freeze.json')['representatives']
 metrics=[];cis=[];counts=[];changes=[]
 for c in ['BAYC','MAYC']:
  with np.load(OUT/(c+'_evaluation_predictions.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  names=['zero']+[cond+'__'+g for cond in CONDITIONS for g in ['metadata','image','early','late']]
  pred=np.column_stack([np.zeros(len(d['y_lrp']))]+[d[n] for n in names[1:]])
  blocks=np.array([(datetime.fromisoformat(t)-datetime(2025,1,1,tzinfo=timezone.utc)).days//14 for t in d['times']])
  for subset,mask in [('base',np.ones(len(d['y_lrp']),dtype=bool)),('strict',d['strict'])]:
   y,p,t,b=d['y_lrp'][mask],pred[mask],d['tokens'][mask],blocks[mask]
   vals=stats.metrics(y,p,t);tr,eq=stats.bootstrap(y,p,t,True,c+subset+'token');br,_=stats.bootstrap(y,p,b,False,c+subset+'calendar')
   counts.append(dict(collection=c,evaluation_set=subset,n_transactions=len(y),n_tokens=len(set(t)),n_calendar_blocks=len(set(b)),source_rows_sha256=ash(d['source_rows'][mask])))
   for j,n in enumerate(names):
    cond,g=('none','zero') if n=='zero' else n.split('__')
    metrics.append(dict(collection=c,evaluation_set=subset,condition=cond,group=g,n_transactions=len(y),n_tokens=len(set(t)),transaction_rmse=float(vals[0][j]),transaction_mae=float(vals[1][j]),transaction_r2=float(vals[2][j]),equal_token_rmse=float(vals[3][j])))
   contrasts=[]
   for cond in CONDITIONS:
    for g in ['image','early','late']:contrasts.append(('feature_comparison',cond+'__'+g,cond+'__metadata'))
    contrasts.append(('fusion_comparison',cond+'__early',cond+'__late'))
    for g in ['metadata','image','early','late']:contrasts.append(('zero_comparison',cond+'__'+g,'zero'))
   for cond in CONDITIONS[1:]:
    for g in ['metadata','image','early','late']:contrasts.append(('cleaning_comparison',cond+'__'+g,'base__'+g))
   for kind,cand,ref in contrasts:
    a,j=names.index(cand),names.index(ref)
    for method,metric,point,reps in [('token','transaction_rmse',vals[0],tr),('calendar14d','transaction_rmse',vals[0],br),('token','equal_token_rmse',vals[3],eq)]:
     delta=reps[:,a]-reps[:,j];lo,hi=np.quantile(delta,[.025,.975])
     cis.append(dict(collection=c,evaluation_set=subset,kind=kind,candidate=cand,reference=ref,resampling=method,metric=metric,n_transactions=len(y),n_tokens=len(set(t)),candidate_rmse=float(point[a]),reference_rmse=float(point[j]),delta_rmse=float(point[a]-point[j]),improvement_percent=float(100*(point[j]-point[a])/point[j]),ci95_low=float(lo),ci95_high=float(hi),iterations=2000,seed=SEED))
   np.savez_compressed(OUT/(c+'_'+subset+'_analysis_arrays.npz'),names=np.array(names),y_lrp=y,source_rows=d['source_rows'][mask],tokens=t,calendar_blocks=b,predictions=p,token_rmse=tr,equal_token_rmse=eq,calendar_rmse=br)
  for cond in CONDITIONS:
   for g in ['metadata','image','early','late']:
    r=next(x for x in selected if (x['collection'],x['condition'],x['group'])==(c,cond,g));base=next(x for x in selected if (x['collection'],x['condition'],x['group'])==(c,'base',g))
    changes.append(dict(collection=c,condition=cond,group=g,encoder=r['encoder'],family=r['family'],config=json.dumps(r['config'],sort_keys=True),metadata_family=r.get('metadata_family',''),metadata_config=json.dumps(r.get('metadata_config'),sort_keys=True),image_weight=r.get('image_weight',''),development_rmse=r['rmse'],encoder_changed=r['encoder']!=base['encoder'],family_changed=r['family']!=base['family'],config_changed=r['config']!=base['config'],metadata_branch_changed=(r.get('metadata_family')!=base.get('metadata_family') or r.get('metadata_config')!=base.get('metadata_config')) if g=='late' else False,weight_changed=r.get('image_weight')!=base.get('image_weight')))
 writecsv(OUT/'performance.csv',metrics);writecsv(OUT/'paired_comparisons.csv',cis);writecsv(OUT/'evaluation_cohorts.csv',counts);writecsv(OUT/'selection_changes.csv',changes);writecsv(OUT/'bootstrap_checks.csv',stats.CHECKS)
 savej(OUT/'analysis_summary.json',dict(completed_utc=now(),performance_rows=len(metrics),comparison_rows=len(cis),bootstrap_checks=len(stats.CHECKS),bootstrap_passed=all(x['passed'] for x in stats.CHECKS),evaluation_cohorts=counts,selection_changed=True))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
