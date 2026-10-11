"""Independent checks from stored transaction targets and row predictions."""
from engine import *
CHECKS=[]
def check(name,ok,detail=''):
 CHECKS.append(dict(check=name,passed=bool(ok),detail=str(detail)))
 if len(CHECKS)%1000==0:savej(OUT/'progress.json',dict(updated_utc=now(),stage='independent_verification',checks=len(CHECKS),completed_checkpoints=15930,expected_checkpoints=15930))
 if not ok:raise AssertionError(name+': '+str(detail))
def close(name,a,b,atol=1e-10):check(name,np.allclose(a,b,rtol=0,atol=atol),float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
def main():
 protocol=readj(OUT/'protocol_freeze.json')
 for r in protocol['inputs']:check('input '+r['path'],sha(ROOT/r['path'])==r['sha256'])
 initial=readj(OUT/'selection_freeze.json');final=readj(OUT/'final_selection_freeze.json');started=readj(OUT/'evaluation_started.json')
 check('initial freeze label flag',initial['evaluation_labels_loaded'] is False)
 check('final freeze label flag',final['evaluation_labels_loaded'] is False)
 check('initial then final then evaluation',initial['created_utc']<=final['created_utc']<=started['created_utc'])
 check('initial freeze hash',sha(OUT/'selection_freeze.json')==final['initial_selection_sha256'])
 check('final freeze hash',sha(OUT/'final_selection_freeze.json')==started['final_selection_sha256'])
 source_manifest=readj(OUT/'execution_sources.json')
 for r in source_manifest['sources']:check('execution source '+r['path'],sha(ROOT/r['path'])==r['sha256'])
 dc={(cond,c):cohort(c,cond) for cond in CONDITIONS for c in ['BAYC','MAYC']}
 prepared=list((OUT/'prepared').rglob('manifest.json'))
 for p in prepared:
  m=readj(p);d=dc[m['condition'],m['collection']];idx=np.flatnonzero(d['times']<m['origin']);v=np.flatnonzero((d['times']>=m['origin'])&(d['times']<m['end']))
  prefix=str(p.relative_to(OUT));folder=p.parent
  check(prefix+' training ids',np.array_equal(d['rowids'][idx],np.load(folder/'train_rows.npy')))
  check(prefix+' training token order',np.array_equal(d['tokens'][idx],np.load(folder/'train_tokens.npy')))
  check(prefix+' validation ids',np.array_equal(d['rowids'][v],np.load(folder/'valid_rows.npy')))
  check(prefix+' temporal boundary',all(t<'2025-01-01' and t<m['origin'] for t in d['times'][idx]))
  check(prefix+' disjoint rows',not set(d['rowids'][idx])&set(d['rowids'][v]))
  inv=np.load(folder/'inverse.npy');counts=np.load(folder/'counts.npy');ut=np.load(folder/'unique_tokens.npy')
  check(prefix+' inverse expansion',np.array_equal(ut[inv],d['tokens'][idx]))
  check(prefix+' count weights',np.array_equal(np.bincount(inv),counts))
  params=C.fit_target_transform(d['y'][idx]);check(prefix+' transform parameters',params==m['target_parameters'])
  close(prefix+' transformed target',C.target_forward(d['y'][idx],params),np.load(folder/'z.npy'))
  if m['origin']!='2025-01-01':check(prefix+' validation target',np.array_equal(d['y'][v],np.load(folder/'valid_y.npy')))
  if m['group']!='image':check(prefix+' unique NFT IDF documents',m['idf_documents']==len(np.unique(d['tokens'][idx])))
  if m['projection']:
   pm=readj(OUT/m['projection']/'manifest.json');check(prefix+' PCA training row hash',pm['identity']['training_row_sha256']==ash(d['rowids'][idx]));check(prefix+' PCA train-only',pm['fitted_only_on_training_images'] and pm['identity']['training_unique_tokens']==len(ut));check(prefix+' PCA state hash',sha(OUT/m['projection']/'pca.joblib')==pm['pca_sha256']);check(prefix+' PCA matrix hash',sha(OUT/m['projection']/'embeddings_1024.npy')==pm['embeddings_sha256'])
  for name,digest in m['files'].items():check(prefix+' file '+name,sha(folder/name)==digest)
 candidates=csvread(OUT/'candidate_development_scores.csv');allck=list((OUT/'checkpoints').rglob('*.json'))
 check('checkpoint total',len(allck)==15930);check('candidate total',len(candidates)==5310)
 invalid=[]
 for r in candidates:
  cond,c,g,enc,fam=r['condition'],r['collection'],r['group'],r['encoder'],r['family'];num=int(r['config_index']);d=dc[cond,c];ss=0.;n=0;valid=True
  for origin,end in FOLDS:
   path=ckpath(cond,c,g,enc,origin,fam,num);ck=readj(path);valid=valid and ck['valid'];ix=np.flatnonzero((d['times']>=origin)&(d['times']<end));n+=len(ix)
   check(str(path.relative_to(OUT))+' prefreeze',ck['completed_utc']<=initial['created_utc'])
   check(str(path.relative_to(OUT))+' settings',ck['config']==C.GRIDS[fam][num] and ck['n']==len(ix))
   check(str(path.relative_to(OUT))+' prepared link',ck['prepared_sha256']==sha(pathprep(cond,c,g,enc,origin)/'manifest.json'))
   if ck['valid']:
    p=np.load(path.with_suffix('.npy'));check(str(path.relative_to(OUT))+' hash',sha(path.with_suffix('.npy'))==ck['prediction_sha256']);s=float(np.sum((d['y'][ix]-p)**2));ss+=s;close(str(path.relative_to(OUT))+' SSE',s,ck['sse'],1e-9)
   else:invalid.append(dict(path=str(path.relative_to(OUT)),status=json.dumps(ck['status'])))
  check('pooled validity '+str((cond,c,g,enc,fam,num)),valid==(r['valid']=='True'))
  if valid:close('pooled RMSE '+str((cond,c,g,enc,fam,num)),np.sqrt(ss/n),float(r['rmse']))
 # All family choices must be the first valid configuration in the archived ordering rule.
 families=csvread(OUT/'family_selections.csv')
 for r in families:
  rr=[dict(x,config=json.loads(x['config']),valid=x['valid']=='True',mse_lrp=None if not x['mse_lrp'] else float(x['mse_lrp'])) for x in candidates if all(x[k]==r[k] for k in ['condition','collection','group','encoder','family']) and '__'.join(x[k] for k in ['condition','collection','group','encoder','family','config_index']) not in final['failed']]
  chosen=C.ordered_choices(r['family'],rr)[0];check('family selection '+str((r['condition'],r['collection'],r['group'],r['encoder'],r['family'])),chosen['config']==json.loads(r['config']))
 check('family selections total',len(families)==720)
 late=csvread(OUT/'late_weight_development_scores.csv');check('late weight grid total',len(late)==3696)
 for cond in CONDITIONS:
  for c in ['BAYC','MAYC']:
   for g in ['metadata','image','early']:
    rr=[r for r in families if (r['condition'],r['collection'],r['group'])==(cond,c,g)]
    winner=min(rr,key=lambda x:(float(x['mse_lrp']),(['TF_IDF']+ENCODERS).index(x['encoder']),FAMILIES.index(x['family'])))
    actual=next(r for r in final['representatives'] if (r['condition'],r['collection'],r['group'])==(cond,c,g))
    check('global selected '+str((cond,c,g)),actual['encoder']==winner['encoder'] and actual['family']==winner['family'] and actual['config']==json.loads(winner['config']))
   rr=[r for r in late if (r['condition'],r['collection'])==(cond,c)];winner=min(rr,key=lambda x:(float(x['mse_lrp']),ENCODERS.index(x['encoder']),FAMILIES.index(x['family']),float(x['image_weight'])))
   actual=next(r for r in final['representatives'] if (r['condition'],r['collection'],r['group'])==(cond,c,'late'))
   check('late selected '+str((cond,c)),actual['encoder']==winner['encoder'] and actual['family']==winner['family'] and actual['image_weight']==float(winner['image_weight']))
   # Independent blend calculation from the archived validation row vectors for every image branch/weight.
   mr=next(r for r in final['representatives'] if (r['condition'],r['collection'],r['group'])==(cond,c,'metadata'))
   def vp(g,enc,fam,idx):return np.concatenate([np.load(ckpath(cond,c,g,enc,o,fam,int(idx)).with_suffix('.npy')) for o,e in FOLDS])
   mp=vp('metadata','TF_IDF',mr['family'],mr['config_index']);d=dc[cond,c];yv=np.concatenate([d['y'][(d['times']>=o)&(d['times']<e)] for o,e in FOLDS])
   cache={}
   for r in rr:
    k=(r['encoder'],r['family'],r['config_index'])
    if k not in cache:cache[k]=vp('image',*k)
    w=float(r['image_weight']);value=float(np.mean((yv-((1-w)*mp+w*cache[k]))**2));close('late development '+str((cond,c,*k,w)),value,float(r['mse_lrp']))
 for path in (OUT/'final_models').glob('*/manifest.json'):
  r=readj(path);check('final timing '+path.parent.name,initial['created_utc']<=r['completed_utc']<=started['created_utc'])
  if r['valid']:
   check('model hash '+path.parent.name,sha(path.parent/'model.joblib')==r['model_sha256'])
   check('token predictions hash '+path.parent.name,sha(path.parent/'token_predictions.npz')==r['predictions_sha256'])
   bundle=joblib.load(path.parent/'model.joblib');man,a=loadprep(OUT/r['prepared'],r['candidate']['family'])
   with threadpool_limits(limits=3):p=C.target_inverse(predict(bundle['model'],bundle['offset'],a['Xvalid']),man['target_parameters'])
   with np.load(path.parent/'token_predictions.npz') as z:close('reloaded model inference '+path.parent.name,p,z['predictions'])
   del bundle,a;gc.collect()
 perf=csvread(OUT/'performance.csv');cis=csvread(OUT/'paired_comparisons.csv')
 for c in ['BAYC','MAYC']:
  d=cohort(c,'base','evaluation')
  with np.load(OUT/(c+'_evaluation_predictions.npz'),allow_pickle=False) as z:
   check(c+' evaluation rows',np.array_equal(d['rowids'],z['source_rows']));check(c+' evaluation targets',np.array_equal(d['y'],z['y_lrp']));check(c+' strict eval mask',np.array_equal(d['strict'],z['strict']))
   def aligned(k):
    with np.load(OUT/'final_models'/k/'token_predictions.npz') as t:lookup=dict(zip(t['tokens'],t['predictions']))
    return np.array([lookup[t] for t in d['tokens']])
   for r in [r for r in final['representatives'] if r['collection']==c]:
    k='__'.join(str(r[x]) for x in ['condition','collection','group','encoder','family','config_index'])
    expected=aligned(k) if r['group']!='late' else (1-r['image_weight'])*aligned(r['metadata_key'])+r['image_weight']*aligned(r['image_key'])
    close('final evaluation alignment '+r['condition']+' '+c+' '+r['group'],z[r['condition']+'__'+r['group']],expected)

  for sub in ['base','strict']:
   with np.load(OUT/(c+'_'+sub+'_analysis_arrays.npz'),allow_pickle=False) as z:
    names=list(z['names']);y=z['y_lrp'];p=z['predictions'];tokens=z['tokens'];unique=np.unique(tokens)
    for j,name in enumerate(names):
     cond,g=('none','zero') if name=='zero' else name.split('__');r=next(r for r in perf if (r['collection'],r['evaluation_set'],r['condition'],r['group'])==(c,sub,cond,g))
     sq=(y-p[:,j])**2;expected=[np.sqrt(sq.mean()),np.abs(y-p[:,j]).mean(),1-sq.sum()/((y-y.mean())**2).sum(),np.sqrt(np.mean([sq[tokens==t].mean() for t in unique]))]
     close('reported metrics '+str((c,sub,name)),expected,[float(r[k]) for k in ['transaction_rmse','transaction_mae','transaction_r2','equal_token_rmse']])
    for r in [r for r in cis if (r['collection'],r['evaluation_set'])==(c,sub)]:
     a,b=names.index(r['candidate']),names.index(r['reference']);reps=z['calendar_rmse'] if r['resampling']=='calendar14d' else z['equal_token_rmse'] if r['metric']=='equal_token_rmse' else z['token_rmse']
     close('reported interval '+str((c,sub,r['candidate'],r['reference'],r['resampling'],r['metric'])),np.quantile(reps[:,a]-reps[:,b],[.025,.975]),[float(r['ci95_low']),float(r['ci95_high'])])
 bootstrap=csvread(OUT/'bootstrap_checks.csv');check('independent bootstrap row checks',all(r['passed']=='True' for r in bootstrap))
 writecsv(OUT/'invalid_development_fits.csv',invalid)
 writecsv(OUT/'verification_checks.csv',CHECKS)
 savej(OUT/'verification_summary.json',dict(completed_utc=now(),passed=True,checks=len(CHECKS),failed=0,prepared_groups=len(prepared),development_fits=len(allck),invalid_development_fits=len(invalid),valid_development_fits=len(allck)-len(invalid),family_selections=len(families),late_weight_scores=len(late),selected_representatives=len(final['representatives']),final_fallbacks=final['failed'],performance_rows=len(perf),paired_comparison_rows=len(cis)))
 print('VERIFIED',len(CHECKS),'checks',flush=True)
if __name__=='__main__':
 try:
  with threadpool_limits(limits=1):main()
 except Exception:
  writecsv(OUT/'verification_checks.csv',CHECKS);savej(OUT/'verification_error.json',dict(time=now(),error=traceback.format_exc()));raise
