"""Strict-clean full reselection. All output stays under this analysis folder."""
from shared import *
import os,time,gc,warnings,traceback
from contextlib import contextmanager
@contextmanager
def pca_lock():
 lock=OUT/"pca.lock"
 while True:
  try:
   with lock.open("x") as f:f.write(str(os.getpid()))
   break
  except FileExistsError:time.sleep(2)
 try:yield
 finally:lock.unlink()
import numpy as np,joblib
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
E,M,C,I=init_modules()
CONDITIONS=['base','strict','strict_no_wei']
ENCODERS=list(E.ENCODERS)
FOLDS=ORIGINS[:3]
SEED=20260908
REG=readj(E.REGISTRY_PATH)

def savej(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf8');temp.replace(p)
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def cohort(c,cond,phase='development'):
 name=f'{c.lower()}_development_targets.jsonl' if phase=='development' else f'{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl'
 rows=lines(E.cr.base.TARGET/name)
 rows=[r for r in rows if ('2022-01-01'<=r['time']<'2025-01-01') or phase!='development']
 lookup={int(r['source_row']):r for r in rows}
 elig={}
 with (E.cr.base.OLD/f'{c.lower()}_trade_eligibility.jsonl').open(encoding='utf-8-sig') as f:
  for line in f:
   r=json.loads(line);rid=int(r['source_row'])
   if rid in lookup:
    t=lookup[rid];assert int(r['token_id'])==int(t['token_id']) and r['time']==t['time'] and r['base_clean']
    assert bool(r['strict_clean'])==bool(r['base_clean'] and not r['observed_multi_nft_tx'] and not r['reverse_pair_24h'])
    elig[rid]=bool(r['strict_clean'])
 assert set(elig)==set(lookup)
 rows=[r for r in rows if (cond=='base' or elig[int(r['source_row'])]) and (cond!='strict_no_wei' or r['price_eth']>1e-18*(1+1e-9))]
 return dict(y=np.array([r['y_log_relative_price'] for r in rows]),tokens=np.array([int(r['token_id']) for r in rows],dtype=np.int64),rowids=np.array([int(r['source_row']) for r in rows],dtype=np.int64),times=np.array([r['time'] for r in rows],dtype=object),prices=np.array([r['price_eth'] for r in rows]),strict=np.array([elig[int(r['source_row'])] for r in rows],dtype=bool))
def pathprep(cond,c,g,enc,origin):return OUT/'prepared'/cond/c/g/enc/origin

def metadata_features(objects,unique,counts,valid_index,all_objects):
 onehot=OneHotEncoder(handle_unknown='ignore',drop=None,sparse_output=True,dtype=np.float64)
 binary=onehot.fit_transform(objects[unique]);tfidf=TfidfTransformer(**C.TFIDF_SETTINGS).fit(binary)
 raw0=C.dense(tfidf.transform(binary));flt=VarianceThreshold(0).fit(raw0);raw=flt.transform(raw0)
 encoding=Pipeline([('onehot',onehot),('tfidf',tfidf),('constant_filter',flt),('dense',FunctionTransformer(C.dense,validate=False))])
 scaler=StandardScaler().fit(raw,sample_weight=counts);scaled=scaler.transform(raw)
 rv=encoding.transform(all_objects[valid_index]) if len(valid_index) else np.empty((0,raw.shape[1]))
 sv=scaler.transform(rv) if len(rv) else rv
 np.testing.assert_allclose(tfidf.idf_,np.log((1+len(unique))/(1+np.asarray(binary.sum(axis=0)).reshape(-1)))+1,rtol=1e-14,atol=1e-14)
 return raw,scaled,rv,sv,encoding,scaler

def prepare_task(cond,c,enc,origin,end,groups=None):
 final=origin=='2025-01-01'
 groups=groups or (['metadata'] if enc=='TF_IDF' else ['image','early'])
 if all((pathprep(cond,c,g,enc,origin)/'manifest.json').exists() for g in groups):return
 d=cohort(c,cond);train=np.flatnonzero(d['times']<origin);valid=np.flatnonzero((d['times']>=origin)&(d['times']<end))
 assert len(train)>0 and (len(valid)>0 or final)
 params=C.fit_target_transform(d['y'][train]);z=C.target_forward(d['y'][train],params)
 md=E.metadata_lookup();columns=E.cr.base.FEATURES+(['generation'] if c=='MAYC' else [])
 projection=None
 if enc=='TF_IDF':
  ids=np.array(sorted(t for (cc,t) in md if cc==c),dtype=np.int64);native=None
 else:
  info=REG[enc][c];ids=np.array(info['token_ids'],dtype=np.int64);native=np.load(ROOT/info['matrix'],mmap_mode='r')
 mapping={int(t):i for i,t in enumerate(ids)};index=np.array([mapping[int(t)] for t in d['tokens']])
 objects=np.array([[md[c,int(t)][col] for col in columns] for t in ids],dtype=object)
 unique,inv,counts=np.unique(index[train],return_inverse=True,return_counts=True)
 assert np.array_equal(ids[unique][inv],d['tokens'][train])
 # Final preparation is label-free: features for every registered token; evaluation file is not opened.
 vi=np.arange(len(ids)) if final else index[valid]
 meta=None
 if 'metadata' in groups or 'early' in groups:meta=metadata_features(objects,unique,counts,vi,objects)
 if enc!='TF_IDF':
  image=native
  if enc in E.REDUCED:
   pdata=dict(d,collection=c,encoder=enc,index=index,feature_tokens=ids,image=native,_native_sha256=info['matrix_sha256'],_collection=c,_encoder=enc)
   folder=OUT/'projections'/cond/c/enc/origin
   with pca_lock():
    _,image=I.pr.fit_projection(pdata,train,origin,folder,components=1024)
   projection=str(folder.relative_to(OUT));del pdata
  raw_image=np.asarray(image[unique],dtype=np.float64);mask=np.ptp(raw_image,axis=0)>0
  ims=StandardScaler().fit(raw_image[:,mask],sample_weight=counts);iz=ims.transform(raw_image[:,mask]);iv=ims.transform(np.asarray(image[vi][:,mask],dtype=np.float64))
  del raw_image
 for g in groups:
  dst=pathprep(cond,c,g,enc,origin);dst.mkdir(parents=True,exist_ok=True)
  if (dst/'manifest.json').exists():continue
  state=dict(group=g,columns=columns,projection=projection)
  if g=='metadata':
   raw,X,rawv,Xv,encoding,scaler=meta;state.update(encoding=encoding,scaler=scaler)
   np.save(dst/'raw.npy',raw);np.save(dst/'raw_valid.npy',rawv)
  elif g=='image':X,Xv=iz,iv;state.update(image_mask=mask,image_scaler=ims)
  else:
   raw,mz,rawv,mv,encoding,scaler=meta;X=np.concatenate([mz,iz],axis=1);Xv=np.concatenate([mv,iv],axis=1)
   state.update(encoding=encoding,scaler=scaler,image_mask=mask,image_scaler=ims)
  sample=X[np.linspace(0,len(X)-1,min(256,len(X)),dtype=int)]
  rank=int(np.linalg.matrix_rank(sample-sample.mean(axis=0)))
  values=dict(X=X,Xvalid=Xv,z=z,inverse=inv,counts=counts,train_rows=d['rowids'][train],train_tokens=d['tokens'][train],unique_tokens=ids[unique],predict_tokens=ids if final else d['tokens'][valid],valid_rows=np.empty(0,dtype=np.int64) if final else d['rowids'][valid],valid_y=np.empty(0) if final else d['y'][valid])
  for name,a in values.items():np.save(dst/(name+'.npy'),a,allow_pickle=False)
  joblib.dump(state,dst/'state.joblib',compress=3)
  files={p.name:sha(p) for p in dst.iterdir() if p.is_file()}
  savej(dst/'manifest.json',dict(created_utc=now(),condition=cond,collection=c,group=g,encoder=enc,origin=origin,end=end,training_n=len(train),training_tokens=len(unique),validation_n=len(valid),dimension=X.shape[1],rank=rank,target_parameters=params,training_min=str(d['times'][train].min()),training_max=str(d['times'][train].max()),validation_min=None if final else str(d['times'][valid].min()),validation_max=None if final else str(d['times'][valid].max()),training_row_sha256=ash(d['rowids'][train]),validation_row_sha256=ash(values['valid_rows']),training_target_sha256=ash(d['y'][train]),idf_documents=len(unique) if g!='image' else None,projection=projection,files=files))
 del d,md,objects;gc.collect()

def loadprep(folder,family):
 man=readj(folder/'manifest.json');a={n:np.load(folder/(n+'.npy'),mmap_mode='r') for n in ['X','Xvalid','z','inverse','counts','valid_y']}
 if man['group']=='metadata' and family in C.TREE_FAMILIES:
  a['X']=np.load(folder/'raw.npy',mmap_mode='r');a['Xvalid']=np.load(folder/'raw_valid.npy',mmap_mode='r')
 return man,a

def fit(a,family,config,rank):
 start=time.perf_counter();model=offset=pred=None
 try:
  if family=='PLS' and config['components']>rank:raise ValueError('PLS components exceed deterministic rank lower bound')
  X=a['X'];z=a['z'];counts=a['counts'];inv=a['inverse'];means=np.bincount(inv,weights=z)/counts
  model=C.build_model(family,config,'development')
  if family=='ElasticNet':model.set_params(precompute=True)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   if family=='PLS':
    mu=np.average(X,axis=0,weights=counts);ym=float(z.mean());xx=(X-mu)*np.sqrt(counts/2)[:,None];yy=(means-ym)*np.sqrt(counts/2)
    model.fit(np.concatenate([xx,-xx]),np.concatenate([yy,-yy]));offset=dict(x=mu,y=ym)
   elif family=='LinearSVR':model=E.cr.vision.CountedPrimalSVR(config['C'],config['epsilon']).fit_counts(X,inv,z.copy())
   else:model.fit(X.copy(),means,sample_weight=counts)
   pred=predict(model,offset,a['Xvalid'])
  converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
  if family=='PLS':converged=converged and len(model.n_iter_)==config['components'] and np.isfinite(model.coef_).all()
  status=dict(valid=bool(converged and np.isfinite(pred).all()),warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],n_iter=np.asarray(getattr(model,'n_iter_',[])).tolist())
  if family=='LinearSVR':status.update(gradient_norm=float(model.gradient_norm_),objective_gap_upper_bound=float(model.objective_gap_upper_bound_))
 except (MemoryError,OSError):raise
 except Exception as e:status=dict(valid=False,error=repr(e))
 status['seconds']=time.perf_counter()-start
 return model,offset,pred,status

def predict(model,offset,X):return C.predict_model(model,X) if offset is None else np.asarray(model.predict(X-offset['x'])).reshape(-1)+offset['y']
def ckpath(cond,c,g,enc,origin,fam,num):return OUT/'checkpoints'/cond/c/g/enc/origin/fam/(str(num)+'.json')
def fit_task(job):
 cond,c,enc,origin,end=job;started=time.perf_counter()
 with threadpool_limits(limits=3):
  prepare_task(*job)
  for g in (['metadata'] if enc=='TF_IDF' else ['image','early']):
   folder=pathprep(cond,c,g,enc,origin)
   for family in FAMILIES:
    man,a=loadprep(folder,family)
    for num,config in enumerate(C.GRIDS[family]):
     dst=ckpath(cond,c,g,enc,origin,family,num)
     if dst.exists():
      r=readj(dst);assert r['config']==config and r['prepared_sha256']==sha(folder/'manifest.json')
      if r['valid']:assert sha(dst.with_suffix('.npy'))==r['prediction_sha256']
      continue
     before=ash(a['z']);model,offset,predz,status=fit(a,family,config,man['rank']);assert before==ash(a['z'])
     pred=None if predz is None else C.target_inverse(predz,man['target_parameters'])
     valid=status['valid'] and pred is not None and np.isfinite(pred).all()
     dst.parent.mkdir(parents=True,exist_ok=True)
     sse=None;digest=None
     if valid:
      np.save(dst.with_suffix('.npy'),pred);digest=sha(dst.with_suffix('.npy'));sse=float(np.sum((a['valid_y']-pred)**2))
     savej(dst,dict(condition=cond,collection=c,group=g,encoder=enc,origin=origin,end=end,family=family,config_index=num,config=config,valid=bool(valid),status=status,n=len(a['valid_y']),sse=sse,rmse=None if sse is None else float(np.sqrt(sse/len(pred))),prediction_sha256=digest,prepared_sha256=sha(folder/'manifest.json'),completed_utc=now()))
     print(now(),cond,c,g,enc,origin,family,num,'valid',valid,'s',round(status['seconds'],3),flush=True)
     del model,predz,pred;gc.collect()
    del a
 savej(OUT/'jobs'/('__'.join(job[:4])+'.json'),dict(job=job,completed_utc=now(),seconds=time.perf_counter()-started))
 return dict(job=job,seconds=time.perf_counter()-started)
