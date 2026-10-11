"""Resumable fixed-specification implementation switch. Never writes legacy results."""
from shared import *
import numpy as np,joblib,time,gc,warnings,traceback,os
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
import psutil
early,meta,common,image_run=init_modules()
PROTOCOL=readj(OUT/"protocol_freeze.json")
REG=readj(early.REGISTRY_PATH)
MD=early.metadata_lookup()
(OUT/"fits").mkdir(exist_ok=True)
def predict(model,offset,X,family):
 if family=="PLS" and offset is not None:return np.asarray(model.predict(X-offset["x"])).reshape(-1)+offset["y"]
 return common.predict_model(model,X)
def metrics(y,p):
 return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),mae=float(np.mean(np.abs(y-p))),r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))
def get_ck(case,origin):
 g=case["group"];rep="TF_IDF" if g=="metadata" else case["encoder"]
 idx=common.GRIDS[case["family"]].index(case["config"])
 return RUNS[g]/"checkpoints"/case["collection"]/rep/origin/case["family"]/(str(idx)+".json")
def prepare(case,origin,end):
 c,g,enc=case["collection"],case["group"],case["encoder"]
 md=meta.windowed_development(c);md["collection"]=c
 train=np.flatnonzero(md["times"]<origin)
 final=origin=="2025-01-01"
 evalrows=lines(ROOT/"revision/target_pipeline_20260909"/f"{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl") if final else None
 valid=np.array([],dtype=int) if final else np.flatnonzero((md["times"]>=origin)&(md["times"]<end))
 ytest=np.array([r["y_log_relative_price"] for r in evalrows]) if final else md["y"][valid]
 testtokens=np.array([int(r["token_id"]) for r in evalrows]) if final else md["tokens"][valid]
 testrows=np.array([r["source_row"] for r in evalrows]) if final else md["rowids"][valid]
 params=common.fit_target_transform(md["y"][train]);z=common.target_forward(md["y"],params)
 if g=="metadata":
  pp=common.prepare(md,train,valid,"TF-IDF")
  uniq,first,inv,counts=np.unique(md["tokens"][train],return_index=True,return_inverse=True,return_counts=True)
  raw=pp["raw_train"][first].copy();scaled=pp["ztrain"][first].copy()
  if final:
   Xobj=np.array([[MD[c,int(t)][col] for col in md["columns"]] for t in testtokens],dtype=object)
   rawtest=common.dense(pp["encoding"].transform(Xobj));scaledtest=pp["scaler"].transform(rawtest)
  else:rawtest=pp["raw_valid"].copy();scaledtest=pp["zvalid"].copy()
  rank=pp["rank"];del pp
  res=dict(raw=raw,scaled=scaled,rawtest=rawtest,scaledtest=scaledtest,rank=rank,counts=counts,inverse=inv)
 else:
  # Reuse audited native matrices and fold-specific PCA; no fitting or writing PCA.
  info=REG[enc][c];ids=np.array(info["token_ids"],dtype=np.int64);mapping={int(t):i for i,t in enumerate(ids)}
  native=np.load(ROOT/info["matrix"],mmap_mode="r",allow_pickle=False)
  data=dict(collection=c,encoder=enc,y=md["y"],tokens=md["tokens"],rowids=md["rowids"],times=md["times"],
    X=np.array([[MD[c,int(t)][col] for col in md["columns"]] for t in ids],dtype=object),columns=md["columns"],image=native,
    feature_tokens=ids,index=np.array([mapping[int(t)] for t in md["tokens"]]),matrix_sha256=info["matrix_sha256"],native_dimension=info["dimensions"])
  im,proj=early.projected_image(data,train,origin)
  pp=early.prepare_early(data,im,train,valid,z) if g=="early" else image_run.prepare_image(data,im,train,valid,z)
  if final:
   ix=np.array([mapping[int(t)] for t in testtokens]);Xim=np.asarray(im[ix],dtype=np.float64)
   if g=="early":
    Xobj=np.array([[MD[c,int(t)][col] for col in md["columns"]] for t in testtokens],dtype=object)
    Xtest=early.cr.vision.transform(pp["state"],Xobj,Xim)
   else:Xtest=pp["scaler"].transform(Xim[:,pp["mask"]])
  else:Xtest=pp["z_valid"].copy()
  res=dict(scaled=pp["z_unique"],scaledtest=Xtest,rank=pp["rank"],counts=pp["counts"],inverse=pp["inverse"])
  del data,pp,im,native
 res.update(z=z[train],params=params,ytest=ytest,testtokens=testtokens,testrows=testrows,trainrows=md["rowids"][train],train_n=len(train),train_tokens=len(res["counts"]))
 del md;gc.collect()
 return res

def fit_impl(X,Xt,z,inv,counts,fam,conf,mode):
 means=np.bincount(inv,weights=z)/counts
 model=common.build_model(fam,conf,"development");offset=None
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter("always")
  if mode=="row":
   XX=np.array(X[inv],dtype=np.float64,order="F")
   if "copy_X" in model.get_params():model.set_params(copy_X=False)
   if "copy" in model.get_params():model.set_params(copy=False)
   model.fit(XX,z.copy());del XX
  elif fam=="PLS":
   mu=np.average(X,axis=0,weights=counts);ym=float(np.mean(z))
   xx=(X-mu)*np.sqrt(counts/2)[:,None];yy=(means-ym)*np.sqrt(counts/2)
   model.fit(np.concatenate([xx,-xx]),np.concatenate([yy,-yy]));offset={"x":mu,"y":ym}
  elif fam=="LinearSVR":
   model=early.cr.vision.CountedPrimalSVR(conf["C"],conf["epsilon"]).fit_counts(X,inv,z.copy())
  else:model.fit(X.copy(),means,sample_weight=counts)
 valid=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
 if fam=="PLS":valid=valid and len(model.n_iter_)==conf["components"]
 predz=predict(model,offset,Xt,fam)
 obj={}
 if fam=="LinearSVR":
  pz=predict(model,offset,X,fam)[inv];res=np.sign(pz-z)*np.maximum(np.abs(pz-z)-conf["epsilon"],0)
  beta=np.r_[np.asarray(model.coef_).reshape(-1),float(np.asarray(model.intercept_).reshape(-1)[0])]
  sums=np.bincount(inv,weights=res,minlength=len(X));grad=beta+2*conf["C"]*np.r_[X.T@sums,sums.sum()]
  obj=dict(primal_objective=float(.5*(beta@beta)+conf["C"]*(res@res)),gradient_norm=float(np.linalg.norm(grad)),objective_gap_upper_bound=float(grad@grad/2))
 status=dict(valid=bool(valid and np.isfinite(predz).all()),warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],n_iter=np.asarray(getattr(model,"n_iter_",[])).tolist(),objective=obj)
 del model;gc.collect()
 return predz,status

def run_one(case,origin,end,prepared):
 stem=case["id"]+"__"+origin
 dst=OUT/"fits"/(stem+".json");npz=dst.with_suffix(".npz")
 if dst.exists() and npz.exists():
  old=readj(dst)
  if old.get("completed"):return old
 c,g,fam=case["collection"],case["group"],case["family"]
 X=prepared["raw"] if g=="metadata" and fam in common.TREE_FAMILIES else prepared["scaled"]
 Xt=prepared["rawtest"] if g=="metadata" and fam in common.TREE_FAMILIES else prepared["scaledtest"]
 z=prepared["z"];inv=prepared["inverse"];counts=prepared["counts"]
 means=np.bincount(inv,weights=z)/counts
 mode=case["alternative_implementation"];conf=case["config"]
 started=time.perf_counter();status={};offset=None
 writej(OUT/"progress.json",dict(time=now(),case=case["id"],origin=origin,mode=mode,status="fitting",train_n=len(z),dimension=X.shape[1],available_ram_gib=psutil.virtual_memory().available/2**30))
 print(now(),stem,mode,"start",len(z),X.shape[1],flush=True)
 if origin=="2025-01-01":
  # Confirm exact archived original inference under this common preprocessing.
  bundle=joblib.load(modelpath(case))
  archived_original=common.target_inverse(predict(bundle["model"],bundle.get("offset"),Xt,fam),prepared["params"])
  with np.load(predpath(case),allow_pickle=False) as zz:
   assert np.array_equal(zz["source_rows"],prepared["testrows"])
   assert np.array_equal(zz["y_lrp"],prepared["ytest"])
   original=zz["predictions_lrp"].copy()
  archive_gap=float(np.max(np.abs(original-archived_original)))
  if archive_gap>1e-7:raise RuntimeError(f"Archived inference gap {archive_gap}: {stem}")
  del bundle,archived_original
 else:
  old=get_ck(case,origin);oldrow=readj(old);original=None;archive_gap=None
  assert oldrow["valid"] and oldrow["config"]==conf
  assert oldrow["training_row_sha256"]==hashlib.sha256(np.ascontiguousarray(prepared["trainrows"]).tobytes()).hexdigest()
  if old.with_suffix(".npy").exists():original=np.load(old.with_suffix(".npy"),allow_pickle=False)
 before_target=hashlib.sha256(z.tobytes()).hexdigest()
 control_z,control_status=fit_impl(X,Xt,z,inv,counts,fam,conf,case["original_implementation"])
 control=common.target_inverse(control_z,prepared["params"])
 predz,alt_status=fit_impl(X,Xt,z,inv,counts,fam,conf,mode)
 pred=common.target_inverse(predz,prepared["params"])
 assert hashlib.sha256(z.tobytes()).hexdigest()==before_target,"Shared target mutated"
 valid=alt_status["valid"] and control_status["valid"]
 if not np.isfinite(pred).all() or not np.isfinite(control).all():raise RuntimeError("Nonfinite predictions")
 s=metrics(prepared["ytest"],pred)
 orig_rmse=metrics(prepared["ytest"],original)["rmse"] if original is not None else float(oldrow["transaction_rmse"])
 obj=alt_status["objective"]
 report=dict(case_id=case["id"],collection=c,group=g,encoder=case["encoder"],family=fam,role=case["role"],config=conf,origin=origin,end=end,
  alternative_implementation=mode,original_implementation=case["original_implementation"],training_n=len(z),training_tokens=len(counts),test_n=len(pred),dimension=X.shape[1],
  target_parameters=prepared["params"],alternative_metrics=s,original_rmse=orig_rmse,rmse_gap=s["rmse"]-orig_rmse,
  prediction_max_abs_gap=None if original is None else float(np.max(np.abs(pred-original))),
  prediction_rmse_gap=None if original is None else float(np.sqrt(np.mean((pred-original)**2))),
  archived_inference_max_abs_gap=archive_gap,valid=valid,warnings=alt_status["warnings"],
  n_iter=alt_status["n_iter"],objective=obj,control_status=control_status,control_metrics=metrics(prepared["ytest"],control),
  control_archived_max_abs_gap=None if original is None else float(np.max(np.abs(control-original))),
  controlled_prediction_max_abs_gap=float(np.max(np.abs(pred-control))),controlled_prediction_rmse_gap=float(np.sqrt(np.mean((pred-control)**2))),seconds=time.perf_counter()-started,completed=True)
 np.savez_compressed(npz,source_rows=prepared["testrows"],tokens=prepared["testtokens"].astype(str),y_lrp=prepared["ytest"],alternative_predictions=pred,control_predictions=control,
   original_predictions=original if original is not None else np.empty(0))
 writej(dst,report);gc.collect()
 print(now(),stem,"done",round(report["seconds"],2),"s gap",report["rmse_gap"],"valid",valid,flush=True)
 return report
def main():
 import sklearn,scipy,platform
 writej(OUT/"runtime.json",dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,xgboost=common.xgboost.__version__,lightgbm=common.lightgbm.__version__,thread_limit=3))
 records=[];last=None;prep=None
 cases=PROTOCOL["cases"]
 # Stable grouping permits reuse of identical preprocessing across learners.
 jobs=sorted([(x,o,e) for x in cases for o,e in ORIGINS],key=lambda t:(t[0]["collection"],t[0]["group"],t[0]["encoder"],t[1],FAMILIES.index(t[0]["family"])))
 with threadpool_limits(limits=3):
  for case,origin,end in jobs:
   stem=case["id"]+"__"+origin;f=OUT/"fits"/(stem+".json")
   if f.exists() and readj(f).get("completed") and f.with_suffix(".npz").exists():records.append(readj(f));continue
   key=(case["collection"],case["group"],case["encoder"],origin)
   if key!=last:
    del prep;gc.collect();prep=prepare(case,origin,end);last=key
   records.append(run_one(case,origin,end,prep))
 writej(OUT/"implementation_fit_summary.json",dict(completed_utc=now(),planned=len(jobs),completed=len(records),invalid_fits=[r["case_id"]+" "+r["origin"] for r in records if not r["valid"]],records=records))
 writej(OUT/"progress.json",dict(time=now(),status="fits_complete",completed=len(records)))
if __name__=="__main__":
 try:main()
 except Exception:
  writej(OUT/"worker_error.json",dict(time=now(),error=traceback.format_exc()));raise
