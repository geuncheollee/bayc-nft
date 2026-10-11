"""Paired matched-family and fixed-configuration implementation comparisons."""
from shared import *
import numpy as np, argparse
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/"rerevision/seen_unseen_20261010/code"))
import run_seen_unseen as stats
P=readj(OUT/"protocol_freeze.json")
def load(case,alternative=False):
 p=OUT/"fits"/(case["id"]+"__2025-01-01.npz") if alternative else predpath(case)
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main(original_only=False):
 metrics=[];cis=[];dev=[];checks=[]
 for c in ["BAYC","MAYC"]:
  cases=[r for r in P["cases"] if r["collection"]==c]
  target=lines(ROOT/"revision/target_pipeline_20260909"/f"{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
  y=np.array([r["y_log_relative_price"] for r in target]);tokens=np.array([str(r["token_id"]) for r in target]);rows=np.array([r["source_row"] for r in target])
  blocks=np.array([(datetime.fromisoformat(r["time"])-datetime(2025,1,1,tzinfo=timezone.utc)).days//14 for r in target])
  names=[];preds=[];lookup={}
  for case in cases:
   a=load(case)
   for key,val in [("y_lrp",y),("source_rows",rows),("tokens",tokens)]:assert np.array_equal(a[key].astype(str) if key=="tokens" else a[key],val)
   lookup[case["id"],"original"]=len(names);names.append(case["id"]+"|original");preds.append(a["predictions_lrp"])
   if not original_only:
    b=load(case,True)
    for key,val in [("y_lrp",y),("source_rows",rows),("tokens",tokens)]:assert np.array_equal(b[key].astype(str) if key=="tokens" else b[key],val)
    lookup[case["id"],"alternative"]=len(names);names.append(case["id"]+"|alternative");preds.append(b["alternative_predictions"])
    lookup[case["id"],"control"]=len(names);names.append(case["id"]+"|control");preds.append(b["control_predictions"])
  def col(case,mode):
   return lookup[case["id"],"original" if mode=="original" else "control" if mode==case["original_implementation"] else "alternative"]
  contrasts=[]
  modes=["original"] if original_only else ["original","row","counted"]
  for fam in FAMILIES:
   ref=next(r for r in cases if r["role"]=="matched" and r["group"]=="metadata" and r["family"]==fam)
   for g in ["image","early"]:
    cand=next(r for r in cases if r["role"]=="matched" and r["group"]==g and r["family"]==fam)
    for mode in modes:contrasts.append(dict(kind="matched",family=fam,group=g,mode=mode,candidate=cand["id"],reference=ref["id"],ci=col(cand,mode),ri=col(ref,mode)))
  if not original_only:
   for case in cases:contrasts.append(dict(kind="implementation",family=case["family"],group=case["group"],mode="counted-minus-row",candidate=case["id"]+"|counted",reference=case["id"]+"|row",ci=col(case,"counted"),ri=col(case,"row")))
  # Frozen global representatives: late retains original blend weight and branches.
  selected=csvread(AUDIT/"selected_models.csv")
  def chosen(g):
   s=next(x for x in selected if x["collection"]==c and x["group"]==g)
   return next(x for x in cases if x["group"]==g and x["family"]==s["family"] and (g=="metadata" or x["encoder"]==s["encoder"]))
  m=chosen("metadata");e=chosen("early");l=next(x for x in P["late_representatives"] if x["collection"]==c)
  lm=next(x for x in cases if x["group"]=="metadata" and x["family"]==l["metadata_family"])
  li=next(x for x in cases if x["group"]=="image" and x["encoder"]==l["encoder"] and x["family"]==l["image_family"])
  w=float(l["image_weight"])
  for mode in modes:
   late=(1-w)*preds[col(lm,mode)]+w*preds[col(li,mode)]
   j=len(names);names.append(c+"_late|"+mode);preds.append(late)
   if mode=="original":
    path=RUNS["late"]/"predictions"/f"{c}_{l['encoder']}_{l['image_family']}.npz"
    with np.load(path,allow_pickle=False) as z:assert np.allclose(z["predictions_lrp"],late,rtol=0,atol=1e-10)
   for g,k,candidate in [("early",col(e,mode),e["id"]),("late",j,c+"_late")]:
    contrasts.append(dict(kind="global_representative",family="different_selected_families",group=g,mode=mode,candidate=candidate,reference=m["id"],ci=k,ri=col(m,mode)))
  pred=np.column_stack(preds);v=stats.metrics(y,pred,tokens)
  tr,eq=stats.bootstrap(y,pred,tokens,True,c+"token");br,_=stats.bootstrap(y,pred,blocks,False,c+"block")
  for j,name in enumerate(names):
   metrics.append(dict(collection=c,model=name,n_transactions=len(y),n_tokens=len(set(tokens)),transaction_rmse=float(v[0][j]),transaction_mae=float(v[1][j]),transaction_r2=float(v[2][j]),equal_token_rmse=float(v[3][j])))
  for con in contrasts:
   for scheme,metric,point,reps in [("token","transaction_rmse",v[0],tr),("calendar14d","transaction_rmse",v[0],br),("token","equal_token_rmse",v[3],eq)]:
    a,b=con["ci"],con["ri"];dd=reps[:,a]-reps[:,b];lo,hi=np.quantile(dd,[.025,.975])
    cis.append(dict(collection=c,**{k:x for k,x in con.items() if k not in ["ci","ri"]},resampling=scheme,metric=metric,candidate_rmse=float(point[a]),reference_rmse=float(point[b]),delta_rmse=float(point[a]-point[b]),improvement_percent=float(100*(point[b]-point[a])/point[b]),ci95_low=float(lo),ci95_high=float(hi),iterations=2000,seed=20260908))
  np.savez_compressed(OUT/(c+("_original" if original_only else "")+"_analysis_arrays.npz"),names=np.array(names),y_lrp=y,source_rows=rows,tokens=tokens,calendar_blocks=blocks,predictions=pred,token_rmse=tr,equal_token_rmse=eq,calendar_rmse=br)
  print(c,"analysis complete",len(names),len(contrasts),flush=True)
 if not original_only:
  for case in P["cases"]:
   rr=[readj(OUT/"fits"/(case["id"]+"__"+o+".json")) for o,_ in ORIGINS[:3]]
   n=sum(r["test_n"] for r in rr)
   orig=float(np.sqrt(sum(r["original_rmse"]**2*r["test_n"] for r in rr)/n));alt=float(np.sqrt(sum(r["alternative_metrics"]["rmse"]**2*r["test_n"] for r in rr)/n))
   assert abs(orig-case["development_rmse"])<1e-10
   ctrl=float(np.sqrt(sum(r["control_metrics"]["rmse"]**2*r["test_n"] for r in rr)/n))
   dev.append(dict(case_id=case["id"],collection=case["collection"],group=case["group"],encoder=case["encoder"],family=case["family"],role=case["role"],original_implementation=case["original_implementation"],alternative_implementation=case["alternative_implementation"],development_n=n,original_pooled_rmse=orig,alternative_pooled_rmse=alt,control_pooled_rmse=ctrl,controlled_alternative_minus_control=alt-ctrl,counted_minus_row_pooled_rmse=(alt-ctrl if case["alternative_implementation"]=="counted" else ctrl-alt),delta_rmse=alt-orig,all_folds_valid=all(r["valid"] for r in rr)))
 prefix="original_" if original_only else ""
 writecsv(OUT/(prefix+"performance.csv"),metrics);writecsv(OUT/(prefix+"paired_comparisons.csv"),cis)
 if dev:writecsv(OUT/"development_comparison.csv",dev)
 writecsv(OUT/(prefix+"bootstrap_checks.csv"),stats.CHECKS)
 writej(OUT/(prefix+"analysis_summary.json"),dict(time=now(),performance_rows=len(metrics),comparisons=len(cis),bootstrap_checks=len(stats.CHECKS),all_passed=all(x["passed"] for x in stats.CHECKS)))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--original-only",action="store_true");args=ap.parse_args()
 with threadpool_limits(limits=1):main(args.original_only)
