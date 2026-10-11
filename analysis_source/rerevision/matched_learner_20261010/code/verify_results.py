"""Independent checks of the matched-family experiment and original integrity."""
from shared import *
import numpy as np
CHECKS=[]
def check(name,ok,detail=""):
 CHECKS.append(dict(check=name,passed=bool(ok),detail=str(detail)))
 if not ok:raise AssertionError(name+" "+str(detail))
def main():
 p=readj(OUT/"protocol_freeze.json");a=readj(OUT/"amendment_01.json")
 for x in p["inputs"]:
  path=ROOT/x["path"];expected=a["new_code_sha256"] if path.resolve()==(OUT/"code/run_implementation.py").resolve() else x["sha256"]
  check("frozen input "+x["path"],sha(path)==expected)
 check("preliminary code preserved",sha(OUT/"preliminary_run_01/run_implementation.py")==a["preliminary_code_sha256"])
 # Selection is re-derived from development tables, never evaluation metrics.
 for case in p["cases"]:
  if case["role"]!="matched":continue
  files=list(RUNS[case["group"]].glob("pre2025*model_selection*.csv"));check("one selection file "+case["group"],len(files)==1)
  rows=csvread(files[0]);rows=[r for r in rows if r["collection"]==case["collection"] and r["family"]==case["family"] and (case["group"]!="metadata" or r.get("metadata_encoding",r.get("encoding"))=="TF-IDF")]
  score=lambda r:float(r["pre2025_validation_transaction_rmse"])
  best=min(rows,key=score)
  check(case["id"]+" development selection",abs(score(best)-case["development_rmse"])<1e-12)
  if case["group"]!="metadata":check(case["id"]+" encoder selected",best["encoder"]==case["encoder"])
 for l in p["late_representatives"]:
  m=next(x for x in p["cases"] if x["collection"]==l["collection"] and x["group"]=="metadata" and x["family"]==l["metadata_family"])
  im=next(x for x in p["cases"] if x["collection"]==l["collection"] and x["group"]=="image" and x["family"]==l["image_family"] and x["encoder"]==l["encoder"])
  check(l["collection"]+" frozen late metadata config",m["config"]==json.loads(l["metadata_config"]))
  check(l["collection"]+" frozen late image config",im["config"]==json.loads(l["image_config"]))
  check(l["collection"]+" frozen late weight",float(l["image_weight"])==(.2 if l["collection"]=="BAYC" else .4))
 records=[]
 for c in ["BAYC","MAYC"]:
  data=lines(ROOT/"revision/target_pipeline_20260909"/f"{c.lower()}_development_targets.jsonl")
  evaluation=lines(ROOT/"revision/target_pipeline_20260909"/f"{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
  for case in [x for x in p["cases"] if x["collection"]==c]:
   for origin,end in ORIGINS:
    stem=case["id"]+"__"+origin;r=readj(OUT/"fits"/(stem+".json"));records.append(r)
    train=[x for x in data if "2022-01-01"<=x["time"]<origin]
    valid=evaluation if origin=="2025-01-01" else [x for x in data if origin<=x["time"]<end]
    y=np.array([x["y_log_relative_price"] for x in valid]);row=np.array([x["source_row"] for x in valid]);tok=np.array([str(x["token_id"]) for x in valid])
    check(stem+" config fixed",r["config"]==case["config"])
    check(stem+" fit valid",r["valid"] and r["control_status"]["valid"],r["warnings"])
    check(stem+" train count",len(train)==r["training_n"])
    check(stem+" train tokens",len({x["token_id"] for x in train})==r["training_tokens"])
    yt=np.array([x["y_log_relative_price"] for x in train]);center=np.median(yt);mad=np.median(np.abs(yt-center))/0.6744897501960817
    check(stem+" target transform",abs(center-r["target_parameters"]["center"])<1e-12 and abs(mad-r["target_parameters"]["normalized_MAD"])<1e-12)
    with np.load(OUT/"fits"/(stem+".npz"),allow_pickle=False) as z:
     for k,v in [("y_lrp",y),("source_rows",row),("tokens",tok)]:check(stem+" "+k,np.array_equal(z[k],v))
     for k,met in [("alternative_predictions",r["alternative_metrics"]),("control_predictions",r["control_metrics"])]:
      pred=z[k];e=y-pred
      for metric,val in [("rmse",np.sqrt(np.dot(e,e)/len(e))),("mae",np.mean(np.abs(e))),("r2",1-np.dot(e,e)/np.dot(y-y.mean(),y-y.mean()))]:
       check(stem+" "+k+" "+metric,np.isclose(val,met[metric],rtol=1e-12,atol=1e-12))
     check(stem+" controlled prediction gap",abs(np.max(np.abs(z["alternative_predictions"]-z["control_predictions"]))-r["controlled_prediction_max_abs_gap"])<1e-12)
    if origin=="2025-01-01":check(stem+" archived inference reproduced",r["archived_inference_max_abs_gap"]<=1e-7)
 check("all196 paired experiments",len(records)==196)
 check("all bootstrap arithmetic checks",all(x["passed"]=="True" for x in csvread(OUT/"bootstrap_checks.csv")))
 check("performance records",len(csvread(OUT/"performance.csv"))==153)
 check("paired comparison records",len(csvread(OUT/"paired_comparisons.csv"))==471)
 final_perf={x["model"]:x for x in csvread(OUT/"performance.csv")}
 for old in csvread(OUT/"original_performance.csv"):
  for metric in ["transaction_rmse","transaction_mae","transaction_r2","equal_token_rmse"]:
   check(old["model"]+" archived analysis unchanged "+metric,abs(float(old[metric])-float(final_perf[old["model"]][metric]))<1e-10)
 full_ci=csvread(OUT/"paired_comparisons.csv")
 keys=["collection","kind","family","group","mode","candidate","reference","resampling","metric"]
 ci_index={tuple(x[k] for k in keys):x for x in full_ci}
 for old in csvread(OUT/"original_paired_comparisons.csv"):
  new=ci_index[tuple(old[k] for k in keys)]
  for metric in ["delta_rmse","ci95_low","ci95_high"]:
   check(str(tuple(old[k] for k in keys))+" original contrast unchanged "+metric,abs(float(old[metric])-float(new[metric]))<1e-10)
 writecsv(OUT/"verification_checks.csv",CHECKS)
 writej(OUT/"verification_summary.json",dict(time=now(),checks=len(CHECKS),failed=0,paired_cases=len(records),refits=2*len(records),archived_inference_cases=49))
 print("PASS",len(CHECKS),"checks",flush=True)
if __name__=="__main__":main()
