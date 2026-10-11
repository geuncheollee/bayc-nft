from verify_results import *
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

print("Preflight passed",len(CHECKS),flush=True)
