"""Independent verification of exported subgroup results and pooled controls."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
ROOT=Path(r"D:/Research/BAYC_NFT")
OUT=Path(__file__).resolve().parents[1]
def csvread(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def jl(p):
    with p.open(encoding="utf-8") as f:return [json.loads(s) for s in f if s.strip()]
checks=[]
def ck(name,ok):
    checks.append(dict(check=name,passed=bool(ok)))
    assert ok,name
perf=csvread(OUT/"performance.csv");comparisons=csvread(OUT/"paired_comparisons.csv")
legacy=[]
for c in ["BAYC","MAYC"]:
    with np.load(OUT/f"{c}_partitioned_predictions.npz",allow_pickle=False) as z:
        y=z["y_lrp"];p=z["predictions_lrp"];tokens=z["tokens"];seen=z["seen_in_2022_2024"];names=list(z["model_names"])
        tt=set(z["training_tokens"]);source=z["source_rows"]
    ck(c+" membership from exported training tokens",np.array_equal(seen,np.array([t in tt for t in tokens])))
    dev=jl(ROOT/"revision/target_pipeline_20260909"/f"{c.lower()}_development_targets.jsonl")
    target=jl(ROOT/"revision/target_pipeline_20260909"/f"{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
    all_dev={str(r["token_id"]) for r in dev};flag=np.array([r["known_training_token"] for r in target])
    ck(c+" legacy flag reflects all development years",np.array_equal(flag,np.array([t in all_dev for t in tokens])))
    legacy.append(dict(collection=c,legacy_definition="All years in canonical development target file, including 2021",disagreeing_rows=int(np.sum(flag!=seen)),
        disagreeing_tokens=len(set(tokens[flag!=seen])),legacy_seen_actual_unseen=int(np.sum(flag&~seen)),legacy_unseen_actual_seen=int(np.sum(~flag&seen))))
    for group,mask in [("all",np.ones(len(y),dtype=bool)),("seen",seen),("unseen",~seen)]:
        yy=y[mask];pp=p[mask];tok=tokens[mask];n=len(yy)
        for j,name in enumerate(names):
            row=next(r for r in perf if r["collection"]==c and r["subgroup"]==group and r["model"]==name)
            sse=math.fsum(float(a-b)**2 for a,b in zip(yy,pp[:,j]))
            mae=math.fsum(abs(float(a-b)) for a,b in zip(yy,pp[:,j]))/n
            avg=math.fsum(float(a) for a in yy)/n
            r2=1-sse/math.fsum((float(a)-avg)**2 for a in yy)
            by={}
            for t,a,b in zip(tok,yy,pp[:,j]):by.setdefault(t,[]).append(float(a-b)**2)
            equal=math.sqrt(math.fsum(math.fsum(v)/len(v) for v in by.values())/len(by))
            for key,v in [("transaction_rmse",math.sqrt(sse/n)),("transaction_mae",mae),("transaction_r2",r2),("equal_token_rmse",equal)]:
                ck(c+" "+group+" "+name+" "+key,abs(v-float(row[key]))<=1e-10)
# All-sample CIs must reproduce the earlier baseline.
old=csvread(ROOT/"results/fusion_uncertainty_20260922_v1/paired_comparisons.csv")
for r in comparisons:
    if r["subgroup"]!="all":continue
    ref=next(x for x in old if all(x[k]==r[k] for k in ["collection","resampling","metric","candidate","reference"]))
    for newkey,oldkey in [("delta_rmse","delta_rmse_candidate_minus_reference"),("ci95_low","delta_rmse_ci95_low"),("ci95_high","delta_rmse_ci95_high")]:
        ck("pooled CI "+str([r[k] for k in ["collection","resampling","metric","candidate","reference"]])+newkey,abs(float(r[newkey])-float(ref[oldkey]))<=1e-10)
    for low,high in [("ci95_low","ci95_high"),("improvement_ci95_low","improvement_ci95_high")]:ck("interval ordering",float(r[low])<=float(r[high]))
(OUT/"independent_verification.json").write_text(json.dumps(dict(passed=True,checks=len(checks),failed_checks=0,legacy_flag=legacy,details=checks),indent=2),encoding="utf-8")
print(json.dumps(dict(passed=True,checks=len(checks),legacy_flag=legacy),indent=2))
