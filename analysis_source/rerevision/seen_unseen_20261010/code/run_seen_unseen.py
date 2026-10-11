"""Seen/unseen evaluation of frozen models; no model fitting or reselection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, sys
import numpy as np

ROOT=Path(r"D:/Research/BAYC_NFT")
OUT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"rerevision/baseline_audit_20261010"
PKG=ROOT/"revision2/revision_package_20260923_v3"
TARGET=ROOT/"revision/target_pipeline_20260909"
NAMES=["zero","metadata","image","early","late"]
CONTRASTS=[("early","metadata"),("late","metadata"),("early","late")]
SEED=20260908
ITERATIONS=2000
CHECKS=[]
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def readcsv(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def readjl(p):
    with Path(p).open(encoding="utf-8-sig") as f:return [json.loads(s) for s in f if s.strip()]
def writej(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def writecsv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def require(name,condition,detail=""):
    CHECKS.append({"check":name,"passed":bool(condition),"detail":detail})
    if not condition:raise AssertionError(name+": "+str(detail))
def close(name,actual,expected,tol=1e-10):
    require(name,bool(np.allclose(actual,expected,rtol=0,atol=tol)),f"max_abs_diff={np.max(np.abs(np.asarray(actual)-np.asarray(expected)))}")
def sources():
    inv=readj(PKG/"evidence/inventory.json")
    chosen=readcsv(AUDIT/"selected_models.csv")
    files=[AUDIT/"Analysis_Plan.md",AUDIT/"plan_freeze.json",AUDIT/"selected_models.csv",AUDIT/"cohort_audit.csv",AUDIT/"main_metrics_recomputed.csv",PKG/"evidence/inventory.json",Path(__file__)]
    paths={}
    for c in ["BAYC","MAYC"]:
        files.extend([TARGET/f"{c.lower()}_development_targets.jsonl",TARGET/f"{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"])
        for g in NAMES[1:]:
            r=next(r for r in chosen if r["collection"]==c and r["group"]==g)
            stem=f"{c}_TF_IDF_{r['family']}" if g=="metadata" else f"{c}_{r['encoder']}_{r['image_family'] if g=='late' else r['family']}"
            p=ROOT/inv[g]["path"]/"predictions"/(stem+".npz");files.append(p);paths[c,g]=p
    return chosen,files,paths
def freeze():
    if (OUT/"protocol_freeze.json").exists():raise FileExistsError("Protocol already frozen.")
    chosen,files,paths=sources()
    bm={str((ROOT/r["path"]).resolve()):r["sha256"] for r in readj(AUDIT/"baseline_manifest.json")["files"]}
    for r in readj(AUDIT/"output_manifest.json")["files"]:bm[str((AUDIT/r["path"]).resolve())]=r["sha256"]
    records=[]
    for p in files:
        actual=sha(p);previous=bm.get(str(p.resolve()))
        if previous is not None:require("baseline_hash "+str(p),actual==previous)
        records.append(dict(path=str(p.relative_to(ROOT)),sha256=actual,bytes=p.stat().st_size,prior_baseline_verified=previous is not None))
    require("plan freeze hash",sha(AUDIT/"Analysis_Plan.md")==readj(AUDIT/"plan_freeze.json")["plan_sha256"])
    writej("protocol_freeze.json",dict(created_utc=now(),analysis="seen_unseen_training_tokens",status="frozen_before_new_subgroup_results",
      reviewer_requests=["Editor 1","Reviewer 3"],training_start="2022-01-01T00:00:00+00:00",training_end_exclusive="2025-01-01T00:00:00+00:00",
      evaluation_start="2025-01-01T00:00:00+00:00",evaluation_end_exclusive="2026-04-14T00:00:00+00:00",
      seen_definition="Token has at least one eligible modelling row in the actual 2022–2024 final training cohort; same definition for all selected models.",
      legacy_flag_policy="Recalculate membership; compare existing known_training_token flag only as a provenance diagnostic.",
      selection_policy="Use the eight baseline development-selected learned representatives unchanged, plus zero. No within-subgroup fitting or reselection.",
      target="Original stored transaction LRP with the base-clean past-only market reference unchanged.",
      subgroups=["seen","unseen"],pooled_role="Reproduce previous full-evaluation results as an integrity control.",
      point_metrics=["transaction_rmse","transaction_mae","transaction_r2","equal_token_rmse"],
      contrasts=[{"candidate":a,"reference":b} for a,b in CONTRASTS],
      bootstrap=dict(iterations=ITERATIONS,seed=SEED,interval="percentile 95%, linear quantile",
        schemes=["paired_token_cluster","paired_14day_calendar_block"],calendar_anchor="2025-01-01 UTC",calendar_days=14,
        equal_token_ci="Token-cluster bootstrap only, as in baseline outputs.",
        rng="Reset default_rng(20260908) for each collection, subgroup and resampling scheme; all models share draws.",
        empty_blocks="Sample occupied blocks within the fixed 14-day calendar partition; retain original calendar block IDs.",
        insufficient_clusters="Do not calculate intervals with fewer than two occupied clusters; flag degenerate target variance.",
        interpretation="Exploratory fixed-prediction uncertainty; no simultaneous subgroup correction, refitting or model-selection uncertainty."),
      limits=["Retrospective evaluation previously examined.","Unseen means absent from final training rows, not necessarily never sold before 2022 or absent from generic encoder pretraining.","No causal inference from seen versus unseen performance levels."],
      selected_models=chosen,inputs=records))
    print("Protocol frozen with",len(records),"input files",flush=True)

def metrics(y,p,tokens):
    _,idx,n=np.unique(tokens,return_inverse=True,return_counts=True)
    sq=(y[:,None]-p)**2
    tr=np.sqrt(sq.mean(axis=0));mae=np.abs(y[:,None]-p).mean(axis=0)
    sst=float(np.sum((y-y.mean())**2))
    r2=(1-sq.sum(axis=0)/sst) if sst>0 else [None]*p.shape[1]
    equal=np.sqrt(np.array([np.mean(np.bincount(idx,weights=sq[:,j])/n) for j in range(p.shape[1])]))
    return tr,mae,r2,equal

def bootstrap(y,p,groups,equal,label):
    vals,idx,n=np.unique(groups,return_inverse=True,return_counts=True)
    if len(n)<2:return None,None
    sq=(y[:,None]-p)**2
    ss=np.column_stack([np.bincount(idx,weights=sq[:,i],minlength=len(n)) for i in range(p.shape[1])])
    rng=np.random.default_rng(SEED)
    tr=np.empty((ITERATIONS,p.shape[1]));eq=np.empty_like(tr) if equal else None
    # Same paired resampling units for all five models.
    for start in range(0,ITERATIONS,64):
        stop=min(start+64,ITERATIONS);draw=rng.integers(0,len(n),size=(stop-start,len(n)))
        tr[start:stop]=np.sqrt(ss[draw].sum(axis=1)/n[draw].sum(axis=1)[:,None])
        if equal:eq[start:stop]=np.sqrt((ss/n[:,None])[draw].mean(axis=1))
    require(label+" finite bootstrap",np.isfinite(tr).all() and (eq is None or np.isfinite(eq).all()))
    # Independent row-expansion check of the first eight draws.
    rng=np.random.default_rng(SEED);members=[np.flatnonzero(idx==i) for i in range(len(n))]
    for b in range(8):
        draw=rng.integers(0,len(n),size=len(n))
        expanded=np.concatenate([members[i] for i in draw])
        direct=np.sqrt(np.mean((y[expanded,None]-p[expanded])**2,axis=0))
        close(label+f" direct_row_draw_{b}",tr[b],direct)
        if equal:
            direct_eq=np.sqrt(np.mean([sq[members[i]].mean(axis=0) for i in draw],axis=0))
            close(label+f" direct_equal_draw_{b}",eq[b],direct_eq)
    return tr,eq

def run():
    if (OUT/"completion.json").exists():raise FileExistsError("Completed output exists; use a separate directory to rerun.")
    protocol=readj(OUT/"protocol_freeze.json")
    for r in protocol["inputs"]:require("frozen_input "+r["path"],sha(ROOT/r["path"])==r["sha256"])
    chosen,_,paths=sources()
    expected=readcsv(AUDIT/"main_metrics_recomputed.csv");expected_cohort=readcsv(AUDIT/"cohort_audit.csv")
    allmetrics=[];allcounts=[];allcis=[];membership=[];notes=[]
    for c in ["BAYC","MAYC"]:
        dev=readjl(TARGET/f"{c.lower()}_development_targets.jsonl")
        training=[r for r in dev if "2022-01-01"<=r["time"]<"2025-01-01"]
        train_tokens={str(r["token_id"]) for r in training}
        prior_tokens={str(r["token_id"]) for r in dev if r["time"]<"2022-01-01"}
        target=readjl(TARGET/f"{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
        tokens=np.array([str(r["token_id"]) for r in target]);source=np.array([r["source_row"] for r in target],dtype=np.int64)
        y=np.array([r["y_log_relative_price"] for r in target],dtype=np.float64)
        timestamps=[datetime.fromisoformat(r["time"]) for r in target]
        anchor=datetime(2025,1,1,tzinfo=timezone.utc)
        blocks=np.array([(t-anchor).days//14 for t in timestamps],dtype=np.int64)
        ec=next(r for r in expected_cohort if r["collection"]==c)
        require(c+" train sales",len(training)==int(ec["training_sales"]))
        require(c+" train tokens",len(train_tokens)==int(ec["training_tokens"]))
        require(c+" eval sales",len(y)==int(ec["evaluation_sales"]))
        require(c+" eval tokens",len(set(tokens))==int(ec["evaluation_tokens"]))
        require(c+" eval period",all(anchor<=t<datetime(2026,4,14,tzinfo=timezone.utc) for t in timestamps))
        require(c+" unique eval rows",len(set(source))==len(source))
        cols=[np.zeros(len(y))]
        for g in NAMES[1:]:
            with np.load(paths[c,g],allow_pickle=False) as z:
                for key,ex in [("source_rows",source),("tokens",tokens),("y_lrp",y)]:
                    ac=z[key].astype(str) if key=="tokens" else z[key]
                    require(c+" "+g+" "+key,np.array_equal(ac,ex))
                cols.append(z["predictions_lrp"].copy())
        p=np.column_stack(cols)
        require(c+" finite arrays",np.isfinite(p).all() and np.isfinite(y).all())
        seen=np.array([t in train_tokens for t in tokens])
        legacy=np.array([r.get("known_training_token",False) for r in target])
        require(c+" disjoint partition",not bool(np.any(seen & ~seen)))
        require(c+" exhaustive partition",int(seen.sum())+int((~seen).sum())==len(y))
        require(c+" token partition disjoint",not (set(tokens[seen])&set(tokens[~seen])))
        require(c+" no unseen training tokens",not (set(tokens[~seen])&train_tokens))
        membership.append(dict(collection=c,legacy_field_present=all("known_training_token" in r for r in target),
            legacy_disagreement_rows=int(np.sum(legacy!=seen)),legacy_disagreement_tokens=len(set(tokens[legacy!=seen])),
            unseen_with_pre2022_history_rows=sum(t in prior_tokens for t in tokens[~seen]),
            unseen_with_pre2022_history_tokens=len(set(tokens[~seen])&prior_tokens)))
        assign=[dict(collection=c,source_row=int(source[i]),token_id=tokens[i],seen_in_2022_2024=bool(seen[i]),subgroup="seen" if seen[i] else "unseen",calendar_block_14d=int(blocks[i])) for i in range(len(y))]
        writecsv(c+"_row_membership.csv",assign)
        np.savez_compressed(OUT/(c+"_partitioned_predictions.npz"),source_rows=source,tokens=tokens,y_lrp=y,
            predictions_lrp=p,model_names=np.array(NAMES),seen_in_2022_2024=seen,calendar_blocks=blocks,
            training_tokens=np.array(sorted(train_tokens)))
        group_scores={}
        for subgroup,mask in [("all",np.ones(len(y),dtype=bool)),("seen",seen),("unseen",~seen)]:
            yy,pp,tt,bb=y[mask],p[mask],tokens[mask],blocks[mask]
            if len(yy)==0:
                notes.append(f"{c} {subgroup}: empty subgroup; no metrics or intervals.")
                continue
            vals=metrics(yy,pp,tt);group_scores[subgroup]=vals
            allcounts.append(dict(collection=c,subgroup=subgroup,n_transactions=len(yy),n_tokens=len(set(tt)),transaction_percent=100*len(yy)/len(y),
                token_percent=100*len(set(tt))/len(set(tokens)),n_calendar_blocks=len(set(bb)),target_sd=float(np.std(yy)),
                evaluation_min=min(timestamps[i].isoformat() for i in np.flatnonzero(mask)),evaluation_max=max(timestamps[i].isoformat() for i in np.flatnonzero(mask))))
            for i,g in enumerate(NAMES):
                m=dict(collection=c,subgroup=subgroup,model=g,n_transactions=len(yy),n_tokens=len(set(tt)),transaction_rmse=float(vals[0][i]),
                    transaction_mae=float(vals[1][i]),transaction_r2=None if vals[2][i] is None else float(vals[2][i]),equal_token_rmse=float(vals[3][i]))
                allmetrics.append(m)
                if subgroup=="all":
                    ref=next(r for r in expected if r["collection"]==c and r["group"]==g)
                    for metric in ["transaction_rmse","transaction_mae","transaction_r2","equal_token_rmse"]:close(c+" pooled "+g+" "+metric,m[metric],float(ref[metric]))
            tok_tr,tok_eq=bootstrap(yy,pp,tt,True,c+" "+subgroup+" token")
            block_tr,_=bootstrap(yy,pp,bb,False,c+" "+subgroup+" block")
            save=dict(model_names=np.array(NAMES))
            for key,v in [("token_transaction_rmse",tok_tr),("token_equal_token_rmse",tok_eq),("block_transaction_rmse",block_tr)]:
                if v is not None:save[key]=v
            np.savez_compressed(OUT/(c+"_"+subgroup+"_bootstrap.npz"),**save)
            for scheme,metric,point,reps,cluster_n in [
                ("paired_token_cluster","transaction_rmse",vals[0],tok_tr,len(set(tt))),
                ("paired_token_cluster","equal_token_rmse",vals[3],tok_eq,len(set(tt))),
                ("paired_14day_calendar_block","transaction_rmse",vals[0],block_tr,len(set(bb)))]:
                for candidate,reference in CONTRASTS:
                    ci,ri=NAMES.index(candidate),NAMES.index(reference)
                    delta=float(point[ci]-point[ri]);utility=float(100*(point[ri]-point[ci])/point[ri]) if point[ri]>0 else None
                    row=dict(collection=c,subgroup=subgroup,resampling=scheme,metric=metric,candidate=candidate,reference=reference,
                        n_transactions=len(yy),n_tokens=len(set(tt)),n_clusters=cluster_n,candidate_rmse=float(point[ci]),
                        reference_rmse=float(point[ri]),delta_rmse=delta,improvement_percent=utility,iterations=ITERATIONS,seed=SEED)
                    if reps is None or float(np.std(yy))==0:
                        row.update(status="interval_unavailable",ci95_low=None,ci95_high=None,improvement_ci95_low=None,improvement_ci95_high=None)
                        notes.append(c+" "+subgroup+" "+scheme+": insufficient clusters or constant target.")
                    else:
                        dd=reps[:,ci]-reps[:,ri];lo,hi=np.quantile(dd,[.025,.975],method="linear")
                        util=100*(reps[:,ri]-reps[:,ci])/reps[:,ri];ul,uh=np.quantile(util,[.025,.975],method="linear")
                        row.update(status="degenerate_contrast" if np.ptp(dd)==0 else "ok",ci95_low=float(lo),ci95_high=float(hi),
                            improvement_ci95_low=float(ul),improvement_ci95_high=float(uh))
                    allcis.append(row)
            print(c,subgroup,"transactions",len(yy),"tokens",len(set(tt)),flush=True)
        if all(k in group_scores for k in ["all","seen","unseen"]):
            ns=seen.sum();nu=(~seen).sum();ks=len(set(tokens[seen]));ku=len(set(tokens[~seen]))
            close(c+" recombine transaction MSE",(ns*group_scores["seen"][0]**2+nu*group_scores["unseen"][0]**2)/(ns+nu),group_scores["all"][0]**2)
            close(c+" recombine transaction MAE",(ns*group_scores["seen"][1]+nu*group_scores["unseen"][1])/(ns+nu),group_scores["all"][1])
            close(c+" recombine equal-token MSE",(ks*group_scores["seen"][3]**2+ku*group_scores["unseen"][3]**2)/(ks+ku),group_scores["all"][3]**2)
        del dev,training,target
    writecsv("subgroup_counts.csv",allcounts);writecsv("performance.csv",allmetrics);writecsv("paired_comparisons.csv",allcis)
    writecsv("membership_definition_audit.csv",membership);writecsv("verification_checks.csv",CHECKS)
    for r in protocol["inputs"]:require("input unchanged "+r["path"],sha(ROOT/r["path"])==r["sha256"])
    writecsv("verification_checks.csv",CHECKS)
    writej("results_summary.json",dict(completed_utc=now(),counts=allcounts,performance=allmetrics,
      token_transaction_contrasts=[r for r in allcis if r["subgroup"]!="all" and r["resampling"]=="paired_token_cluster" and r["metric"]=="transaction_rmse"],
      calendar_transaction_contrasts=[r for r in allcis if r["subgroup"]!="all" and r["resampling"]=="paired_14day_calendar_block"],
      membership_definition=membership,notes=notes,checks=len(CHECKS),failed_checks=0,python=sys.version,numpy=np.__version__,analysis_scope="Stored fixed-prediction subgroup evaluation; no refitting/reselection."))
    writej("completion.json",dict(completed_utc=now(),completed_step=3,passed=True,checks=len(CHECKS),failed_checks=0,
       performance_rows=len(allmetrics),comparison_rows=len(allcis),new_models_fitted=False,model_selection_changed=False,
       source_manuscript_modified=False,protocol_sha256=sha(OUT/"protocol_freeze.json")))
    print("Completed",len(allmetrics),"performance rows;",len(allcis),"comparison rows;",len(CHECKS),"checks",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--freeze",action="store_true");args=ap.parse_args()
    if args.freeze:freeze()
    else:run()
