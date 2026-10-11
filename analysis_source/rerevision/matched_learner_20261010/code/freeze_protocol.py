from shared import *
import numpy as np
early,meta,common,image=init_modules()
if (OUT/"protocol_freeze.json").exists():raise FileExistsError("Protocol exists")
cases=[];inputs=set([AUDIT/"Analysis_Plan.md",AUDIT/"plan_freeze.json",AUDIT/"baseline_manifest.json"])
for g in GROUPS:
 f=next(RUNS[g].glob("pre2025*model_selection*.csv"));inputs.add(f);rs=csvread(f);freeze=readj(RUNS[g]/"selection_freeze.json");inputs.add(RUNS[g]/"selection_freeze.json")
 for c in ["BAYC","MAYC"]:
  for fam in FAMILIES:
   eligible=[r for r in rs if r["collection"]==c and r["family"]==fam and (g!="metadata" or r["encoding"]=="TF-IDF")]
   r=min(eligible,key=lambda r:float(r["pre2025_validation_transaction_rmse"]))
   cases.append(dict(id=f"{c}_{g}_{r.get('encoder','TF_IDF')}_{fam}",collection=c,group=g,encoder=r.get("encoder","TF_IDF"),family=fam,
      config=json.loads(r["selected_config"]),development_rmse=float(r["pre2025_validation_transaction_rmse"]),role="matched",
      original_implementation="row" if g=="metadata" else "counted",alternative_implementation="counted" if g=="metadata" else "row"))
late=csvread(RUNS["late"]/"development_selected_late_model.csv");inputs.add(RUNS["late"]/"development_selected_late_model.csv")
for r in late:
 if not any(x["collection"]==r["collection"] and x["group"]=="image" and x["encoder"]==r["encoder"] and x["family"]==r["image_family"] for x in cases):
  rr=next(x for x in csvread(next(RUNS["image"].glob("pre2025*model_selection*.csv"))) if x["collection"]==r["collection"] and x["encoder"]==r["encoder"] and x["family"]==r["image_family"])
  cases.append(dict(id=f"{r['collection']}_image_{r['encoder']}_{r['image_family']}",collection=r["collection"],group="image",encoder=r["encoder"],family=r["image_family"],config=json.loads(rr["selected_config"]),development_rmse=float(rr["pre2025_validation_transaction_rmse"]),role="late_component",original_implementation="counted",alternative_implementation="row"))
registry=readj(early.REGISTRY_PATH);inputs.add(early.REGISTRY_PATH)
for case in cases:
 inputs.add(predpath(case));inputs.add(modelpath(case))
 if case["group"]!="metadata":
  reg=registry[case["encoder"]][case["collection"]];inputs.add(ROOT/reg["matrix"]);inputs.add(ROOT/reg["manifest"])
  if case["encoder"] in ["SDXL_VAE","DreamSim"]:
   for origin,_ in ORIGINS:
    folder=RUNS["image"]/"reduced_embeddings"/case["collection"]/case["encoder"]/origin
    inputs.update(folder/n for n in ["manifest.json","embeddings_1024.npy","pca.joblib"])
for c in ["bayc","mayc"]:
 inputs.update(ROOT/"revision/target_pipeline_20260909"/f"{c}_{n}.jsonl" for n in ["development_targets","temporal_test_targets.NOT_FOR_SELECTION"])
inputs.add(ROOT/"revision/data_audit_20260908/metadata_normalized.jsonl")
for module in list(sys.modules.values()):
 f=getattr(module,"__file__",None)
 if f:
  p=Path(f)
  if p.is_relative_to(ROOT) and p.suffix==".py" and "packages" not in p.parts and ".venv_revision_py314" not in p.parts:inputs.add(p)
inputs.update((OUT/"code").glob("*.py"))
baseline={str((ROOT/r["path"]).resolve()):r.get("sha256") for r in readj(AUDIT/"baseline_manifest.json")["files"]}
records=[]
for p in sorted(inputs):
 h=sha(p);old=baseline.get(str(p.resolve()))
 if old:assert old==h,f"Changed baseline: {p}"
 records.append(dict(path=str(p.relative_to(ROOT)),sha256=h,bytes=p.stat().st_size,baseline_verified=old is not None))
writej(OUT/"protocol_freeze.json",dict(created_utc=now(),scope="Matched learner comparisons and row-versus-counted implementation audit; no new model selection from evaluation",
 cases=cases,late_representatives=late,inputs=records,origins=ORIGINS,planned_alternative_fits=len(cases)*4,
 selection="Within each of eight learners choose metadata TF-IDF, image encoder and early encoder separately by original pooled pre-2025 score; preserve config. Same-family does not mean same selected hyperparameters or identical encoder.",
 implementation="Reuse archived original fits; refit the opposite implementation on the same features, target transform, training rows, learner and settings. Metadata row->counted; image/early counted->row. Final archived models are re-predicted to verify preprocessing and runtime.",
 count_rules="Squared-loss learners use transformed-target token means and sale-count weights. PLS uses signed sqrt(count/2) weighted-centred construction. Counted SVR retains each transaction response and intercept penalty.",
 preprocessing="Preserve metadata raw TF-IDF for trees and standardized TF-IDF for linear learners. Image/early preserve standardized features. Use existing train-only PCA at each origin. Identical matrices within every row/count pair.",
 solver="Historical model defaults retained; memory-only copy_X/copy=False on disposable expanded arrays allowed. Record convergence warnings. No evaluation-based fallback.",
 development="Three fixed 2024 folds for the same config. No retuning. Compare counterpart pooled score with archived original. Final training all 2022–2024.",
 evaluation="2025-01-01 through 2026-04-13 unchanged, plus fixed original late weights for implementation sensitivity.",
 inference="2000 paired token-cluster and 14-day calendar bootstrap, seed 20260908; 95% exploratory CI. Include equal-token CI via token bootstrap. Compare feature gains under original mixed, all-row and all-counted implementations.",
 differences="Report all numeric changes and signs; do not assert equivalence from crossing-zero intervals. For objective-equivalent non-tree models investigate material differences and optimization/rank details. Trees need not match because of bins/minimum-row rules.",
 memory="One model worker, three threads; unique-feature matrices cached only within a task; PCA reused without writing original paths.",
 prior_evaluation_inspection=True))
writecsv(OUT/"frozen_cases.csv",[{**x,"config":json.dumps(x["config"])} for x in cases])
print("Frozen",len(cases),"cases",len(cases)*4,"alternative fits",len(records),"input hashes",flush=True)
