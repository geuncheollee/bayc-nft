"""Run independent future pairs while the single-threaded row SVR is busy."""
import run_implementation as w
from shared import *
import shutil,gc
MAIN=OUT;AUX=MAIN/"auxiliary";AUX.mkdir(exist_ok=True);(AUX/"fits").mkdir(exist_ok=True)
w.OUT=AUX
cases=[x for x in w.PROTOCOL["cases"] if x["collection"]=="MAYC" and (x["group"]=="metadata" or (x["group"]=="image" and x["family"]=="XGBoost"))]
writej(AUX/"execution.json",dict(time=now(),reason="Independent future pairs computed during single-threaded LinearSVR; same worker, settings, data and three-thread limit.",cases=[x["id"] for x in cases],worker_sha256=sha(MAIN/"code/run_implementation.py"),authoritative_outputs="../fits",no_new_model_selection=True))
jobs=sorted([(x,o,e) for x in cases for o,e in ORIGINS],key=lambda t:(0 if t[0]["group"]=="metadata" else 1,t[1],FAMILIES.index(t[0]["family"])))
last=None;prep=None;published=[]
with w.threadpool_limits(limits=3):
 for case,origin,end in jobs:
  stem=case["id"]+"__"+origin;dst=MAIN/"fits"/(stem+".json")
  if dst.exists() and dst.with_suffix(".npz").exists():continue
  key=(case["collection"],case["group"],case["encoder"],origin)
  if key!=last:
   del prep;gc.collect();prep=w.prepare(case,origin,end);last=key
  result=w.run_one(case,origin,end,prep)
  progress=readj(MAIN/"progress.json")
  # Publish only while the main worker is in a different feature group,
  # well before this group. This prevents simultaneous writes for one pair.
  current=progress.get("case","")
  if not current.startswith("MAYC_"+case["group"]+"_") and not dst.exists():
   src=AUX/"fits"/(stem+".json")
   shutil.move(str(src.with_suffix(".npz")),str(dst.with_suffix(".npz")))
   shutil.move(str(src),str(dst))
   published.append(stem)
   writej(AUX/"published.json",dict(time=now(),published=published))
  else:
   writej(AUX/"not_published_"+stem+".json",dict(reason="Main worker reached feature group; private result excluded to avoid concurrent output writes."))
writej(AUX/"complete.json",dict(time=now(),published=len(published),planned=len(jobs)))
print("Auxiliary complete",len(published),flush=True)
