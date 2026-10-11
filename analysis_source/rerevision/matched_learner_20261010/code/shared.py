from pathlib import Path
from datetime import datetime,timezone
import sys,json,csv,hashlib,importlib.util
sys.dont_write_bytecode=True
ROOT=Path(r"D:/Research/BAYC_NFT");OUT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"rerevision/baseline_audit_20261010"
PKG=ROOT/"revision2/revision_package_20260923_v3"
FAMILIES=["OLS","Ridge","ElasticNet","PLS","LinearSVR","HistGradientBoosting","XGBoost","LightGBM"]
GROUPS=["metadata","image","early"]
ORIGINS=[("2024-04-01","2024-07-01"),("2024-07-01","2024-10-01"),("2024-10-01","2025-01-01"),("2025-01-01","2026-04-14")]
def readj(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def csvread(p):
 with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def lines(p):
 with Path(p).open(encoding="utf-8-sig") as f:return [json.loads(s) for s in f if s.strip()]
def writej(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def writecsv(p,rows):
 if not rows:return
 with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
INV=readj(PKG/"evidence/inventory.json")
RUNS={g:ROOT/v["path"] for g,v in INV.items()}
def init_modules():
 sys.path.insert(0,str(ROOT/"results/early_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260921/code"))
 import run_early_fusion_tfidf_experiment as early
 spec=importlib.util.spec_from_file_location("transaction_metadata_run",ROOT/"results/metadata_asinh_transaction_2022_2024_nine_regressors_20260920/code/run_experiment.py")
 meta=importlib.util.module_from_spec(spec);spec.loader.exec_module(meta)
 return early,meta,early.common,early.image_run
def predpath(case):
 stem=case["collection"]+"_"+("TF_IDF" if case["group"]=="metadata" else case["encoder"])+"_"+case["family"]
 return RUNS[case["group"]]/"predictions"/(stem+".npz")
def modelpath(case):return RUNS[case["group"]]/"models"/(predpath(case).stem+".joblib")
