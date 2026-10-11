from engine import *
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import subprocess

def main():
 protocol=readj(OUT/'protocol_freeze.json')
 for r in readj(OUT/'execution_sources.json')['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
 jobs=[(cond,c,enc,o,e) for cond in CONDITIONS for c in ['BAYC','MAYC'] for enc in ['TF_IDF']+ENCODERS for o,e in FOLDS]
 todo=[j for j in jobs if not (OUT/'jobs'/('__'.join(j[:4])+'.json')).exists()]
 savej(OUT/'run_started.json',dict(started_utc=now(),protocol_sha256=sha(OUT/'protocol_freeze.json'),jobs=len(jobs),resume_jobs=len(todo)))
 with ProcessPoolExecutor(max_workers=2) as pool:
  pending={pool.submit(fit_task,j):j for j in todo}
  while pending:
   done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
   for f in done:
    j=pending.pop(f);res=f.result();print('JOB_COMPLETE',res,flush=True)
   count=sum(1 for _ in (OUT/'checkpoints').rglob('*.json'))
   savej(OUT/'progress.json',dict(updated_utc=now(),stage='development_search',completed_checkpoints=count,expected_checkpoints=15930,completed_jobs=len(jobs)-len(pending),expected_jobs=len(jobs)))
 assert sum(1 for _ in (OUT/'checkpoints').rglob('*.json'))==15930
 savej(OUT/'development_complete.json',dict(completed_utc=now(),checkpoints=15930,jobs=len(jobs)))
 for name in ['select_final.py','analyze_results.py','verify_results.py','write_reports.py']:
  subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'code'/name)],check=True,cwd=OUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 savej(OUT/'pipeline_done.json',dict(completed_utc=now(),status='complete'))
if __name__=='__main__':
 try:main()
 except Exception:
  savej(OUT/'pipeline_error.json',dict(time=now(),error=traceback.format_exc()));raise
