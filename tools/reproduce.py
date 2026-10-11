"""Recompute reported metrics and paired confidence intervals from public rows.
Requires only NumPy. Never opens original project inputs or fitted models.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEED=20260908
ITERATIONS=2000
METRICS=['transaction_rmse','transaction_mae','transaction_r2','equal_token_rmse']

def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writecsv(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def verify_integrity():
    records=readj(ROOT/'MANIFEST_SHA256.json')
    for r in records:
        p=ROOT/r['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Integrity failure: '+r['path'])
    return len(records)
def load_dataset(item):
    with np.load(ROOT/item['path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
    y,p=d['target_lrp'],d['predictions_lrp']
    if p.shape!=(len(y),len(d['model_names'])):raise ValueError('Prediction shape')
    if len(set(d['row_id']))!=len(y):raise ValueError('Duplicate row IDs')
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite input')
    lookup={int(x):i for i,x in enumerate(d['token_cluster_order'])}
    if len(lookup)!=len(set(d['token_group'])) or set(lookup)!=set(d['token_group']):raise ValueError('Token order mismatch')
    d['token_index']=np.array([lookup[int(x)] for x in d['token_group']],dtype=np.int64)
    return d
def calculate(d,bootstrap_needed=True):
    y=d['target_lrp'];p=d['predictions_lrp'];idx=d['token_index']
    n=np.bincount(idx);sq=(y[:,None]-p)**2
    ss=np.column_stack([np.bincount(idx,weights=sq[:,j]) for j in range(p.shape[1])])
    sst=((y-y.mean())**2).sum()
    points=dict(transaction_rmse=np.sqrt(sq.mean(axis=0)),transaction_mae=np.abs(y[:,None]-p).mean(axis=0),
                transaction_r2=1-sq.sum(axis=0)/sst,equal_token_rmse=np.sqrt((ss/n[:,None]).mean(axis=0)))
    out={};direct=[]
    if not bootstrap_needed:return points,out,direct
    for scheme in ['token','calendar14d']:
        if scheme=='token':group_index=idx;counts=n;sums=ss
        else:
            _,group_index,counts=np.unique(d['calendar_block_14d'],return_inverse=True,return_counts=True)
            sums=np.column_stack([np.bincount(group_index,weights=sq[:,j]) for j in range(p.shape[1])])
        rng=np.random.default_rng(SEED)
        tr=np.empty((ITERATIONS,p.shape[1]));eq=np.empty_like(tr) if scheme=='token' else None
        for start in range(0,ITERATIONS,16):
            stop=min(start+16,ITERATIONS)
            draw=rng.integers(0,len(counts),size=(stop-start,len(counts)))
            tr[start:stop]=np.sqrt(sums[draw].sum(axis=1)/counts[draw].sum(axis=1)[:,None])
            if eq is not None:eq[start:stop]=np.sqrt((sums/counts[:,None])[draw].mean(axis=1))
        out[scheme,'transaction_rmse']=tr
        if eq is not None:out[scheme,'equal_token_rmse']=eq
        # Independent explicit row expansion verifies the cluster calculation.
        rng=np.random.default_rng(SEED);members=[np.flatnonzero(group_index==i) for i in range(len(counts))]
        for k in range(2):
            draw=rng.integers(0,len(counts),size=len(counts));expanded=np.concatenate([members[i] for i in draw])
            gap=float(np.max(np.abs(np.sqrt(sq[expanded].mean(axis=0))-tr[k])))
            if gap>1e-10:raise ValueError('Explicit bootstrap row expansion failed')
            direct.append(dict(scheme=scheme,draw=k,max_abs_difference=gap))
            if eq is not None:
                gap=float(np.max(np.abs(np.sqrt(np.mean([sq[members[i]].mean(axis=0) for i in draw],axis=0))-eq[k])))
                if gap>1e-10:raise ValueError('Equal-token expansion failed')
                direct.append(dict(scheme='token_equal',draw=k,max_abs_difference=gap))
    return points,out,direct
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'reproduced')
    ap.add_argument('--dataset',action='append',help='Optional dataset ID; omission runs the entire package.')
    args=ap.parse_args()
    start=time.time();nfiles=verify_integrity()
    catalog=readj(ROOT/'data/catalog.json');expected=readj(ROOT/'expected/reproduction_contract.json')
    selected=[d for d in catalog if not args.dataset or d['id'] in args.dataset]
    if args.dataset and set(args.dataset)-{d['id'] for d in selected}:raise ValueError('Unknown dataset')
    out=args.output_dir.resolve()
    if out==ROOT or any(out==ROOT/x for x in ['data','expected','tools','analysis_source','legacy_v4_0_3']):raise ValueError('Choose a separate output directory')
    out.mkdir(parents=True,exist_ok=True)
    checks=[];metrics=[];intervals=[];direct=[]
    for item in selected:
        ds=item['id'];d=load_dataset(item);names=list(d['model_names'])
        req=[r for r in expected if r['dataset']==ds]
        points,reps,drawchecks=calculate(d,any(r['operation']=='interval' for r in req))
        direct.extend(dict(dataset=ds,**r) for r in drawchecks)
        for j,name in enumerate(names):
            metrics.append(dict(dataset=ds,collection=item['collection'],model=name,n_transactions=len(d['target_lrp']),
                n_tokens=len(d['token_cluster_order']),**{k:float(v[j]) for k,v in points.items()}))
        cache={}
        for r in req:
            op=r['operation']
            if op=='count':actual=len(d['target_lrp']) if r['metric']=='n_transactions' else len(d['token_cluster_order'])
            elif op=='point':actual=float(points[r['metric']][names.index(r['model'])])
            elif op=='delta':actual=float(points[r['metric']][names.index(r['candidate'])]-points[r['metric']][names.index(r['reference'])])
            elif op=='utility':
                a,b=names.index(r['candidate']),names.index(r['reference']);v=points[r['metric']]
                actual=float(100*(v[b]-v[a])/v[b])
            elif op=='interval':
                a,b=names.index(r['candidate']),names.index(r['reference'])
                key=(a,b,r['resampling'],r['metric'],r['confidence'],r['statistic'])
                if key not in cache:
                    v=reps[r['resampling'],r['metric']]
                    delta=v[:,a]-v[:,b] if r['statistic']=='delta' else 100*(v[:,b]-v[:,a])/v[:,b]
                    alpha=(1-r['confidence'])/2
                    cache[key]=np.quantile(delta,[alpha,1-alpha],method='linear')
                    intervals.append(dict(dataset=ds,collection=item['collection'],candidate=r['candidate'],reference=r['reference'],
                        resampling=r['resampling'],metric=r['metric'],statistic=r['statistic'],confidence=r['confidence'],
                        ci_low=float(cache[key][0]),ci_high=float(cache[key][1]),iterations=ITERATIONS,seed=SEED))
                actual=float(cache[key][0 if r['side']=='low' else 1])
            else:raise ValueError(op)
            gap=abs(actual-r['expected'])
            checks.append(dict(dataset=ds,source_file=r['source_file'],source_row=r['source_row'],source_field=r['source_field'],
                expected=r['expected'],actual=actual,absolute_error=gap,passed=gap<=1e-10))
        print(ds,len(d['target_lrp']),'rows;',len(names),'models;',len(req),'checks',flush=True)
    writecsv(out/'metrics.csv',metrics);writecsv(out/'confidence_intervals.csv',intervals);writecsv(out/'comparison_checks.csv',checks)
    writecsv(out/'bootstrap_row_expansion_checks.csv',direct)
    failed=[x for x in checks if not x['passed']]
    summary=dict(passed=not failed,full_package_run=not bool(args.dataset),datasets=len(selected),metric_rows=len(metrics),
        interval_rows=len(intervals),numeric_checks=len(checks),failed_checks=len(failed),
        independent_bootstrap_checks=len(direct),integrity_files=nfiles,absolute_tolerance=1e-10,
        max_absolute_error=max((x['absolute_error'] for x in checks),default=0),
        bootstrap_iterations=ITERATIONS,seed=SEED,seconds=time.time()-start,python=sys.version,numpy=np.__version__,
        scope='Fixed-prediction metric/interval replay. No raw data reconstruction, encoder extraction, model fitting or selection-uncertainty estimation.')
    writej(out/'verification_summary.json',summary);print(json.dumps(summary,indent=2))
    if failed:sys.exit(1)
if __name__=='__main__':main()
