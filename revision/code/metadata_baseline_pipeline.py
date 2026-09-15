"""Metadata pipelines and bounded nested-development Ridge run.

Reads ONLY development label files. No CLI path exists for final-period labels.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'revision/data_audit_20260908'
COHORT=ROOT/'revision/image_validation_20260909'
TARGET=ROOT/'revision/target_pipeline_20260909'
AMENDMENT=ROOT/'revision/Analysis Amendment 20260909.json'
OUT=ROOT/'revision/metadata_baseline_20260909'
FEATURES=['background','fur','eyes','clothes','hat','mouth','earring']
SEED=20260908


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(s) for s in stream if s.strip()]


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def journal(record):
    with (OUT/'execution_journal.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n')


def effective_spec():
    change=read(AMENDMENT)
    assert sha(OLD/'analysis_specification_v1.json')==change['base_specification_sha256']
    spec=read(OLD/'analysis_specification_v1.json')
    for patch in change['patches']:
        assert patch['op']=='replace' and patch['path']=='/regressors/ridge/alpha'
        spec['regressors']['ridge']['alpha']=patch['value']
    return spec


def model_pipeline(family,config):
    if family=='Ridge':
        model=Ridge(alpha=config['alpha'],solver='cholesky',fit_intercept=True)
    elif family=='PLS':
        model=PLSRegression(n_components=config['components'],scale=False,max_iter=500,tol=1e-6)
    elif family=='LinearSVR':
        model=LinearSVR(C=config['C'],epsilon=config['epsilon'],loss='squared_epsilon_insensitive',
                        dual=False,tol=1e-4,max_iter=5000,random_state=SEED)
    elif family=='ElasticNet':
        model=ElasticNet(alpha=config['alpha'],l1_ratio=config['l1_ratio'],
                         max_iter=20000,tol=1e-4,random_state=SEED,selection='cyclic')
    elif family=='HistGradientBoosting':
        model=HistGradientBoostingRegressor(learning_rate=config['learning_rate'],
            max_leaf_nodes=config['max_leaf_nodes'],l2_regularization=config['l2_regularization'],
            max_iter=300,early_stopping=False,random_state=SEED)
    elif family=='Mean':
        model=DummyRegressor(strategy='mean')
    else:
        raise ValueError(family)
    return Pipeline([
        ('onehot',OneHotEncoder(handle_unknown='ignore',drop=None,sparse=False,dtype=np.float64)),
        ('constant_filter',VarianceThreshold(threshold=0)),
        ('scale','passthrough' if family=='HistGradientBoosting' else StandardScaler()),
        ('model',model)])


def unknown_categories(pipe,X,columns):
    ohe=pipe.named_steps['onehot']
    result={}
    any_unknown=np.zeros(len(X),dtype=bool)
    for i,(column,known) in enumerate(zip(columns,ohe.categories_)):
        mask=~np.isin(X[:,i],known)
        any_unknown|=mask
        result[column]=dict(rows=int(mask.sum()),values=sorted(set(X[mask,i].tolist())))
    return dict(any_unknown_rows=int(any_unknown.sum()),by_category=result)


def fit_predict(family,config,X,y,train,valid):
    assert np.intersect1d(train,valid).size==0
    pipe=model_pipeline(family,config)
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipe.fit(X[train],y[train])
        pred=np.asarray(pipe.predict(X[valid])).reshape(-1)
    flags=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    assert np.isfinite(pred).all()
    info=dict(seconds=round(time.perf_counter()-started,4),warnings=flags,converged=converged,
        onehot_dimensions=len(pipe.named_steps['onehot'].get_feature_names_out()),
        nonconstant_dimensions=int(pipe.named_steps['constant_filter'].get_support().sum()))
    return pipe,pred,info


def metrics(y,pred,tokens):
    residual=y-pred
    sse=float(np.dot(residual,residual))
    sst=float(np.dot(y-y.mean(),y-y.mean()))
    _,inverse=np.unique(tokens,return_inverse=True)
    token_mse=np.bincount(inverse,weights=residual**2)/np.bincount(inverse)
    return dict(n=len(y),sse=sse,sst=sst,rmse=float(np.sqrt(sse/len(y))),
        mae=float(np.mean(np.abs(residual))),r2=None if sst==0 else 1-sse/sst,
        equal_token_rmse=float(np.sqrt(token_mse.mean())))


def unit_tests():
    X=np.array([['a','None'],['b','None'],['a','None'],['b','None']],dtype=object)
    y=np.array([0.,1.,0.,1.])
    pipe=model_pipeline('Ridge',dict(alpha=1))
    pipe.fit(X,y)
    before=[c.copy() for c in pipe.named_steps['onehot'].categories_]
    pipe.predict(np.array([['new','None']],dtype=object))
    assert all(np.array_equal(a,b) for a,b in zip(before,pipe.named_steps['onehot'].categories_))
    assert 'new' not in before[0]
    assert pipe.named_steps['constant_filter'].get_support().sum()==2
    assert np.allclose(pipe.named_steps['scale'].mean_,[.5,.5])
    return dict(unseen_category_not_fitted=True,training_only_scaler=True,constant_column_removed=True,
                duplicate_token_rows_not_collapsed=True)


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output
    spec=effective_spec()
    release=read(TARGET/'target_release_v1.json')
    # Verify hashes without interpreting final label values. No final file is parsed.
    for r in release['artifacts']:
        assert sha(ROOT/r['path'])==r['sha256']
    assert sha(OLD/'metadata_normalized.jsonl')==release['metadata_normalized_sha256']
    assert sha(COHORT/'image_ready_split_manifest_v1.json')==release['split_manifest_sha256']
    OUT.mkdir(exist_ok=False)
    save('run_configuration.json',dict(stage='nested_development_Ridge_and_pipeline_smoke',
        base_specification_sha256=sha(OLD/'analysis_specification_v1.json'),amendment_sha256=sha(AMENDMENT),
        target_release_sha256=sha(TARGET/'target_release_v1.json'),
        ridge_alpha=spec['regressors']['ridge']['alpha'],cpu_threads=6,
        python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,platform=platform.platform(),
        target_scaling='none',mean_fit_weight='one per transaction',
        final_labels_parsed=False,final_metrics_computed=False,
        full_grids_for_other_models_executed=False,script_sha256=sha(Path(__file__))))
    tests=unit_tests()
    metadata=lines(OLD/'metadata_normalized.jsonl')
    meta={(r['collection'],r['token_id']):r for r in metadata}
    splits=read(COHORT/'image_ready_split_manifest_v1.json')
    definitions=read(OLD/'trade_audit.json')
    smoke=[]; fit_log=[]; summaries={}; all_predictions=[]
    started=time.perf_counter()
    with threadpool_limits(limits=6):
        for collection in ('BAYC','MAYC'):
            rows=lines(TARGET/(collection.lower()+'_development_targets.jsonl'))
            assert all(r['split']=='development' and r['time']<'2025-01-01 00:00:00+00:00' for r in rows)
            columns=FEATURES+(['generation'] if collection=='MAYC' else [])
            X=np.asarray([[meta[collection,r['token_id']][c] for c in columns] for r in rows],dtype=object)
            y=np.asarray([r['y_log_relative_price'] for r in rows],dtype=np.float64)
            rowids=np.asarray([r['source_row'] for r in rows]); tokens=np.asarray([r['token_id'] for r in rows])
            times=np.asarray([r['time'] for r in rows],dtype=object); rowmap={r:i for i,r in enumerate(rowids)}
            assert set(rowids)==set(splits[collection]['development_source_rows'])
            assert all(isinstance(v,str) and v!='' for v in X.ravel())
            first=splits[collection]['folds'][0]
            smoke_train=np.array([rowmap[r] for r in first['train_source_rows']][:4096])
            smoke_valid=np.array([rowmap[r] for r in first['validation_source_rows']][:256])
            assert times[smoke_train].max()<times[smoke_valid].min()
            configs=[('Ridge',dict(alpha=1)),('PLS',dict(components=8)),
                ('LinearSVR',dict(C=.01,epsilon=.1)),('ElasticNet',dict(alpha=.01,l1_ratio=.5)),
                ('HistGradientBoosting',dict(learning_rate=.05,max_leaf_nodes=15,l2_regularization=1)),('Mean',{})]
            for family,config in configs:
                pipe,pred,info=fit_predict(family,config,X,y,smoke_train,smoke_valid)
                result=dict(collection=collection,family=family,config=config,
                    train_rows=len(smoke_train),check_rows=len(smoke_valid),**info,
                    finite_prediction_check=True,accuracy_scored=False)
                smoke.append(result)
                journal(dict(stage='smoke',**result))
                print(json.dumps(dict(stage='smoke',**result)),flush=True)
                del pipe,pred
            if args.smoke_only:
                continue
            collection_predictions=[]; fold_summaries=[]
            for fold,definition in zip(splits[collection]['folds'],definitions[collection]['folds']):
                train=np.asarray([rowmap[r] for r in fold['train_source_rows']])
                valid=np.asarray([rowmap[r] for r in fold['validation_source_rows']])
                assert times[train].max()<times[valid].min()
                scores={a:dict(sse=0.,n=0,valid=True) for a in spec['regressors']['ridge']['alpha']}
                inner_logs=[]
                for q in fold['inner_quarter_schedule']:
                    begin,end=q['train_end_exclusive'],q['validation_end_exclusive']
                    itr=train[times[train]<begin]
                    iva=train[(times[train]>=begin)&(times[train]<end)]
                    assert len(itr)==q['train_trades'] and len(iva)==q['validation_trades']
                    assert times[itr].max()<times[iva].min()
                    for alpha in spec['regressors']['ridge']['alpha']:
                        pipe,pred,info=fit_predict('Ridge',dict(alpha=alpha),X,y,itr,iva)
                        sse=float(np.sum((y[iva]-pred)**2))
                        scores[alpha]['sse']+=sse; scores[alpha]['n']+=len(iva)
                        scores[alpha]['valid'] &= info['converged']
                        log=dict(collection=collection,outer_fold=fold['name'],stage='inner',
                            validation_start=begin,validation_end_exclusive=end,alpha=alpha,
                            train_rows=len(itr),validation_rows=len(iva),sse=sse,
                            unknown=unknown_categories(pipe,X[iva],columns),**info)
                        fit_log.append(log); inner_logs.append(log)
                        journal(log)
                        del pipe,pred
                    print(json.dumps(dict(stage='inner_quarter_completed',collection=collection,
                        fold=fold['name'],validation_start=begin,seconds=round(time.perf_counter()-started,2))),flush=True)
                allowed={a:d['sse']/d['n'] for a,d in scores.items() if d['valid']}
                assert allowed
                best_mse=min(allowed.values())
                selected=max(a for a,mse in allowed.items() if mse<=best_mse+1e-12)
                pipe,pred,info=fit_predict('Ridge',dict(alpha=selected),X,y,train,valid)
                assert info['converged']
                unknown=unknown_categories(pipe,X[valid],columns)
                mean=np.full(len(valid),y[train].mean()); zero=np.zeros(len(valid))
                report=dict(collection=collection,fold=fold['name'],
                    selected_alpha=selected,selected_at_grid_boundary=selected in (min(allowed),max(allowed)),
                    pooled_inner_mse={str(a):v for a,v in allowed.items()},
                    ridge=metrics(y[valid],pred,tokens[valid]),
                    training_mean_control=metrics(y[valid],mean,tokens[valid]),
                    prior_median_control=metrics(y[valid],zero,tokens[valid]),unknown=unknown,**info)
                fold_summaries.append(report)
                fit_log.append(dict(collection=collection,outer_fold=fold['name'],stage='outer_refit',
                                    alpha=selected,train_rows=len(train),validation_rows=len(valid),**info))
                journal(fit_log[-1])
                for j,index in enumerate(valid):
                    collection_predictions.append(dict(collection=collection,fold=fold['name'],
                        source_row=int(rowids[index]),token_id=int(tokens[index]),time=str(times[index]),
                        y=float(y[index]),ridge_prediction=float(pred[j]),
                        training_mean_prediction=float(mean[j]),prior_median_prediction=0.))
                save(collection.lower()+'_'+fold['name']+'_fitted_ridge_parameters.json',dict(
                    selected_alpha=selected,columns=columns,
                    categories=[v.tolist() for v in pipe.named_steps['onehot'].categories_],
                    retained_onehot_mask=pipe.named_steps['constant_filter'].get_support().tolist(),
                    feature_mean=pipe.named_steps['scale'].mean_.tolist(),feature_scale=pipe.named_steps['scale'].scale_.tolist(),
                    coefficients=pipe.named_steps['model'].coef_.tolist(),intercept=float(pipe.named_steps['model'].intercept_),
                    fit_source_row_hash=hashlib.sha256(rowids[train].tobytes()).hexdigest(),
                    note='development-fold model, not final fitted model'))
                print(json.dumps(dict(stage='outer_completed',collection=collection,fold=fold['name'],
                                      selected_alpha=selected,development_rmse=report['ridge']['rmse'])),flush=True)
                del pipe,pred
            assert len({r['source_row'] for r in collection_predictions})==len(collection_predictions)
            vals=np.asarray([r['y'] for r in collection_predictions])
            tids=np.asarray([r['token_id'] for r in collection_predictions])
            pooled={key:metrics(vals,np.asarray([r[key] for r in collection_predictions]),tids)
                    for key in ('ridge_prediction','training_mean_prediction','prior_median_prediction')}
            summaries[collection]=dict(folds=fold_summaries,pooled_nonoverlapping_development=pooled)
            all_predictions.extend(collection_predictions)
    save('pipeline_smoke_results.json',dict(tests=tests,models=smoke,final_data_used=False))
    with (OUT/'fit_log.jsonl').open('x',encoding='utf-8') as stream:
        for r in fit_log:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (OUT/'ridge_development_predictions.jsonl').open('x',encoding='utf-8') as stream:
        for r in all_predictions:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    save('development_results.json',dict(collections=summaries,seconds=round(time.perf_counter()-started,2),
        development_only=True,final_test_predictions=False,final_test_metrics=False,
        strong_metadata_model_selection_complete=False,images_compared=False,
        statistical_significance_tested=False,full_ridge_fits=len(fit_log),smoke_fits=len(smoke)))
    artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(OUT.iterdir())]
    save('baseline_release_v1.json',dict(status='INITIAL_RIDGE_DEVELOPMENT_COMPLETE; OTHER_FAMILY_GRIDS_PENDING',
        artifacts=artifacts,amendment_sha256=sha(AMENDMENT),target_release_sha256=sha(TARGET/'target_release_v1.json')))
    print(json.dumps(dict(status='development_stage_complete',full_ridge_fits=len(fit_log),smoke_fits=len(smoke),
                          final_scored=False,seconds=round(time.perf_counter()-started,2))),flush=True)


if __name__=='__main__':
    main()
