"""Full fixed-grid metadata comparison, chronological cached fits, no final scores.

Only development labels are parsed. Original Ridge results are hash-verified and
reused. Every new learned transform sees its own training prefix only.
"""
import argparse
import gc
import itertools
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import metadata_baseline_pipeline as base

ROOT=base.ROOT
POLICY=ROOT/'revision/Metadata Comparison Execution Specification 20260909.json'
DEFAULT=ROOT/'revision/full_metadata_comparison_20260909'
SOURCES={
    'original_v1': ('target_pipeline_20260909','image_validation_20260909','metadata_baseline_20260909_run2'),
    'exact_one_wei_sensitivity_v1': ('one_wei_sensitivity_targets_20260909','one_wei_sensitivity_targets_20260909','one_wei_metadata_20260909'),
}


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def config_id(config):
    return '_'.join(k+'-'+str(v) for k,v in sorted(config.items()))


def candidates(spec):
    r=spec['regressors']
    return {
        'Ridge':[dict(alpha=a) for a in r['ridge']['alpha']],
        'PLS':[dict(components=k) for k in r['PLS']['components']],
        'LinearSVR':[dict(C=c,epsilon=e) for c,e in itertools.product(r['LinearSVR_primary']['C'],r['LinearSVR_primary']['epsilon'])],
        'ElasticNet':[dict(alpha=a,l1_ratio=l) for a,l in itertools.product(r['ElasticNet']['alpha'],r['ElasticNet']['l1_ratio'])],
        'HistGradientBoosting':[dict(learning_rate=l,max_leaf_nodes=n,l2_regularization=r2) for l,n,r2 in itertools.product(
            r['metadata_nonlinear']['learning_rate'],r['metadata_nonlinear']['max_leaf_nodes'],r['metadata_nonlinear']['l2_regularization'])],
    }


def complexity(family,c):
    if family=='Ridge': return (-c['alpha'],)
    if family=='PLS': return (c['components'],)
    if family=='LinearSVR': return (c['C'],-c['epsilon'])
    if family=='ElasticNet': return (-c['alpha'],-c['l1_ratio'])
    return (c['max_leaf_nodes'],-c['l2_regularization'],c['learning_rate'])


def ordered_valid(family, scores):
    pending=[s for s in scores if s['valid']]
    result=[]
    while pending:
        best=min(s['mse'] for s in pending)
        tied=[s for s in pending if s['mse']<=best+1e-12]
        chosen=min(tied,key=lambda s:complexity(family,s['config']))
        result.append(chosen)
        pending.remove(chosen)
    return result


class Runner:
    def __init__(self,out,threads):
        self.out=out
        self.threads=threads
        self.new_fits=0
        self.started=time.perf_counter()

    def log(self,r):
        r=dict(elapsed_seconds=round(time.perf_counter()-self.started,2),**r)
        with (self.out/'execution_journal.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        if r.get('stage')!='fit_completed' or not r.get('valid',True):
            print(json.dumps(r,ensure_ascii=False),flush=True)

    def memory_gate(self):
        while psutil.virtual_memory().available<1.5*2**30:
            self.log(dict(stage='waiting_for_memory',available_gib=psutil.virtual_memory().available/2**30))
            time.sleep(10)

    def prepare(self,X,train,valid,columns):
        self.memory_gate()
        proto=base.model_pipeline('Ridge',dict(alpha=1))
        encoding=Pipeline(proto.steps[:2])
        raw_train=encoding.fit_transform(X[train])
        raw_valid=encoding.transform(X[valid]) if len(valid) else np.empty((0,raw_train.shape[1]))
        scaler=StandardScaler().fit(raw_train)
        ztrain=scaler.transform(raw_train)
        zvalid=scaler.transform(raw_valid) if len(valid) else raw_valid.copy()
        # Rank of a subset provides a conservative lower bound on training rank.
        selected=np.linspace(0,len(train)-1,min(1024,len(train)),dtype=int)
        sample=ztrain[selected]
        rank=int(np.linalg.matrix_rank(sample-sample.mean(axis=0)))
        return dict(encoding=encoding,scaler=scaler,raw_train=raw_train,raw_valid=raw_valid,
            ztrain=ztrain,zvalid=zvalid,rank=rank,
            unknown=base.unknown_categories(encoding,X[valid],columns) if len(valid) else None)

    def fit(self,family,config,p,ytrain,yvalid,context):
        self.memory_gate()
        started=time.perf_counter()
        info=dict(family=family,config=config,valid=False,training_rank_lower_bound=p['rank'])
        model=None; pred=None
        try:
            if family=='PLS' and config['components']>p['rank']:
                raise ValueError('requested PLS components exceed verified training rank lower bound')
            model=base.model_pipeline(family,config).named_steps['model']
            tr=p['raw_train'] if family=='HistGradientBoosting' else p['ztrain']
            va=p['raw_valid'] if family=='HistGradientBoosting' else p['zvalid']
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(tr,ytrain)
                pred=np.asarray(model.predict(va)).reshape(-1) if len(va) else np.empty(0)
            flags=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            if family=='PLS':
                converged &= len(model.n_iter_)==config['components']
            check_pred=pred if len(va) else np.asarray(model.predict(tr[np.linspace(0,len(tr)-1,min(20,len(tr)),dtype=int)])).reshape(-1)
            finite=np.isfinite(check_pred).all()
            info.update(valid=bool(converged and finite),converged=bool(converged),finite=bool(finite),warnings=flags,
                n_iter=np.asarray(getattr(model,'n_iter_',[])).tolist())
            if info['valid'] and len(yvalid):
                info['sse']=float(np.sum((yvalid-pred)**2))
                info['n']=len(yvalid)
        except Exception as error:
            info['error']=repr(error)
        self.new_fits+=1
        info.update(seconds=round(time.perf_counter()-started,4),process_rss_gib=psutil.Process().memory_info().rss/2**30)
        self.log(dict(stage='fit_completed',**context,**info))
        return model,pred,info

    def persist_model(self,folder,name,family,model,p,train_rowids,config):
        pipe=Pipeline(p['encoding'].steps+[
            ('scale','passthrough' if family=='HistGradientBoosting' else p['scaler']),('model',model)])
        model_path=folder/(name+'.joblib')
        assert not model_path.exists()
        joblib.dump(pipe,model_path,compress=3)
        info=dict(family=family,config=config,model_path=str(model_path.relative_to(ROOT)),model_sha256=base.sha(model_path),
            training_source_rows=[int(i) for i in train_rowids],
            encoding_categories=[v.tolist() for v in p['encoding'].named_steps['onehot'].categories_],
            retained_mask=p['encoding'].named_steps['constant_filter'].get_support().tolist(),
            scaler_mean=None if family=='HistGradientBoosting' else p['scaler'].mean_.tolist(),
            scaler_scale=None if family=='HistGradientBoosting' else p['scaler'].scale_.tolist())
        save(folder/(name+'_provenance.json'),info)
        return info

    def run_collection(self,sample,collection,grids,spec):
        target_name,split_name,ridge_name=SOURCES[sample]
        target=ROOT/'revision'/target_name
        prior=ROOT/'revision'/ridge_name
        folder=self.out/sample/collection
        folder.mkdir(parents=True,exist_ok=True)
        rows=base.lines(target/(collection.lower()+'_development_targets.jsonl'))
        assert all(r['split']=='development' and r['time']<'2025-01-01' for r in rows)
        split=base.read(ROOT/'revision'/split_name/'image_ready_split_manifest_v1.json')[collection]
        meta={(r['collection'],r['token_id']):r for r in base.lines(base.OLD/'metadata_normalized.jsonl')}
        columns=base.FEATURES+(['generation'] if collection=='MAYC' else [])
        X=np.asarray([[meta[collection,r['token_id']][c] for c in columns] for r in rows],dtype=object)
        y=np.asarray([r['y_log_relative_price'] for r in rows])
        rowids=np.asarray([r['source_row'] for r in rows],dtype=np.int64)
        tokens=np.asarray([r['token_id'] for r in rows],dtype=np.int64)
        times=np.asarray([r['time'] for r in rows],dtype=object)
        rowmap={v:i for i,v in enumerate(rowids)}
        assert set(rowmap)==set(split['development_source_rows'])
        del rows,meta
        old_results=base.read(prior/'development_results.json')['collections'][collection]
        old_predictions={r['source_row']:r for r in base.lines(prior/'ridge_development_predictions.jsonl') if r['collection']==collection}
        old_fits=[r for r in base.lines(prior/'fit_log.jsonl') if r['collection']==collection and r['stage']=='inner']
        cache={}; predictions=[]; reports=[]
        for fold in split['folds']:
            context=dict(sample=sample,collection=collection,fold=fold['name'])
            train=np.asarray([rowmap[i] for i in fold['train_source_rows']])
            valid=np.asarray([rowmap[i] for i in fold['validation_source_rows']])
            assert times[train].max()<times[valid].min()
            for q in fold['inner_quarter_schedule']:
                begin,end=q['train_end_exclusive'],q['validation_end_exclusive']
                itr=train[times[train]<begin]
                iva=train[(times[train]>=begin)&(times[train]<end)]
                assert len(itr)==q['train_trades'] and len(iva)==q['validation_trades']
                assert times[itr].max()<times[iva].min()
                if begin in cache:
                    assert cache[begin]['end']==end
                    assert cache[begin]['train_source_rows']==rowids[itr].tolist()
                    assert cache[begin]['validation_source_rows']==rowids[iva].tolist()
                    continue
                p=self.prepare(X,itr,iva,columns)
                scores={}
                for family,configs in grids.items():
                    scores[family]=[]
                    for config in configs:
                        if family=='Ridge':
                            matches=[r for r in old_fits if r['validation_start']==begin and r['alpha']==config['alpha']]
                            assert matches and all(r['train_rows']==len(itr) and r['validation_rows']==len(iva) for r in matches)
                            assert all(math.isclose(r['sse'],matches[0]['sse'],rel_tol=1e-10,abs_tol=1e-10) for r in matches)
                            info=dict(family=family,config=config,valid=all(r['converged'] for r in matches),
                                sse=matches[0]['sse'],n=len(iva),source='hash_verified_prior_Ridge',source_path=str(prior.relative_to(ROOT)))
                        else:
                            _,_,info=self.fit(family,config,p,y[itr],y[iva],dict(**context,role='inner',quarter=begin))
                        scores[family].append(info)
                    self.log(dict(stage='quarter_family_complete',**context,quarter=begin,family=family,
                        configs=len(configs),valid=sum(r['valid'] for r in scores[family])))
                cache[begin]=dict(end=end,scores=scores,train_source_rows=rowids[itr].tolist(),validation_source_rows=rowids[iva].tolist(),unknown=p['unknown'])
                save(folder/('inner_'+begin[:10]+'.json'),cache[begin])
                del p
                gc.collect()
            outer_p=self.prepare(X,train,valid,columns)
            trained_tokens=set(tokens[train])
            output=[dict(collection=collection,sample=sample,fold=fold['name'],source_row=int(rowids[i]),
                token_id=int(tokens[i]),time=str(times[i]),y=float(y[i]),seen_in_fold_training=bool(tokens[i] in trained_tokens),
                prediction_Mean=float(y[train].mean()),prediction_Zero=0.) for i in valid]
            f_report=dict(name=fold['name'],train_rows=len(train),validation_rows=len(valid),unknown=outer_p['unknown'],families={},
                train_target=dict(mean=float(y[train].mean()),median=float(np.median(y[train])),sd=float(y[train].std())),
                validation_target=dict(mean=float(y[valid].mean()),median=float(np.median(y[valid])),sd=float(y[valid].std())))
            for family,configs in grids.items():
                combined=[]
                for j,config in enumerate(configs):
                    parts=[cache[q['train_end_exclusive']]['scores'][family][j] for q in fold['inner_quarter_schedule']]
                    good=all(r['valid'] for r in parts)
                    combined.append(dict(config=config,valid=good,mse=sum(r['sse'] for r in parts)/sum(r['n'] for r in parts) if good else None))
                ranking=ordered_valid(family,combined)
                attempts=[]; chosen=None
                for candidate in ranking:
                    if family=='Ridge':
                        existing=next(r for r in old_results['folds'] if r['fold']==fold['name'])
                        assert candidate['config']['alpha']==existing['selected_alpha']
                        assert all(old_predictions[int(rowids[i])]['y']==y[i] and old_predictions[int(rowids[i])]['fold']==fold['name'] for i in valid)
                        pred=np.asarray([old_predictions[int(rowids[i])]['ridge_prediction'] for i in valid])
                        info=dict(valid=True,reused=True,source_path=str(prior.relative_to(ROOT)))
                        chosen=candidate
                    else:
                        model,pred,info=self.fit(family,candidate['config'],outer_p,y[train],y[valid],dict(**context,role='outer'))
                        attempts.append(info)
                        if not info['valid']:
                            continue
                        chosen=candidate
                        info['saved_model']=self.persist_model(folder,fold['name']+'_'+family,family,model,outer_p,rowids[train],candidate['config'])
                    break
                if chosen is None:
                    f_report['families'][family]=dict(available=False,inner_scores=combined,attempts=attempts)
                    continue
                for row,predicted in zip(output,pred):
                    row['prediction_'+family]=float(predicted)
                f_report['families'][family]=dict(available=True,selected_config=chosen['config'],inner_scores=combined,
                    outer_fit=info,attempts=attempts,metrics=base.metrics(y[valid],pred,tokens[valid]))
                self.log(dict(stage='outer_family_complete',**context,family=family,selected_config=chosen['config'],rmse=f_report['families'][family]['metrics']['rmse']))
            for family in ('Mean','Zero'):
                pred=np.asarray([r['prediction_'+family] for r in output])
                f_report['families'][family]=dict(available=True,metrics=base.metrics(y[valid],pred,tokens[valid]))
            save(folder/(fold['name']+'_results.json'),f_report)
            with (folder/(fold['name']+'_predictions.jsonl')).open('x',encoding='utf-8') as stream:
                for r in output: stream.write(json.dumps(r,allow_nan=False)+'\n')
            predictions.extend(output); reports.append(f_report)
            del outer_p
            gc.collect()
        assert len({r['source_row'] for r in predictions})==len(predictions)
        pooled={}
        for family in list(grids)+['Mean','Zero']:
            if not all('prediction_'+family in r for r in predictions):
                pooled[family]=dict(available=False)
                continue
            pooled[family]=dict(available=True,**base.metrics(np.asarray([r['y'] for r in predictions]),
                np.asarray([r['prediction_'+family] for r in predictions]),np.asarray([r['token_id'] for r in predictions])))
        eligible=[f for f in grids if pooled[f]['available']]
        assert eligible
        best=min(pooled[f]['sse']/pooled[f]['n'] for f in eligible)
        selected=next(f for f in base.read(POLICY)['family_tie_order'] if f in eligible and pooled[f]['sse']/pooled[f]['n']<=best+1e-12)
        result=dict(sample=sample,collection=collection,pooled=pooled,selected_family=selected,
            selection_uses_development_only=True,outer_results_used_for_family_selection_not_independent_final_accuracy=True,
            final_predictions=False,final_metrics=False)
        save(folder/'comparison_results.json',result)
        self.log(dict(stage='family_selected',sample=sample,collection=collection,selected_family=selected,pooled=pooled))
        # Final-preparation hyperparameters: prescribed 2024 quarters; still development only.
        final_scores=[dict(config=c,valid=True,sse=0.,n=0) for c in grids[selected]]
        for begin,end in spec['validation']['final_hyperparameter_tuning_quarters']:
            begin+=' 00:00:00+00:00'; end+=' 00:00:00+00:00'
            tr=np.flatnonzero(times<begin); va=np.flatnonzero((times>=begin)&(times<end))
            assert times[tr].max()<times[va].min()
            p=self.prepare(X,tr,va,columns)
            logs=[]
            for score in final_scores:
                _,_,info=self.fit(selected,score['config'],p,y[tr],y[va],dict(sample=sample,collection=collection,role='final_preparation_inner',quarter=begin))
                logs.append(info); score['valid'] &= info['valid']
                if info['valid']:
                    score['sse']+=info['sse']; score['n']+=info['n']
            save(folder/('final_preparation_'+begin[:10]+'.json'),dict(train_source_rows=rowids[tr].tolist(),validation_source_rows=rowids[va].tolist(),fits=logs))
            self.log(dict(stage='final_preparation_quarter_complete',sample=sample,collection=collection,quarter=begin,family=selected))
            del p
        for s in final_scores: s['mse']=s['sse']/s['n'] if s['valid'] else None
        ranking=ordered_valid(selected,final_scores)
        train=np.arange(len(y)); valid=np.asarray([],dtype=int)
        p=self.prepare(X,train,valid,columns)
        attempts=[]; fitted=None
        for choice in ranking:
            model,_,info=self.fit(selected,choice['config'],p,y,np.empty(0),dict(sample=sample,collection=collection,role='all_development_refit'))
            attempts.append(info)
            if info['valid']:
                fitted=self.persist_model(folder,'selected_all_development',selected,model,p,rowids,choice['config'])
                break
        assert fitted is not None,'No converged selected-family all-development model'
        save(folder/'selected_model_preparation.json',dict(selected_family=selected,model=fitted,inner_scores=final_scores,
            refit_attempts=attempts,final_predictions_computed=False,final_metrics_computed=False,
            provisional_until_common_encoder_rows_verified=True))
        self.log(dict(stage='collection_complete',sample=sample,collection=collection,family=selected,config=fitted['config']))
        del p,X,y
        gc.collect()
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--threads',type=int,default=3)
    args=parser.parse_args()
    out=args.output.resolve()
    assert 1<=args.threads<=3
    spec=base.effective_spec(); policy=base.read(POLICY); grids=candidates(spec)
    assert base.sha(base.OLD/'analysis_specification_v1.json')==policy['base_specification_sha256']
    inputs=[]
    for sample,(target,split,ridge) in SOURCES.items():
        for folder,manifest in ((target,'target_release_v1.json'),(ridge,'baseline_release_v1.json')):
            path=ROOT/'revision'/folder/manifest
            for r in base.read(path)['artifacts']:
                assert base.sha(ROOT/r['path'])==r['sha256'],r['path']
            inputs.append(dict(sample=sample,path=str(path.relative_to(ROOT)),sha256=base.sha(path)))
        release=base.read(ROOT/'revision'/target/'target_release_v1.json')
        assert base.sha(base.OLD/'metadata_normalized.jsonl')==release['metadata_normalized_sha256']
        assert base.sha(ROOT/'revision'/split/'image_ready_split_manifest_v1.json')==release['split_manifest_sha256']
    out.mkdir(exist_ok=False)
    save(out/'run_configuration.json',dict(policy_sha256=base.sha(POLICY),script_sha256=base.sha(Path(__file__)),
        reused_pipeline_sha256=base.sha(Path(base.__file__)),inputs=inputs,grids=grids,threads=args.threads,
        python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,platform=platform.platform(),
        source_data_final_labels_parsed=False,cohort_membership_changed=False,seed=base.SEED))
    runner=Runner(out,args.threads)
    summary=[]
    with threadpool_limits(limits=args.threads):
        for sample in SOURCES:
            for collection in ('BAYC','MAYC'):
                summary.append(runner.run_collection(sample,collection,grids,spec))
    save(out/'comparison_summary.json',dict(results=summary,new_fits=runner.new_fits,
        seconds=round(time.perf_counter()-runner.started,2),final_scored=False,images_used=False))
    artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    save(out/'comparison_release_v1.json',dict(status='FULL_METADATA_DEVELOPMENT_COMPARISON_AND_SELECTED_MODEL_PREPARATION_COMPLETE',
        artifacts=artifacts,policy_sha256=base.sha(POLICY),final_scored=False,encoder_common_rows_pending=True))
    print(json.dumps(dict(stage='all_complete',new_fits=runner.new_fits,seconds=time.perf_counter()-runner.started)),flush=True)


if __name__=='__main__':
    main()
