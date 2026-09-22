"""DINOv2 Full-Frame image/early/late development comparison; never load final labels."""
import argparse
import gc
import json
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

from counted_primal_svr_v3 import CountedPrimalSVR
import full_metadata_comparison as metadata
base = metadata.base
ROOT = base.ROOT
REV = ROOT / 'revision'
POLICY = REV / 'Encoder Development Execution Specification 20260909 v4.json'
REGISTRY = REV / 'Encoder Development Registry 20260909 v4.json'
REFERENCE = REV / 'siglip2_development_20260909_v2'
ENCODER = 'dinov2_fullframe'
FEATURE = REV / 'siglip2_features_20260909'
META = REV / 'full_metadata_comparison_20260909'
DEFAULT = REV / 'dinov2_fullframe_development_20260909_v4'
FAMILIES = ['Ridge', 'ElasticNet', 'PLS', 'LinearSVR']
MFAMILIES = FAMILIES + ['HistGradientBoosting']
save = metadata.save


def load_data(sample, collection):
    target, split, _ = metadata.SOURCES[sample]
    rows = base.lines(REV / target / (collection.lower() + '_development_targets.jsonl'))
    assert all(r['split'] == 'development' and r['time'] < '2025-01-01' for r in rows)
    info = base.read(REGISTRY)[ENCODER]['collections'][collection]
    manifest = base.lines(ROOT / info['manifest'])
    image = np.load(ROOT / info['matrix'], allow_pickle=False).astype(np.float64)
    assert image.shape[1] == info['dimensions']
    ids = np.empty(len(manifest), dtype=np.int64)
    for r in manifest:
        ids[r['feature_row_idx']] = r['token_id']
    tokenmap = {v: i for i, v in enumerate(ids)}
    met = {(r['collection'], r['token_id']): r for r in base.lines(base.OLD / 'metadata_normalized.jsonl')}
    cols = base.FEATURES + (['generation'] if collection == 'MAYC' else [])
    X = np.asarray([[met[collection, int(t)][c] for c in cols] for t in ids], dtype=object)
    data = dict(X=X, image=image, feature_tokens=ids, columns=cols,
                y=np.asarray([r['y_log_relative_price'] for r in rows]),
                tokens=np.asarray([r['token_id'] for r in rows], dtype=np.int64),
                rowids=np.asarray([r['source_row'] for r in rows], dtype=np.int64),
                times=np.asarray([r['time'] for r in rows], dtype=object),
                index=np.asarray([tokenmap[r['token_id']] for r in rows], dtype=np.int64),
                split=base.read(REV / split / 'image_ready_split_manifest_v1.json')[collection])
    data['rowmap'] = {v: i for i, v in enumerate(data['rowids'])}
    assert set(data['rowmap']) == set(data['split']['development_source_rows'])
    return data


def prepare(data, train, valid, mode):
    ti, vi = data['index'][train], data['index'][valid]
    unique, inverse, counts = np.unique(ti, return_inverse=True, return_counts=True)
    y = data['y'][train]
    ym = np.bincount(inverse, weights=y) / counts
    encoding = Pipeline(base.model_pipeline('Ridge', dict(alpha=1)).steps[:2])
    raw = encoding.fit_transform(data['X'][unique])
    ms = StandardScaler().fit(raw, sample_weight=counts)
    image_mask = np.ptp(data['image'][unique], axis=0) > 0
    ims = StandardScaler().fit(data['image'][unique][:, image_mask], sample_weight=counts)
    zi = ims.transform(data['image'][:, image_mask])
    if mode == 'early':
        zm = ms.transform(encoding.transform(data['X']))
        z = np.concatenate([zm, zi], axis=1)
    else:
        z = zi
    zmean = np.average(z[unique], axis=0, weights=counts)
    selected = unique[np.linspace(0, len(unique)-1, min(256, len(unique)), dtype=int)]
    subset = z[selected]
    rank = int(np.linalg.matrix_rank(subset - subset.mean(axis=0)))
    state = dict(mode=mode, encoding=encoding, metadata_scaler=ms, image_mask=image_mask,
                 image_scaler=ims, metadata_columns=data['columns'])
    return dict(state=state, z=z, unique=unique, inverse=inverse, counts=counts, ym=ym,
                ymean=float(y.mean()), zmean=zmean, ti=ti, vi=vi, y=y, rank=rank,
                full=None, gram=None, signed=None, signed_y=None)


def transform(state, X, image):
    zi = state['image_scaler'].transform(image[:, state['image_mask']])
    if state['mode'] == 'image':
        return zi
    zm = state['metadata_scaler'].transform(state['encoding'].transform(X))
    return np.concatenate([zm, zi], axis=1)


def model_predict(model, z, offset=None):
    if offset is None:
        return np.asarray(model.predict(z)).reshape(-1)
    return np.asarray(model.predict(z - offset['x'])).reshape(-1) + offset['y']


def fit_prepared(p, family, config):
    model = base.model_pipeline(family, config).named_steps['model']
    offset = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if family == 'Ridge':
            model.fit(p['z'][p['unique']], p['ym'], sample_weight=p['counts'])
        elif family == 'PLS':
            assert p['rank'] >= config['components'], 'PLS training rank bound too small'
            if p['signed'] is None:
                a = (p['z'][p['unique']] - p['zmean']) * np.sqrt(p['counts'] / 2)[:, None]
                b = (p['ym'] - p['ymean']) * np.sqrt(p['counts'] / 2)
                p['signed'] = np.concatenate([a, -a])
                p['signed_y'] = np.concatenate([b, -b])
            model.fit(p['signed'], p['signed_y'])
            offset = dict(x=p['zmean'], y=p['ymean'])
        elif family == 'LinearSVR':
            model = CountedPrimalSVR(config['C'], config['epsilon']).fit_counts(p['z'][p['unique']], p['inverse'], p['y'])
        else:
            if p['full'] is None:
                p['full'] = np.asfortranarray(p['z'][p['ti']])
            if family == 'ElasticNet':
                if p['gram'] is None:
                    center = p['full'].mean(axis=0)
                    p['gram'] = np.ascontiguousarray(p['full'].T @ p['full'] - len(p['full']) * np.outer(center, center))
                model.set_params(precompute=p['gram'])
            model.fit(p['full'], p['y'])
        pred = model_predict(model, p['z'][p['vi']], offset)
    flags = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
    valid = np.isfinite(pred).all() and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if family == 'PLS':
        valid = valid and len(model.n_iter_) == config['components'] and np.isfinite(model.coef_).all()
    info = dict(valid=bool(valid), warnings=flags, n_iter=np.asarray(getattr(model, 'n_iter_', [])).tolist())
    if family == 'LinearSVR':
        info.update(primal_objective=model.objective_, primal_gradient_norm=model.gradient_norm_,
                    objective_gap_upper_bound=model.objective_gap_upper_bound_)
    return model, offset, pred, info


def prediction_key(mode, family, number):
    return mode + '_' + family + '_' + str(number)


def combine_scores(cache, schedule, mode, family, configs):
    scores = []
    for i, c in enumerate(configs):
        parts = [cache[q['train_end_exclusive']]['scores'][prediction_key(mode, family, i)] for q in schedule]
        good = all(p['valid'] for p in parts)
        scores.append(dict(config=c, number=i, valid=good,
                           mse=sum(p['sse'] for p in parts)/sum(p['n'] for p in parts) if good else None))
    return scores


def late_weights(parts, weights):
    scores = []
    for w in weights:
        sse = sum(float(np.sum((y - ((1-w)*m + w*i))**2)) for y,m,i in parts)
        n = sum(len(y) for y,m,i in parts)
        scores.append(dict(weight=w, sse=sse, n=n, mse=sse/n))
    best = min(s['mse'] for s in scores)
    selected = min(s['weight'] for s in scores if s['mse'] <= best + 1e-12)
    return selected, scores


class Runner(metadata.Runner):
    def fit_image(self, p, family, config, yvalid, context):
        self.memory_gate()
        started = time.perf_counter()
        model = offset = pred = None
        info = dict(valid=False)
        try:
            model, offset, pred, info = fit_prepared(p, family, config)
            if info['valid']:
                info.update(sse=float(np.sum((yvalid-pred)**2)), n=len(yvalid))
        except Exception as error:
            info['error'] = repr(error)
        self.new_fits += 1
        info.update(family=family, config=config, seconds=round(time.perf_counter()-started,4),
                    process_rss_gib=psutil.Process().memory_info().rss/2**30,
                    training_rank_lower_bound=p['rank'])
        self.log(dict(stage='fit_completed', **context, **info))
        return model, offset, pred, info

    def collection(self, sample, collection, grids, policy):
        data = load_data(sample, collection)
        folder = self.out / sample / collection
        folder.mkdir(parents=True, exist_ok=True)
        prior = META / sample / collection
        rowmap, times, rowids, y = data['rowmap'], data['times'], data['rowids'], data['y']
        cache, predictions, folds = {}, [], []
        # Determine all metadata inner refits from already frozen inner-selected settings.
        needed = {}
        oldfolds = {}
        for fold in data['split']['folds']:
            old = base.read(prior / (fold['name'] + '_results.json'))
            oldfolds[fold['name']] = old
            for q in fold['inner_quarter_schedule']:
                need = needed.setdefault(q['train_end_exclusive'], {})
                for f in MFAMILIES:
                    if old['families'][f]['available']:
                        c = old['families'][f]['selected_config']
                        need[f + '_' + metadata.config_id(c)] = (f,c)
        for fold in data['split']['folds']:
            context = dict(sample=sample, collection=collection, fold=fold['name'])
            if (folder / (fold['name'] + '_results.json')).exists():
                report = base.read(folder / (fold['name'] + '_results.json'))
                predictions.extend(base.lines(folder / (fold['name'] + '_predictions.jsonl')))
                folds.append(report)
                self.log(dict(stage='resume_completed_fold', **context))
                continue
            train = np.asarray([rowmap[v] for v in fold['train_source_rows']])
            valid = np.asarray([rowmap[v] for v in fold['validation_source_rows']])
            assert times[train].max() < times[valid].min()
            for q in fold['inner_quarter_schedule']:
                begin, end = q['train_end_exclusive'], q['validation_end_exclusive']
                tr = train[times[train] < begin]
                va = train[(times[train] >= begin) & (times[train] < end)]
                assert len(tr) == q['train_trades'] and len(va) == q['validation_trades']
                assert times[tr].max() < times[va].min()
                if begin in cache:
                    assert cache[begin]['train_source_rows'] == rowids[tr].tolist()
                    assert cache[begin]['validation_source_rows'] == rowids[va].tolist()
                    continue
                checkpoint = folder / ('inner_' + begin[:10] + '.json')
                if checkpoint.exists():
                    pack = base.read(checkpoint)
                    assert pack['train_source_rows'] == rowids[tr].tolist()
                    assert pack['validation_source_rows'] == rowids[va].tolist()
                    with np.load(folder / ('inner_' + begin[:10] + '_predictions.npz'), allow_pickle=False) as archived:
                        pack['arrays'] = {k: archived[k] for k in archived.files}
                    cache[begin] = pack
                    self.log(dict(stage='resume_completed_quarter', **context, quarter=begin))
                    continue
                arrays, scores = {}, {}
                reference_folder = REFERENCE / sample / collection
                refpack = base.read(reference_folder / ('inner_' + begin[:10] + '.json'))
                assert refpack['train_source_rows'] == rowids[tr].tolist()
                assert refpack['validation_source_rows'] == rowids[va].tolist()
                with np.load(reference_folder / ('inner_' + begin[:10] + '_predictions.npz'), allow_pickle=False) as archive:
                    refarrays = {k: archive[k] for k in archive.files}
                for mode in ('image', 'early'):
                    p = prepare(data, tr, va, mode)
                    for family in FAMILIES:
                        for j,c in enumerate(grids[family]):
                            key = prediction_key(mode, family, j)
                            if ENCODER == 'siglip2' and family != 'LinearSVR':
                                info = dict(refpack['scores'][key], reused=True, source=str(reference_folder.relative_to(ROOT)))
                                assert info['config'] == c
                                pred = refarrays.get(key)
                                self.log(dict(stage='fit_reused', **context, role='inner', quarter=begin, mode=mode, **info))
                            else:
                                _,_,pred,info = self.fit_image(p, family, c, y[va], dict(**context, role='inner', quarter=begin, mode=mode))
                            scores[key] = info
                            if info['valid']:
                                arrays[key] = pred
                        self.log(dict(stage='inner_family_complete', **context, quarter=begin, mode=mode, family=family))
                    del p
                    gc.collect()
                # Only component configurations actually needed for late weight tuning.
                for key,(family,c) in needed[begin].items():
                    cachekey = 'meta_' + key
                    info = dict(refpack['scores'][cachekey], reused=True, source=str(reference_folder.relative_to(ROOT)))
                    assert info['config'] == c
                    scores[cachekey] = info
                    if info['valid']:
                        arrays[cachekey] = refarrays[cachekey]
                    self.log(dict(stage='fit_reused', **context, role='late_metadata_inner_replay', quarter=begin, **info))
                del refarrays
                pack = dict(end=end, train_source_rows=rowids[tr].tolist(), validation_source_rows=rowids[va].tolist(), scores=scores)
                np.savez_compressed(folder / ('inner_' + begin[:10] + '_predictions.npz'), **arrays)
                save(folder / ('inner_' + begin[:10] + '.json'), pack)
                pack['arrays'] = arrays
                cache[begin] = pack
                self.log(dict(stage='inner_quarter_complete', **context, quarter=begin))
            oldpred = base.lines(prior / (fold['name'] + '_predictions.jsonl'))
            assert [r['source_row'] for r in oldpred] == rowids[valid].tolist()
            assert np.array_equal([r['y'] for r in oldpred], y[valid])
            output = [dict(r) for r in oldpred]
            selected, report = {}, dict(name=fold['name'], candidates={}, train_source_rows=rowids[train].tolist(), validation_source_rows=rowids[valid].tolist())
            for mode in ('image', 'early'):
                p = prepare(data, train, valid, mode)
                for family in FAMILIES:
                    scores = combine_scores(cache, fold['inner_quarter_schedule'], mode, family, grids[family])
                    attempts, chosen = [], None
                    for choice in metadata.ordered_valid(family, scores):
                        if ENCODER == 'siglip2' and family != 'LinearSVR':
                            reference_folder = REFERENCE / sample / collection
                            reference_candidate = base.read(reference_folder / (fold['name'] + '_results.json'))['candidates'][mode + '_' + family]
                            assert reference_candidate['selected_config'] == choice['config']
                            old_bundle = joblib.load(ROOT / reference_candidate['model_path'])
                            model, offset = old_bundle['model'], old_bundle['offset']
                            old_rows = base.lines(reference_folder / (fold['name'] + '_predictions.jsonl'))
                            assert [r['source_row'] for r in old_rows] == rowids[valid].tolist()
                            pred = np.asarray([r['prediction_' + mode + '_' + family] for r in old_rows])
                            info = dict(reference_candidate['attempts'][-1], reused=True, source=str(reference_folder.relative_to(ROOT)))
                            self.log(dict(stage='fit_reused', **context, role='outer', mode=mode, **info))
                        else:
                            model, offset, pred, info = self.fit_image(p, family, choice['config'], y[valid], dict(**context, role='outer', mode=mode))
                        attempts.append(info)
                        if info['valid']:
                            chosen = choice
                            break
                    name = mode + '_' + family
                    if chosen is None:
                        report['candidates'][name] = dict(available=False, inner_scores=scores, attempts=attempts)
                        continue
                    selected[name] = chosen
                    bundle = dict(state=p['state'], model=model, offset=offset, family=family, config=chosen['config'])
                    modelpath = folder / (fold['name'] + '_' + name + '.joblib')
                    joblib.dump(bundle, modelpath, compress=3)
                    report['candidates'][name] = dict(available=True, selected_config=chosen['config'], inner_scores=scores,
                        attempts=attempts, model_path=str(modelpath.relative_to(ROOT)), model_sha256=base.sha(modelpath),
                        dimensions=p['z'].shape[1], metrics=base.metrics(y[valid], pred, data['tokens'][valid]))
                    for r,v in zip(output,pred):
                        r['prediction_' + name] = float(v)
                    self.log(dict(stage='outer_candidate_complete', **context, candidate=name, config=chosen['config']))
                del p
                gc.collect()
            for mf in MFAMILIES:
                old = oldfolds[fold['name']]['families'][mf]
                for imf in FAMILIES:
                    name = 'late_' + mf + '_' + imf
                    ikey = 'image_' + imf
                    if not old['available'] or ikey not in selected:
                        report['candidates'][name] = dict(available=False)
                        continue
                    mc = old['selected_config']
                    ic = selected[ikey]
                    parts = []
                    for q in fold['inner_quarter_schedule']:
                        cached = cache[q['train_end_exclusive']]
                        va = np.asarray([rowmap[v] for v in cached['validation_source_rows']])
                        arrays = cached['arrays']
                        parts.append((y[va], arrays['meta_' + mf + '_' + metadata.config_id(mc)],
                                      arrays[prediction_key('image', imf, ic['number'])]))
                    weight, wscores = late_weights(parts, policy['late_weights'])
                    pred = np.asarray([(1-weight)*r['prediction_' + mf] + weight*r['prediction_' + ikey] for r in output])
                    for r,v in zip(output,pred):
                        r['prediction_' + name] = float(v)
                    report['candidates'][name] = dict(available=True, metadata_config=mc, image_config=ic['config'],
                        image_weight=weight, weight_scores=wscores, metrics=base.metrics(y[valid],pred,data['tokens'][valid]))
            with (folder / (fold['name'] + '_predictions.jsonl')).open('x', encoding='utf-8') as stream:
                for r in output:
                    stream.write(json.dumps(r, allow_nan=False) + '\n')
            save(folder / (fold['name'] + '_results.json'), report)
            predictions.extend(output)
            folds.append(report)
            self.log(dict(stage='outer_fold_complete', **context))
        assert len({r['source_row'] for r in predictions}) == len(predictions)
        keys = [f for f in MFAMILIES] + ['Zero','Mean'] + ['image_'+f for f in FAMILIES] + ['early_'+f for f in FAMILIES]
        augmented = ['late_'+m+'_'+i for m in MFAMILIES for i in FAMILIES] + ['early_'+f for f in FAMILIES]
        keys += [k for k in augmented if k.startswith('late_')]
        pooled = {}
        for k in keys:
            if not all('prediction_'+k in r for r in predictions):
                pooled[k] = dict(available=False)
            else:
                pooled[k] = dict(available=True, **base.metrics(np.asarray([r['y'] for r in predictions]),
                    np.asarray([r['prediction_'+k] for r in predictions]), np.asarray([r['token_id'] for r in predictions])))
        def choose(order):
            eligible = [k for k in order if pooled[k]['available']]
            best = min(pooled[k]['sse']/pooled[k]['n'] for k in eligible)
            return next(k for k in eligible if pooled[k]['sse']/pooled[k]['n'] <= best+1e-12)
        mstar = base.read(prior / 'comparison_results.json')['selected_family']
        astar = choose(augmented)
        result = dict(sample=sample, collection=collection, pooled=pooled, selected_metadata=mstar,
            selected_augmented=astar, selected_image=choose(['image_'+f for f in FAMILIES]),
            development_utility_percent=100*(1-pooled[astar]['rmse']/pooled[mstar]['rmse']),
            provisional_encoder_specific_selection=True, final_scored=False)
        save(folder / 'comparison_results.json',result)
        self.log(dict(stage='collection_complete', sample=sample, collection=collection, selected=astar, utility=result['development_utility_percent']))
        return result


def configure_encoder(name):
    global ENCODER, DEFAULT
    assert name in base.read(REGISTRY)
    ENCODER = name
    DEFAULT = REV / (name + '_development_20260909_v4')


def verify_inputs():
    paths = [POLICY, REGISTRY, REV/'fullframe_handoff_review_20260909/acceptance_review.json', REV/'dinov2_fullframe_features_20260909/spec_and_registry.json', REV/'SVR Stable Line Search Amendment 20260909.json',
             REV/'code/counted_primal_svr_v3.py', REV/'svr_stable_line_search_verification_20260909/verification.json',
             REV/'Analysis Amendment 20260909.json', base.OLD/'analysis_specification_v1.json',
             REV/'Metadata Comparison Execution Specification 20260909.json']
    feature = base.read(REGISTRY)[ENCODER]
    assert base.sha(ROOT/feature['source_summary']) == feature['source_summary_sha256']
    paths.append(ROOT/feature['source_summary'])
    master = {(r['collection'],r['token_id']): r for r in base.lines(base.COHORT/'master_tokens_v1.jsonl')}
    for collection,info in feature['collections'].items():
        for field in ('matrix','manifest'):
            assert base.sha(ROOT/info[field]) == info[field+'_sha256']
            paths.append(ROOT/info[field])
        records = base.lines(ROOT/info['manifest'])
        expected = {t for c,t in master if c == collection}
        assert len(records) == len(expected) and {r['token_id'] for r in records} == expected
        assert {r['feature_row_idx'] for r in records} == set(range(len(records)))
        for r in records:
            m=master[collection,r['token_id']]
            assert r['collection']==collection and r['image_sha256']==m['image_sha256'] and r['image_path']==m['image_path']
    for release in (META/'comparison_release_v1.json', REFERENCE/'release.json'):
        for rec in base.read(release)['artifacts']:
            assert base.sha(ROOT/rec['path']) == rec['sha256'], rec['path']
        paths.append(release)
    for target,split,_ in metadata.SOURCES.values():
        release=REV/target/'target_release_v1.json'
        for rec in base.read(release)['artifacts']:
            assert base.sha(ROOT/rec['path']) == rec['sha256'], rec['path']
        assert base.sha(base.OLD/'metadata_normalized.jsonl') == base.read(release)['metadata_normalized_sha256']
        assert base.sha(REV/split/'image_ready_split_manifest_v1.json') == base.read(release)['split_manifest_sha256']
        paths += [release, REV/split/'image_ready_split_manifest_v1.json']
    return [dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p)) for p in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=['dinov2_fullframe'], required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    configure_encoder(args.encoder)
    inputs = verify_inputs()
    out = (args.output or DEFAULT).resolve()
    if args.resume:
        assert out.is_dir() and not (out/'release.json').exists()
        old_config=base.read(out/'configuration.json')
        assert old_config['script_sha256']==base.sha(Path(__file__))
        assert old_config['inputs']==inputs
    else:
        out.mkdir(exist_ok=False)
    policy = base.read(POLICY)
    grids = metadata.candidates(base.effective_spec())
    qa = base.read(REV/'dinov2_fullframe_development_precheck_20260909_v4/verification.json')
    assert qa['passed'] and qa['implementation_sha256'] == base.sha(Path(__file__))
    configuration=dict(encoder=ENCODER,inputs=inputs,grids=grids,policy=policy,
        script_sha256=base.sha(Path(__file__)),precheck_sha256=base.sha(REV/'dinov2_fullframe_development_precheck_20260909_v4/verification.json'),
        python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,platform=platform.platform(),threads=3,
        source_scripts=[dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p)) for p in (Path(metadata.__file__),Path(base.__file__))],
        final_labels_parsed=False)
    if not args.resume:
        save(out/'configuration.json',configuration)
    runner = Runner(out,3)
    results=[]
    with threadpool_limits(limits=3):
        for sample in policy['samples']:
            for collection in policy['collections']:
                path=out/sample/collection/'comparison_results.json'
                if path.exists():
                    results.append(base.read(path))
                else:
                    results.append(runner.collection(sample,collection,grids,policy))
    fits=[r for r in base.lines(out/'execution_journal.jsonl') if r['stage']=='fit_completed']
    reused=[r for r in base.lines(out/'execution_journal.jsonl') if r['stage']=='fit_reused']
    save(out/'summary.json',dict(encoder=ENCODER,results=results,new_fits=len(fits),reused_fits=len(reused),current_process_seconds=time.perf_counter()-runner.started,
        total_fit_seconds=sum(r['seconds'] for r in fits),resumed=args.resume,final_scored=False))
    save(out/'release.json',dict(status='ENCODER_DEVELOPMENT_COMPARISON_COMPLETE_PENDING_INDEPENDENT_QA',final_scored=False,
        artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
    print(json.dumps(dict(stage='all_complete',fits=runner.new_fits)),flush=True)


if __name__ == '__main__':
    main()
